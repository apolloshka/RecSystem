import os
import random
import time
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor

from sklearn.model_selection import train_test_split

from src.db.clickhouse_client import (
    get_client,
    create_ml_dataset_table,
    truncate_ml_dataset,
    insert_ml_dataset,
)
from src.recommenders.common import (
    build_user_to_groups,
    build_group_to_users,
    build_group_popularity,
    build_profile_members,
    get_user_based_scores_for_profile,
    get_item_based_scores_for_profile,
    extract_feature_dict_for_profile_candidate,
    sample_leave_one_out_targets,
    sample_negative_groups,
)
from src.recommenders.config import (
    RANDOM_SEED,
    DATASET_BUILD_WORKERS,
    DATASET_BUILD_CHUNK_SIZE,
    DATASET_BUILD_BACKEND,
    MIN_USER_GROUPS,
    MAX_USERS_FOR_DATASET,
    MAX_POSITIVES_PER_USER,
    NEGATIVES_PER_POSITIVE,
    NEGATIVE_HARD_RATIO,
    TOP_K_SIMILAR_USERS,
    TOP_USER_BASED_CANDIDATES,
    TOP_ITEM_BASED_CANDIDATES,
    MIN_GROUP_SIZE,
    MIN_USER_SIMILARITY,
    MIN_COMMON_GROUPS,
    MAX_GROUP_POPULARITY,
    MIN_ITEM_GROUP_SIZE,
    MIN_ITEM_SIMILARITY,
    MIN_ITEM_SUPPORT,
    MAX_ITEM_CANDIDATES,
    ENABLE_ITEM_BASED,
    MODEL_FEATURE_NAMES,
    TRAIN_USERS_PATH,
    TEST_USERS_PATH,
)

feature_names = MODEL_FEATURE_NAMES
WORKER_USER_TO_GROUPS = None
WORKER_GROUP_TO_USERS = None
WORKER_GROUP_POPULARITY = None
WORKER_ALL_GROUP_IDS = None


def init_worker(user_to_groups, group_to_users, group_popularity, all_group_ids):
    global WORKER_USER_TO_GROUPS, WORKER_GROUP_TO_USERS, WORKER_GROUP_POPULARITY, WORKER_ALL_GROUP_IDS
    WORKER_USER_TO_GROUPS = user_to_groups
    WORKER_GROUP_TO_USERS = group_to_users
    WORKER_GROUP_POPULARITY = group_popularity
    WORKER_ALL_GROUP_IDS = all_group_ids


def build_rows_for_user(task):
    idx, total_users, user_id = task

    user_to_groups = WORKER_USER_TO_GROUPS
    group_to_users = WORKER_GROUP_TO_USERS
    group_popularity = WORKER_GROUP_POPULARITY
    all_group_ids = WORKER_ALL_GROUP_IDS

    dataset_rows = []
    positive_count = 0
    negative_count = 0

    real_groups = set(user_to_groups[user_id])
    targets = sample_leave_one_out_targets(
        real_groups=real_groups,
        max_positives_per_user=MAX_POSITIVES_PER_USER,
        random_seed=RANDOM_SEED + idx,
    )

    for target_group in targets:
        profile_groups = real_groups - {target_group}
        if len(profile_groups) < 2:
            continue

        profile_members = build_profile_members(
            profile_groups=profile_groups,
            group_to_users=group_to_users,
            target_user_id=user_id,
        )

        user_based_scores = get_user_based_scores_for_profile(
            profile_groups=profile_groups,
            all_user_groups=user_to_groups,
            group_popularity=group_popularity,
            top_k_users=TOP_K_SIMILAR_USERS,
            min_group_size=MIN_GROUP_SIZE,
            min_similarity=MIN_USER_SIMILARITY,
            min_common_groups=MIN_COMMON_GROUPS,
            max_group_popularity=MAX_GROUP_POPULARITY,
            target_user_id=user_id,
        )

        if ENABLE_ITEM_BASED:
            item_based_scores = get_item_based_scores_for_profile(
                profile_groups=profile_groups,
                group_to_users=group_to_users,
                user_to_groups=user_to_groups,
                min_item_group_size=MIN_ITEM_GROUP_SIZE,
                min_item_similarity=MIN_ITEM_SIMILARITY,
                min_item_support=MIN_ITEM_SUPPORT,
                max_item_candidates=MAX_ITEM_CANDIDATES,
                target_user_id=user_id,
            )
        else:
            item_based_scores = {}

        pos_features = extract_feature_dict_for_profile_candidate(
            profile_groups=profile_groups,
            candidate_group=target_group,
            user_based_scores=user_based_scores,
            item_based_scores=item_based_scores,
            group_to_users=group_to_users,
            profile_members=profile_members,
            target_user_id=user_id,
        )
        dataset_rows.append([user_id, target_group, 1] + [pos_features[f] for f in feature_names])
        positive_count += 1

        negative_groups = sample_negative_groups(
            real_groups=real_groups,
            user_based_scores=user_based_scores,
            item_based_scores=item_based_scores,
            all_group_ids=all_group_ids,
            group_popularity=group_popularity,
            negatives_per_positive=NEGATIVES_PER_POSITIVE,
            hard_ratio=NEGATIVE_HARD_RATIO,
            min_group_size=MIN_GROUP_SIZE,
            top_user_based_candidates=TOP_USER_BASED_CANDIDATES,
            top_item_based_candidates=TOP_ITEM_BASED_CANDIDATES,
        )

        for neg_group in negative_groups:
            neg_features = extract_feature_dict_for_profile_candidate(
                profile_groups=profile_groups,
                candidate_group=neg_group,
                user_based_scores=user_based_scores,
                item_based_scores=item_based_scores,
                group_to_users=group_to_users,
                profile_members=profile_members,
                target_user_id=user_id,
            )
            dataset_rows.append([user_id, neg_group, 0] + [neg_features[f] for f in feature_names])
            negative_count += 1

    return {
        "idx": idx,
        "total_users": total_users,
        "user_id": user_id,
        "rows": dataset_rows,
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def resolve_worker_count():
    if DATASET_BUILD_WORKERS and DATASET_BUILD_WORKERS > 0:
        return DATASET_BUILD_WORKERS
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def main():
    print("=== Building ML Dataset for train users only ===")
    random.seed(RANDOM_SEED)

    print("[INFO] Connecting to ClickHouse...")
    client = get_client()

    print("[INFO] Loading user_groups...")
    t0 = time.time()
    rows = client.query("SELECT user_id, group_id FROM user_groups").result_rows
    print(f"[INFO] user_groups loaded in {time.time() - t0:.2f} sec")

    print("[INFO] Building maps...")
    t0 = time.time()
    user_to_groups = build_user_to_groups(rows)
    group_to_users = build_group_to_users(user_to_groups)
    group_popularity = build_group_popularity(group_to_users)
    print(f"[INFO] maps built in {time.time() - t0:.2f} sec")

    print(f"[INFO] Users loaded: {len(user_to_groups)}")
    print(f"[INFO] Groups loaded: {len(group_to_users)}")

    eligible_users = [
        user_id
        for user_id, groups in user_to_groups.items()
        if len(groups) >= MIN_USER_GROUPS
    ]

    random.shuffle(eligible_users)
    eligible_users = eligible_users[:MAX_USERS_FOR_DATASET]
    print(f"[INFO] Eligible users total: {len(eligible_users)}")

    train_users, test_users = train_test_split(
        eligible_users,
        test_size=0.2,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    train_users_set = set(train_users)
    user_to_groups = {u: g for u, g in user_to_groups.items() if u in train_users_set}
    group_to_users = build_group_to_users(user_to_groups)
    group_popularity = build_group_popularity(group_to_users)
    all_group_ids = list(group_to_users.keys())

    print(f"[INFO] After filter: {len(user_to_groups)} users, {len(group_to_users)} groups")

    train_users = list(train_users)
    test_users = list(test_users)

    joblib.dump(train_users, TRAIN_USERS_PATH)
    joblib.dump(test_users, TEST_USERS_PATH)

    print(f"[INFO] Train users: {len(train_users)}")
    print(f"[INFO] Test users: {len(test_users)}")
    print(f"[INFO] Saved split to {TRAIN_USERS_PATH} and {TEST_USERS_PATH}")

    dataset_rows = []
    positive_count = 0
    negative_count = 0
    dataset_start = time.time()

    worker_count = resolve_worker_count()
    backend = (DATASET_BUILD_BACKEND or "process").strip().lower()
    if backend not in {"process", "thread"}:
        backend = "process"
    if os.name == "nt" and DATASET_BUILD_BACKEND == "process":
        print("[WARN] Windows + process backend may have slow startup due to worker spawn/serialization.")
        print("[WARN] Consider DATASET_BUILD_BACKEND='thread' or fewer workers.")

    print(f"[INFO] Building dataset with {worker_count} worker(s), backend={backend}")

    tasks = [(idx, len(train_users), user_id) for idx, user_id in enumerate(train_users, 1)]

    if worker_count == 1:
        init_worker(user_to_groups, group_to_users, group_popularity, all_group_ids)
        results_iter = map(build_rows_for_user, tasks)
        completed = 0
        for result in results_iter:
            completed += 1
            dataset_rows.extend(result["rows"])
            positive_count += result["positive_count"]
            negative_count += result["negative_count"]
            if completed % DATASET_BUILD_CHUNK_SIZE == 0 or completed == len(tasks):
                elapsed = time.time() - dataset_start
                avg_per_user = elapsed / completed
                print(
                    f"[PROGRESS] {completed}/{len(tasks)} users | "
                    f"rows={len(dataset_rows)} | avg={avg_per_user:.2f}s/user | "
                    f"elapsed={elapsed:.1f}s"
                )
    else:
        executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
        print("[INFO] Initializing worker pool...")

        pool_kwargs = {"max_workers": worker_count}
        if backend == "process":
            pool_kwargs["initializer"] = init_worker
            pool_kwargs["initargs"] = (user_to_groups, group_to_users, group_popularity, all_group_ids)
        else:
            init_worker(user_to_groups, group_to_users, group_popularity, all_group_ids)

        with executor_cls(**pool_kwargs) as executor:
            print("[INFO] Submitting user tasks...")
            futures = [executor.submit(build_rows_for_user, task) for task in tasks]
            print(f"[INFO] Submitted {len(futures)} tasks. Waiting for first completed user...")
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                dataset_rows.extend(result["rows"])
                positive_count += result["positive_count"]
                negative_count += result["negative_count"]
                if completed % DATASET_BUILD_CHUNK_SIZE == 0 or completed == len(tasks):
                    elapsed = time.time() - dataset_start
                    avg_per_user = elapsed / completed
                    print(
                        f"[PROGRESS] {completed}/{len(tasks)} users | "
                        f"rows={len(dataset_rows)} | avg={avg_per_user:.2f}s/user | "
                        f"elapsed={elapsed:.1f}s"
                    )

    print("\n=== Dataset summary ===")
    print(f"Rows total: {len(dataset_rows)}")
    print(f"Positive: {positive_count}")
    print(f"Negative: {negative_count}")
    print(f"Features: {feature_names}")
    print(f"Dataset built in {time.time() - dataset_start:.2f} sec")

    print("[INFO] Creating ml_dataset table...")
    t0 = time.time()
    create_ml_dataset_table(feature_names=feature_names, recreate=True)
    print(f"[INFO] Table created in {time.time() - t0:.2f} sec")

    print("[INFO] Truncating ml_dataset...")
    t0 = time.time()
    truncate_ml_dataset()
    print(f"[INFO] Table truncated in {time.time() - t0:.2f} sec")

    print("[INFO] Inserting rows...")
    t0 = time.time()
    insert_ml_dataset(dataset_rows, feature_names)
    print(f"[INFO] Inserted in {time.time() - t0:.2f} sec")

    print("ML dataset saved successfully.")


if __name__ == "__main__":
    main()