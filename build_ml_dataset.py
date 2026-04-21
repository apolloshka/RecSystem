import random
import time
import joblib

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

print("=== Building ML Dataset for train users only ===")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

MIN_USER_GROUPS = 5
MAX_USERS_FOR_DATASET = 500

MAX_POSITIVES_PER_USER = 3
NEGATIVES_PER_POSITIVE = 4

TOP_K_SIMILAR_USERS = 20
TOP_USER_BASED_CANDIDATES = 50
TOP_ITEM_BASED_CANDIDATES = 50

MIN_GROUP_SIZE = 3
MIN_USER_SIMILARITY = 0.01
MIN_COMMON_GROUPS = 1
MAX_GROUP_POPULARITY = 1_000_000

MIN_ITEM_GROUP_SIZE = 20
MIN_ITEM_SIMILARITY = 0.02
MIN_ITEM_SUPPORT = 1
MAX_ITEM_CANDIDATES = 3000

ENABLE_ITEM_BASED = True

feature_names = [
    "group_popularity",
    "log_group_popularity",
    "user_based_score",
    "item_based_score",
    "max_group_similarity",
    "sum_group_similarity",
    "common_members_with_profile",
    "is_in_both_recs",
]

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
all_group_ids = list(group_to_users.keys())
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

# ФИКС: перестраиваем словари ТОЛЬКО на train_users
train_users_set = set(train_users)

user_to_groups = {u: g for u, g in user_to_groups.items() if u in train_users_set}
group_to_users = build_group_to_users(user_to_groups)
group_popularity = build_group_popularity(group_to_users)
all_group_ids = list(group_to_users.keys())

print(f"[INFO] After filter: {len(user_to_groups)} users, {len(group_to_users)} groups")

train_users = list(train_users)
test_users = list(test_users)

joblib.dump(train_users, "train_users.pkl")
joblib.dump(test_users, "test_users.pkl")

print(f"[INFO] Train users: {len(train_users)}")
print(f"[INFO] Test users: {len(test_users)}")
print("[INFO] Saved split to train_users.pkl and test_users.pkl")

dataset_rows = []
positive_count = 0
negative_count = 0
dataset_start = time.time()

for idx, user_id in enumerate(train_users, 1):
    user_start = time.time()
    real_groups = set(user_to_groups[user_id])

    print(f"\n[USER {idx}/{len(train_users)}] user_id={user_id} real_groups={len(real_groups)}")

    targets = sample_leave_one_out_targets(
        real_groups=real_groups,
        max_positives_per_user=MAX_POSITIVES_PER_USER,
        random_seed=RANDOM_SEED + idx,
    )

    print(f"[USER {idx}] positive targets count={len(targets)}")

    for target_group in targets:
        profile_groups = real_groups - {target_group}

        if len(profile_groups) < 2:
            print(f"[USER {idx}] skipped target={target_group}: profile too small")
            continue

        print(f"[USER {idx}] target_group={target_group} profile_size={len(profile_groups)}")

        t0 = time.time()
        profile_members = build_profile_members(
            profile_groups=profile_groups,
            group_to_users=group_to_users,
            target_user_id=user_id,
        )
        print(f"[USER {idx}] profile_members={len(profile_members)} built in {time.time() - t0:.2f} sec")

        t0 = time.time()
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
        print(f"[USER {idx}] user-based candidates={len(user_based_scores)} in {time.time() - t0:.2f} sec")

        if ENABLE_ITEM_BASED:
            t0 = time.time()
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
            print(f"[USER {idx}] item-based candidates={len(item_based_scores)} in {time.time() - t0:.2f} sec")
        else:
            item_based_scores = {}
            print(f"[USER {idx}] item-based disabled")

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
            hard_ratio=0.5,
            min_group_size=MIN_GROUP_SIZE,
            top_user_based_candidates=TOP_USER_BASED_CANDIDATES,
            top_item_based_candidates=TOP_ITEM_BASED_CANDIDATES,
        )

        print(f"[USER {idx}] sampled negatives={len(negative_groups)}")

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

    print(f"[USER {idx}] done in {time.time() - user_start:.2f} sec | total_rows={len(dataset_rows)}")

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