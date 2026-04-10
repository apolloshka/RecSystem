import math
import random
import time
from collections import defaultdict

from src.db.clickhouse_client import (
    get_client,
    create_ml_dataset_table,
    truncate_ml_dataset,
    insert_ml_dataset,
)

print("=== Building ML Dataset for many users ===")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# -----------------------------
# Параметры
# -----------------------------
MIN_USER_GROUPS = 5
MAX_USERS_FOR_DATASET = 20
NEGATIVES_PER_POSITIVE = 2

TOP_K_SIMILAR_USERS = 10
TOP_USER_BASED_CANDIDATES = 10
TOP_ITEM_BASED_CANDIDATES = 10

MIN_GROUP_SIZE = 3
MIN_USER_SIMILARITY = 0.01
MIN_COMMON_GROUPS = 1

MIN_ITEM_GROUP_SIZE = 20
MIN_ITEM_SIMILARITY = 0.02
MIN_ITEM_SUPPORT = 1

ENABLE_ITEM_BASED = True

# -----------------------------
# Вспомогательные функции
# -----------------------------
def jaccard(a: set, b: set) -> float:
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def get_user_based_scores(
    profile_groups: set,
    all_user_groups: dict,
    group_popularity: dict,
) -> dict:
    """
    User-based CF для произвольного профиля пользователя.
    Возвращает dict[group_id] = score
    """
    similar_users = []

    for other_user_id, other_groups in all_user_groups.items():
        common_groups = len(profile_groups & other_groups)
        if common_groups < MIN_COMMON_GROUPS:
            continue

        sim = jaccard(profile_groups, other_groups)
        if sim < MIN_USER_SIMILARITY:
            continue

        similar_users.append((other_user_id, sim, other_groups, common_groups))

    similar_users.sort(key=lambda x: x[1], reverse=True)
    top_users = similar_users[:TOP_K_SIMILAR_USERS]

    scores = defaultdict(float)

    for _, sim, other_groups, _ in top_users:
        for g in other_groups:
            if g in profile_groups:
                continue

            pop = group_popularity.get(g, 0)
            if pop < MIN_GROUP_SIZE:
                continue

            scores[g] += sim

    final_scores = {}
    for g, score in scores.items():
        popularity_penalty = math.log1p(group_popularity.get(g, 0))
        if popularity_penalty <= 0:
            continue
        final_scores[g] = score / popularity_penalty

    return final_scores


def get_item_based_scores(
    profile_groups: set,
    group_to_users: dict,
) -> dict:
    """
    Item-based CF для произвольного профиля пользователя.
    Возвращает dict[group_id] = score
    """
    scores = defaultdict(float)
    counts = defaultdict(int)

    all_groups_items = list(group_to_users.items())

    for profile_group in profile_groups:
        users_a = group_to_users.get(profile_group, set())

        if len(users_a) < MIN_ITEM_GROUP_SIZE:
            continue

        for candidate_group, users_b in all_groups_items:
            if candidate_group in profile_groups:
                continue

            if len(users_b) < MIN_ITEM_GROUP_SIZE:
                continue

            sim = jaccard(users_a, users_b)
            if sim < MIN_ITEM_SIMILARITY:
                continue

            scores[candidate_group] += sim
            counts[candidate_group] += 1

    final_scores = {}
    for g in scores:
        if counts[g] < MIN_ITEM_SUPPORT:
            continue

        avg_score = scores[g] / counts[g]
        popularity_penalty = math.log1p(len(group_to_users.get(g, set())))
        if popularity_penalty <= 0:
            continue

        final_scores[g] = avg_score / popularity_penalty

    return final_scores


def build_profile_members(profile_groups: set, group_to_users: dict) -> set:
    members = set()
    for g in profile_groups:
        members |= group_to_users.get(g, set())
    return members


def extract_features(
    profile_groups: set,
    candidate_group: int,
    user_based_scores: dict,
    item_based_scores: dict,
    group_to_users: dict,
    profile_members: set,
) -> dict:
    candidate_users = group_to_users.get(candidate_group, set())
    group_pop = len(candidate_users)

    similarities = []
    for g in profile_groups:
        users_a = group_to_users.get(g, set())
        if not users_a or not candidate_users:
            continue
        sim = jaccard(users_a, candidate_users)
        if sim > 0:
            similarities.append(sim)

    max_group_similarity = max(similarities) if similarities else 0.0
    sum_group_similarity = sum(similarities) if similarities else 0.0
    common_members_with_profile = len(candidate_users & profile_members) if candidate_users else 0

    return {
        "group_popularity": float(group_pop),
        "log_group_popularity": float(math.log1p(group_pop)),
        "user_based_score": float(user_based_scores.get(candidate_group, 0.0)),
        "item_based_score": float(item_based_scores.get(candidate_group, 0.0)),
        "max_group_similarity": float(max_group_similarity),
        "sum_group_similarity": float(sum_group_similarity),
        "common_members_with_profile": float(common_members_with_profile),
        "is_in_both_recs": float(
            1.0 if candidate_group in user_based_scores and candidate_group in item_based_scores else 0.0
        ),
    }


# -----------------------------
# Загрузка данных
# -----------------------------
print("[INFO] Connecting to ClickHouse...")
client = get_client()

print("[INFO] Loading user_groups...")
t0 = time.time()

user_to_groups = defaultdict(set)
for user_id, group_id in client.query("SELECT user_id, group_id FROM user_groups").result_rows:
    user_to_groups[int(user_id)].add(int(group_id))

print(f"[INFO] user_groups loaded in {time.time() - t0:.2f} sec")

print("[INFO] Building group_to_users...")
t0 = time.time()

group_to_users = defaultdict(set)
for user_id, groups in user_to_groups.items():
    for group_id in groups:
        group_to_users[group_id].add(user_id)

print(f"[INFO] group_to_users built in {time.time() - t0:.2f} sec")

group_popularity = {gid: len(users) for gid, users in group_to_users.items()}

print(f"[INFO] Users loaded: {len(user_to_groups)}")
print(f"[INFO] Groups loaded: {len(group_to_users)}")

eligible_users = [
    user_id for user_id, groups in user_to_groups.items()
    if len(groups) >= MIN_USER_GROUPS
]

random.shuffle(eligible_users)
eligible_users = eligible_users[:MAX_USERS_FOR_DATASET]

print(f"[INFO] Eligible users: {len(eligible_users)}")

popular_negative_pool = [
    gid for gid, pop in group_popularity.items()
    if pop >= MIN_GROUP_SIZE
]

print(f"[INFO] Popular negative pool size: {len(popular_negative_pool)}")

# -----------------------------
# Формирование датасета
# -----------------------------
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

rows = []
positive_count = 0
negative_count = 0

dataset_start = time.time()

for idx, user_id in enumerate(eligible_users, 1):
    user_start = time.time()
    print(f"\n[USER {idx}/{len(eligible_users)}] Start user_id={user_id}")

    real_groups = set(user_to_groups[user_id])
    print(f"[USER {idx}] real_groups={len(real_groups)}")

    if len(real_groups) < MIN_USER_GROUPS:
        print(f"[USER {idx}] skipped: too few groups")
        continue

    hidden_positive = random.choice(list(real_groups))
    profile_groups = real_groups - {hidden_positive}

    if len(profile_groups) < 2:
        print(f"[USER {idx}] skipped: profile too small after holdout")
        continue

    print(f"[USER {idx}] hidden_positive={hidden_positive}")
    print(f"[USER {idx}] profile_groups={len(profile_groups)}")

    print(f"[USER {idx}] building profile members...")
    t0 = time.time()
    profile_members = build_profile_members(profile_groups, group_to_users)
    print(f"[USER {idx}] profile_members={len(profile_members)} built in {time.time() - t0:.2f} sec")

    print(f"[USER {idx}] computing user-based scores...")
    t0 = time.time()
    user_based_scores = get_user_based_scores(
        profile_groups=profile_groups,
        all_user_groups=user_to_groups,
        group_popularity=group_popularity,
    )
    print(
        f"[USER {idx}] user-based candidates={len(user_based_scores)} "
        f"in {time.time() - t0:.2f} sec"
    )

    if ENABLE_ITEM_BASED:
        print(f"[USER {idx}] computing item-based scores...")
        t0 = time.time()
        item_based_scores = get_item_based_scores(
            profile_groups=profile_groups,
            group_to_users=group_to_users,
        )
        print(
            f"[USER {idx}] item-based candidates={len(item_based_scores)} "
            f"in {time.time() - t0:.2f} sec"
        )
    else:
        item_based_scores = {}
        print(f"[USER {idx}] item-based disabled")

    print(f"[USER {idx}] building positive example...")
    t0 = time.time()
    pos_features = extract_features(
        profile_groups=profile_groups,
        candidate_group=hidden_positive,
        user_based_scores=user_based_scores,
        item_based_scores=item_based_scores,
        group_to_users=group_to_users,
        profile_members=profile_members,
    )

    rows.append([user_id, hidden_positive, 1] + [pos_features[f] for f in feature_names])
    positive_count += 1
    print(f"[USER {idx}] positive added in {time.time() - t0:.2f} sec")

    print(f"[USER {idx}] selecting negatives...")
    t0 = time.time()

    user_based_candidates = [
        g for g, _ in sorted(user_based_scores.items(), key=lambda x: x[1], reverse=True)
        if g not in real_groups
    ][:TOP_USER_BASED_CANDIDATES]

    item_based_candidates = [
        g for g, _ in sorted(item_based_scores.items(), key=lambda x: x[1], reverse=True)
        if g not in real_groups
    ][:TOP_ITEM_BASED_CANDIDATES]

    combined_negative_candidates = list(set(user_based_candidates) | set(item_based_candidates))

    if len(combined_negative_candidates) < NEGATIVES_PER_POSITIVE:
        extra_needed = NEGATIVES_PER_POSITIVE - len(combined_negative_candidates)
        random_candidates = [
            g for g in popular_negative_pool
            if g not in real_groups and g not in combined_negative_candidates
        ]
        if random_candidates:
            combined_negative_candidates.extend(
                random.sample(random_candidates, min(extra_needed, len(random_candidates)))
            )

    print(
        f"[USER {idx}] negatives pool={len(combined_negative_candidates)} "
        f"prepared in {time.time() - t0:.2f} sec"
    )

    if not combined_negative_candidates:
        print(f"[USER {idx}] skipped negatives: empty candidate pool")
        continue

    selected_negatives = random.sample(
        combined_negative_candidates,
        min(NEGATIVES_PER_POSITIVE, len(combined_negative_candidates))
    )

    print(f"[USER {idx}] selected_negatives={selected_negatives}")

    for neg_group in selected_negatives:
        t0 = time.time()
        neg_features = extract_features(
            profile_groups=profile_groups,
            candidate_group=neg_group,
            user_based_scores=user_based_scores,
            item_based_scores=item_based_scores,
            group_to_users=group_to_users,
            profile_members=profile_members,
        )

        rows.append([user_id, neg_group, 0] + [neg_features[f] for f in feature_names])
        negative_count += 1
        print(
            f"[USER {idx}] negative added: group={neg_group} "
            f"in {time.time() - t0:.2f} sec"
        )

    print(
        f"[USER {idx}] done in {time.time() - user_start:.2f} sec | "
        f"total_rows={len(rows)}"
    )

print("\n=== Dataset summary ===")
print(f"Rows total: {len(rows)}")
print(f"Positive: {positive_count}")
print(f"Negative: {negative_count}")
print(f"Features: {feature_names}")
print(f"Dataset built in {time.time() - dataset_start:.2f} sec")

# -----------------------------
# Сохранение
# -----------------------------
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
insert_ml_dataset(rows, feature_names)
print(f"[INFO] Inserted in {time.time() - t0:.2f} sec")

print("✅ ML dataset saved successfully.")