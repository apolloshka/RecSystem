import math
import random
from collections import defaultdict

from src.db.clickhouse_client import (
    get_client,
    create_ml_dataset_table,
    truncate_ml_dataset,
    insert_ml_dataset
)

# ----------------------------
# Настройки датасета
# ----------------------------
MAX_USERS = 300              # сколько пользователей брать для первой версии
POSITIVE_PER_USER = 5        # сколько positive-примеров брать на пользователя
NEGATIVE_PER_POSITIVE = 3    # сколько negative-примеров на один positive
MIN_USER_GROUPS = 3          # минимум групп у пользователя, чтобы брать его в датасет
RANDOM_SEED = 42


def jaccard(a, b):
    a = set(a)
    b = set(b)

    inter = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0.0

    return inter / union


def sample_users(user_to_groups, max_users, min_user_groups):
    eligible_users = [
        user_id for user_id, groups in user_to_groups.items()
        if len(groups) >= min_user_groups
    ]

    random.shuffle(eligible_users)
    return eligible_users[:max_users]


def build_indices():
    client = get_client()

    print("Loading user_groups from ClickHouse...")
    result = client.query("""
        SELECT user_id, group_id
        FROM user_groups
    """)

    user_to_groups = defaultdict(set)
    group_to_users = defaultdict(set)

    for user_id, group_id in result.result_rows:
        user_id = int(user_id)
        group_id = int(group_id)

        user_to_groups[user_id].add(group_id)
        group_to_users[group_id].add(user_id)

    print(f"✓ Users loaded: {len(user_to_groups)}")
    print(f"✓ Groups loaded: {len(group_to_users)}")

    return user_to_groups, group_to_users


def compute_user_based_features(target_user_id, candidate_group_id, user_to_groups, group_to_users):
    """
    Признаки по user-based логике для пары (user, candidate_group)
    """
    target_groups = user_to_groups[target_user_id]

    similar_users_in_group_count = 0
    sum_similarity_to_group_members = 0.0
    max_similarity_to_group_members = 0.0

    candidate_users = group_to_users.get(candidate_group_id, set())

    for other_user_id in candidate_users:
        if other_user_id == target_user_id:
            continue

        other_groups = user_to_groups.get(other_user_id, set())
        sim = jaccard(target_groups, other_groups)

        if sim > 0:
            similar_users_in_group_count += 1
            sum_similarity_to_group_members += sim
            if sim > max_similarity_to_group_members:
                max_similarity_to_group_members = sim

    group_popularity = len(candidate_users)
    if group_popularity > 0:
        user_cf_score = sum_similarity_to_group_members / math.log(1 + group_popularity)
    else:
        user_cf_score = 0.0

    return (
        similar_users_in_group_count,
        sum_similarity_to_group_members,
        max_similarity_to_group_members,
        user_cf_score
    )


def compute_item_based_features(target_user_id, candidate_group_id, user_to_groups, group_to_users):
    """
    Признаки по item-based логике для пары (user, candidate_group)
    """
    target_groups = user_to_groups[target_user_id]
    candidate_users = group_to_users.get(candidate_group_id, set())

    max_item_similarity = 0.0
    sum_item_similarity = 0.0
    similar_user_groups_count = 0

    for user_group_id in target_groups:
        users_of_user_group = group_to_users.get(user_group_id, set())
        sim = jaccard(candidate_users, users_of_user_group)

        if sim > 0:
            similar_user_groups_count += 1
            sum_item_similarity += sim
            if sim > max_item_similarity:
                max_item_similarity = sim

    group_popularity = len(candidate_users)

    if group_popularity > 0 and similar_user_groups_count > 0:
        item_cf_score = (sum_item_similarity / similar_user_groups_count) / math.log(1 + group_popularity)
    else:
        item_cf_score = 0.0

    return (
        max_item_similarity,
        sum_item_similarity,
        similar_user_groups_count,
        item_cf_score
    )


def build_feature_row(target_user_id, candidate_group_id, label, user_to_groups, group_to_users):
    target_groups = user_to_groups[target_user_id]
    candidate_users = group_to_users.get(candidate_group_id, set())

    user_group_count = len(target_groups)
    group_popularity = len(candidate_users)
    log_group_popularity = math.log(1 + group_popularity)

    (
        similar_users_in_group_count,
        sum_similarity_to_group_members,
        max_similarity_to_group_members,
        user_cf_score
    ) = compute_user_based_features(
        target_user_id,
        candidate_group_id,
        user_to_groups,
        group_to_users
    )

    (
        max_item_similarity,
        sum_item_similarity,
        similar_user_groups_count,
        item_cf_score
    ) = compute_item_based_features(
        target_user_id,
        candidate_group_id,
        user_to_groups,
        group_to_users
    )

    return (
        int(target_user_id),
        int(candidate_group_id),
        int(label),

        int(user_group_count),
        int(group_popularity),
        float(log_group_popularity),

        int(similar_users_in_group_count),
        float(sum_similarity_to_group_members),
        float(max_similarity_to_group_members),
        float(user_cf_score),

        float(max_item_similarity),
        float(sum_item_similarity),
        int(similar_user_groups_count),
        float(item_cf_score)
    )


def build_dataset_rows(user_to_groups, group_to_users):
    all_group_ids = list(group_to_users.keys())
    rows = []

    sampled_users = sample_users(
        user_to_groups=user_to_groups,
        max_users=MAX_USERS,
        min_user_groups=MIN_USER_GROUPS
    )

    print(f"Users selected for ML dataset: {len(sampled_users)}")

    for idx, user_id in enumerate(sampled_users, start=1):
        user_groups = list(user_to_groups[user_id])

        # ----------------------------
        # Positive examples
        # ----------------------------
        random.shuffle(user_groups)
        positive_groups = user_groups[:POSITIVE_PER_USER]

        for candidate_group_id in positive_groups:
            row = build_feature_row(
                target_user_id=user_id,
                candidate_group_id=candidate_group_id,
                label=1,
                user_to_groups=user_to_groups,
                group_to_users=group_to_users
            )
            rows.append(row)

        # ----------------------------
        # Negative examples
        # ----------------------------
        negatives_needed = len(positive_groups) * NEGATIVE_PER_POSITIVE
        negative_candidates = []

        user_group_set = user_to_groups[user_id]

        for gid in all_group_ids:
            if gid not in user_group_set:
                negative_candidates.append(gid)

        random.shuffle(negative_candidates)
        negative_groups = negative_candidates[:negatives_needed]

        for candidate_group_id in negative_groups:
            row = build_feature_row(
                target_user_id=user_id,
                candidate_group_id=candidate_group_id,
                label=0,
                user_to_groups=user_to_groups,
                group_to_users=group_to_users
            )
            rows.append(row)

        if idx % 20 == 0 or idx == len(sampled_users):
            print(f"processed users: {idx}/{len(sampled_users)} | dataset rows: {len(rows)}")

    return rows


def main():
    random.seed(RANDOM_SEED)

    print("Preparing ml_dataset table...")
    create_ml_dataset_table()
    truncate_ml_dataset()

    user_to_groups, group_to_users = build_indices()

    print("Building ML dataset rows...")
    rows = build_dataset_rows(user_to_groups, group_to_users)

    print(f"Total rows prepared: {len(rows)}")

    print("Saving ML dataset to ClickHouse...")
    insert_ml_dataset(rows)

    print("✓ ML dataset build finished")


if __name__ == "__main__":
    main()