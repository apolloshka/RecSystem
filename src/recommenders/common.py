import math
import random
from collections import defaultdict, Counter


def jaccard(a: set, b: set) -> float:
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def build_user_to_groups(result_rows) -> dict:
    user_to_groups = defaultdict(set)
    for user_id, group_id in result_rows:
        user_to_groups[int(user_id)].add(int(group_id))
    return user_to_groups


def build_group_to_users(user_to_groups: dict) -> dict:
    group_to_users = defaultdict(set)
    for user_id, groups in user_to_groups.items():
        for group_id in groups:
            group_to_users[group_id].add(user_id)
    return group_to_users


def build_group_popularity(group_to_users: dict) -> dict:
    return {gid: len(users) for gid, users in group_to_users.items()}


def build_profile_members(profile_groups: set, group_to_users: dict) -> set:
    members = set()
    for g in profile_groups:
        members |= group_to_users.get(g, set())
    return members


def get_user_based_scores_for_profile(
    profile_groups: set,
    all_user_groups: dict,
    group_popularity: dict,
    top_k_users: int = 50,
    min_group_size: int = 3,
    min_similarity: float = 0.01,
    min_common_groups: int = 1,
    max_group_popularity: int = 1000000,
) -> dict:
    similar_users = []

    for other_user_id, other_groups in all_user_groups.items():
        common_groups = len(profile_groups & other_groups)
        if common_groups < min_common_groups:
            continue

        sim = jaccard(profile_groups, other_groups)
        if sim < min_similarity:
            continue

        similar_users.append((other_user_id, sim, other_groups, common_groups))

    similar_users.sort(key=lambda x: x[1], reverse=True)
    top_users = similar_users[:top_k_users]

    scores = defaultdict(float)

    for _, sim, other_groups, _ in top_users:
        for g in other_groups:
            if g in profile_groups:
                continue

            pop = group_popularity.get(g, 0)
            if pop < min_group_size:
                continue
            if pop > max_group_popularity:
                continue

            scores[g] += sim

    final_scores = {}
    for g, score in scores.items():
        popularity_penalty = math.log1p(group_popularity.get(g, 0))
        if popularity_penalty <= 0:
            continue
        final_scores[g] = score / popularity_penalty

    return final_scores


def get_item_based_scores_for_profile(
    profile_groups: set,
    group_to_users: dict,
    user_to_groups: dict,
    min_item_group_size: int = 20,
    min_item_similarity: float = 0.02,
    min_item_support: int = 1,
    max_item_candidates: int = 5000,
) -> dict:
    candidate_counts = Counter()

    for profile_group in profile_groups:
        users_a = group_to_users.get(profile_group, set())

        if len(users_a) < min_item_group_size:
            continue

        for user_id in users_a:
            for candidate_group in user_to_groups.get(user_id, set()):
                if candidate_group in profile_groups:
                    continue
                candidate_counts[candidate_group] += 1

    if not candidate_counts:
        return {}

    candidate_groups = [
        g for g, _ in candidate_counts.most_common(max_item_candidates)
        if len(group_to_users.get(g, set())) >= min_item_group_size
    ]

    scores = defaultdict(float)
    counts = defaultdict(int)

    for profile_group in profile_groups:
        users_a = group_to_users.get(profile_group, set())

        if len(users_a) < min_item_group_size:
            continue

        for candidate_group in candidate_groups:
            users_b = group_to_users.get(candidate_group, set())
            if not users_b:
                continue

            sim = jaccard(users_a, users_b)
            if sim < min_item_similarity:
                continue

            counts[candidate_group] += 1
            scores[candidate_group] += sim

    final_scores = {}
    for g in scores:
        if counts[g] < min_item_support:
            continue

        avg_score = scores[g] / counts[g]
        popularity_penalty = math.log1p(len(group_to_users.get(g, set())))
        if popularity_penalty <= 0:
            continue

        final_scores[g] = avg_score / popularity_penalty

    return final_scores


def extract_feature_dict_for_profile_candidate(
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


def sample_leave_one_out_targets(
    real_groups: set,
    max_positives_per_user: int = 3,
    random_seed: int | None = None,
) -> list:
    groups_list = list(real_groups)
    if random_seed is not None:
        rnd = random.Random(random_seed)
        rnd.shuffle(groups_list)
    else:
        random.shuffle(groups_list)
    return groups_list[:min(max_positives_per_user, len(groups_list))]


def sample_negative_groups(
    real_groups: set,
    user_based_scores: dict,
    item_based_scores: dict,
    all_group_ids: list,
    group_popularity: dict,
    negatives_per_positive: int = 4,
    hard_ratio: float = 0.3,
    min_group_size: int = 3,
    top_user_based_candidates: int = 50,
    top_item_based_candidates: int = 50,
) -> list:
    hard_needed = max(1, int(round(negatives_per_positive * hard_ratio)))
    random_needed = max(0, negatives_per_positive - hard_needed)

    # формируем пул сложных негатив групп, в которых не состоит пользователь, имеющихся в user-based рекомендациях для этого пользователя
    hard_pool_user = [
        g for g, _ in sorted(user_based_scores.items(), key=lambda x: x[1], reverse=True)
        if g not in real_groups
    ][-top_user_based_candidates:]

    # формируем пул сложных негатив групп, в которых не состоит пользователь, имеющихся в item-based рекомендациях для этого пользователя
    hard_pool_item = [
        g for g, _ in sorted(item_based_scores.items(), key=lambda x: x[1], reverse=True)
        if g not in real_groups
    ][-top_item_based_candidates:]

    hard_pool = list(set(hard_pool_user) | set(hard_pool_item))

    selected = []

    if hard_pool:
        selected.extend(random.sample(hard_pool, min(hard_needed, len(hard_pool))))

    # формируем пул негатив групп, в которых не состоит пользователь, имеющихся в user_groups 
    random_pool = [
        g for g in all_group_ids
        if g not in real_groups
        and g not in selected
        and group_popularity.get(g, 0) >= min_group_size
    ]

    if random_pool and len(selected) < negatives_per_positive:
        need = negatives_per_positive - len(selected)
        selected.extend(random.sample(random_pool, min(need, len(random_pool))))

    return selected