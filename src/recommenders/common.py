import math
import random
from collections import Counter, defaultdict


def jaccard(a: set, b: set) -> float:
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def build_user_to_groups(result_rows) -> dict[int, set[int]]:
    user_to_groups = defaultdict(set)
    for user_id, group_id in result_rows:
        user_to_groups[int(user_id)].add(int(group_id))
    return user_to_groups


def build_group_to_users(user_to_groups: dict[int, set[int]]) -> dict[int, set[int]]:
    group_to_users = defaultdict(set)
    for user_id, groups in user_to_groups.items():
        for group_id in groups:
            group_to_users[group_id].add(user_id)
    return group_to_users


def build_group_popularity(group_to_users: dict[int, set[int]]) -> dict[int, int]:
    return {gid: len(users) for gid, users in group_to_users.items()}


def build_profile_members(
    profile_groups: set[int],
    group_to_users: dict[int, set[int]],
    target_user_id: int | None = None,
) -> set[int]:
    members = set()
    for group_id in profile_groups:
        users = set(group_to_users.get(group_id, set()))
        if target_user_id is not None:
            users.discard(target_user_id)
        members |= users
    return members


def get_user_based_scores_for_profile(
    profile_groups: set[int],
    all_user_groups: dict[int, set[int]],
    group_popularity: dict[int, int],
    top_k_users: int = 50,
    min_group_size: int = 3,
    min_similarity: float = 0.01,
    min_common_groups: int = 1,
    max_group_popularity: int = 1_000_000,
    target_user_id: int | None = None,
) -> dict[int, float]:
    """
    User-based CF.
    Важно: target_user_id исключается из кандидатов похожих пользователей,
    чтобы юзер не "находил сам себя".
    """
    similar_users = []

    for other_user_id, other_groups in all_user_groups.items():
        if target_user_id is not None and other_user_id == target_user_id:
            continue

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
        for group_id in other_groups:
            if group_id in profile_groups:
                continue

            pop = group_popularity.get(group_id, 0)
            if pop < min_group_size:
                continue
            if pop > max_group_popularity:
                continue

            scores[group_id] += sim

    final_scores = {}
    for group_id, score in scores.items():
        popularity_penalty = math.log1p(group_popularity.get(group_id, 0))
        if popularity_penalty <= 0:
            continue
        final_scores[group_id] = score / popularity_penalty

    return final_scores


def get_item_based_scores_for_profile(
    profile_groups: set[int],
    group_to_users: dict[int, set[int]],
    user_to_groups: dict[int, set[int]],
    min_item_group_size: int = 20,
    min_item_similarity: float = 0.02,
    min_item_support: int = 1,
    max_item_candidates: int = 5000,
    target_user_id: int | None = None,
) -> dict[int, float]:
    """
    Item-based CF по схожести аудиторий групп.
    Важно: target_user_id исключается из аудиторий всех групп при расчете.
    """
    candidate_counts = Counter()

    for profile_group in profile_groups:
        users_a = set(group_to_users.get(profile_group, set()))
        if target_user_id is not None:
            users_a.discard(target_user_id)

        if len(users_a) < min_item_group_size:
            continue

        for user_id in users_a:
            for candidate_group in user_to_groups.get(user_id, set()):
                if candidate_group in profile_groups:
                    continue
                candidate_counts[candidate_group] += 1

    if not candidate_counts:
        return {}

    def group_size_without_target(group_id: int) -> int:
        users = set(group_to_users.get(group_id, set()))
        if target_user_id is not None:
            users.discard(target_user_id)
        return len(users)

    candidate_groups = [
        group_id
        for group_id, _ in candidate_counts.most_common(max_item_candidates)
        if group_size_without_target(group_id) >= min_item_group_size
    ]

    scores = defaultdict(float)
    counts = defaultdict(int)

    for profile_group in profile_groups:
        users_a = set(group_to_users.get(profile_group, set()))
        if target_user_id is not None:
            users_a.discard(target_user_id)

        if len(users_a) < min_item_group_size:
            continue

        for candidate_group in candidate_groups:
            users_b = set(group_to_users.get(candidate_group, set()))
            if target_user_id is not None:
                users_b.discard(target_user_id)

            if not users_b:
                continue

            sim = jaccard(users_a, users_b)
            if sim < min_item_similarity:
                continue

            counts[candidate_group] += 1
            scores[candidate_group] += sim

    final_scores = {}
    for group_id in scores:
        if counts[group_id] < min_item_support:
            continue

        avg_score = scores[group_id] / counts[group_id]
        popularity_penalty = math.log1p(group_size_without_target(group_id))
        if popularity_penalty <= 0:
            continue

        final_scores[group_id] = avg_score / popularity_penalty

    return final_scores


def extract_feature_dict_for_profile_candidate(
    profile_groups: set[int],
    candidate_group: int,
    user_based_scores: dict[int, float],
    item_based_scores: dict[int, float],
    group_to_users: dict[int, set[int]],
    profile_members: set[int],
    target_user_id: int | None = None,
) -> dict[str, float]:
    """
    Безопасное извлечение признаков:
    target_user_id исключается из candidate audience и из profile audiences.
    """
    candidate_users = set(group_to_users.get(candidate_group, set()))
    if target_user_id is not None:
        candidate_users.discard(target_user_id)

    group_pop = len(candidate_users)

    similarities = []
    for profile_group in profile_groups:
        users_a = set(group_to_users.get(profile_group, set()))
        if target_user_id is not None:
            users_a.discard(target_user_id)

        if not users_a or not candidate_users:
            continue

        sim = jaccard(users_a, candidate_users)
        if sim > 0:
            similarities.append(sim)

    profile_members_wo_user = set(profile_members)
    if target_user_id is not None:
        profile_members_wo_user.discard(target_user_id)

    common_members_with_profile = len(candidate_users & profile_members_wo_user) if candidate_users else 0

    return {
        "group_popularity": float(group_pop),
        "log_group_popularity": float(math.log1p(group_pop)),
        "user_based_score": float(user_based_scores.get(candidate_group, 0.0)),
        "item_based_score": float(item_based_scores.get(candidate_group, 0.0)),
        "max_group_similarity": float(max(similarities) if similarities else 0.0),
        "sum_group_similarity": float(sum(similarities) if similarities else 0.0),
        "common_members_with_profile": float(common_members_with_profile),
        "is_in_both_recs": float(
            1.0 if candidate_group in user_based_scores and candidate_group in item_based_scores else 0.0
        ),
    }


def sample_leave_one_out_targets(
    real_groups: set[int],
    max_positives_per_user: int = 3,
    random_seed: int | None = None,
) -> list[int]:
    groups_list = list(real_groups)

    if random_seed is not None:
        rnd = random.Random(random_seed)
        rnd.shuffle(groups_list)
    else:
        random.shuffle(groups_list)

    return groups_list[: min(max_positives_per_user, len(groups_list))]


def sample_negative_groups(
    real_groups: set[int],
    user_based_scores: dict[int, float],
    item_based_scores: dict[int, float],
    all_group_ids: list[int],
    group_popularity: dict[int, int],
    negatives_per_positive: int = 4,
    hard_ratio: float = 0.3,
    min_group_size: int = 3,
    top_user_based_candidates: int = 50,
    top_item_based_candidates: int = 50,
) -> list[int]:
    """
    Важно: hard negatives берутся из ТОП-а, а не из хвоста.
    """
    hard_needed = max(1, int(round(negatives_per_positive * hard_ratio)))
    selected = []

    hard_pool_user = [
        group_id
        for group_id, _ in sorted(user_based_scores.items(), key=lambda x: x[1], reverse=True)
        if group_id not in real_groups and group_popularity.get(group_id, 0) >= min_group_size
    ][:top_user_based_candidates]

    hard_pool_item = [
        group_id
        for group_id, _ in sorted(item_based_scores.items(), key=lambda x: x[1], reverse=True)
        if group_id not in real_groups and group_popularity.get(group_id, 0) >= min_group_size
    ][:top_item_based_candidates]

    hard_pool = list(set(hard_pool_user) | set(hard_pool_item))

    if hard_pool:
        selected.extend(random.sample(hard_pool, min(hard_needed, len(hard_pool))))

    random_pool = [
        group_id
        for group_id in all_group_ids
        if group_id not in real_groups
        and group_id not in selected
        and group_popularity.get(group_id, 0) >= min_group_size
    ]

    need = negatives_per_positive - len(selected)
    if need > 0 and random_pool:
        selected.extend(random.sample(random_pool, min(need, len(random_pool))))

    return selected