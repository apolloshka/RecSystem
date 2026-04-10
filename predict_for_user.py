import math
import time
import joblib
import pandas as pd
from collections import defaultdict

from src.db.clickhouse_client import get_client, get_group_name_from_db, save_group_name

print("=== ML Predictor for my_groups ===")

client = get_client()

# -----------------------------
# 1. Загружаем модель
# -----------------------------
model = joblib.load("recommendation_model.pkl")
scaler = joblib.load("feature_scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

print(f"Loaded feature_names: {feature_names}")

# -----------------------------
# 2. Загружаем данные
# -----------------------------
my_groups = {int(row[0]) for row in client.query("SELECT group_id FROM my_groups").result_rows}
print(f"My groups: {len(my_groups)}")

user_to_groups = defaultdict(set)
for user_id, group_id in client.query("SELECT user_id, group_id FROM user_groups").result_rows:
    user_to_groups[int(user_id)].add(int(group_id))

group_to_users = defaultdict(set)
for user_id, groups in user_to_groups.items():
    for group_id in groups:
        group_to_users[group_id].add(user_id)

group_popularity = {gid: len(users) for gid, users in group_to_users.items()}

print(f"Loaded users: {len(user_to_groups)}")
print(f"Loaded groups: {len(group_to_users)}")

# -----------------------------
# 3. Те же функции, что в build_ml_dataset.py
# -----------------------------
MIN_GROUP_SIZE = 3
MIN_USER_SIMILARITY = 0.01
MIN_COMMON_GROUPS = 1

MIN_ITEM_GROUP_SIZE = 20
MIN_ITEM_SIMILARITY = 0.02
MIN_ITEM_SUPPORT = 1

TOP_K_SIMILAR_USERS = 10
TOP_CANDIDATES_FROM_USER_BASED = 300
TOP_CANDIDATES_FROM_ITEM_BASED = 300
MAX_ITEM_CANDIDATES = 5000
ENABLE_ITEM_BASED = True


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
    user_to_groups: dict,
) -> dict:
    from collections import Counter

    candidate_counts = Counter()

    for profile_group in profile_groups:
        users_a = group_to_users.get(profile_group, set())

        if len(users_a) < MIN_ITEM_GROUP_SIZE:
            continue

        for user_id in users_a:
            for candidate_group in user_to_groups.get(user_id, set()):
                if candidate_group in profile_groups:
                    continue
                candidate_counts[candidate_group] += 1

    if not candidate_counts:
        return {}

    candidate_groups = [
        g for g, _ in candidate_counts.most_common(MAX_ITEM_CANDIDATES)
        if len(group_to_users.get(g, set())) >= MIN_ITEM_GROUP_SIZE
    ]

    scores = defaultdict(float)
    counts = defaultdict(int)

    for profile_group in profile_groups:
        users_a = group_to_users.get(profile_group, set())

        if len(users_a) < MIN_ITEM_GROUP_SIZE:
            continue

        for candidate_group in candidate_groups:
            users_b = group_to_users.get(candidate_group, set())

            if not users_b:
                continue

            sim = jaccard(users_a, users_b)
            if sim < MIN_ITEM_SIMILARITY:
                continue

            counts[candidate_group] += 1
            scores[candidate_group] += sim

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


def extract_feature_dict(
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

    full_feature_dict = {
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

    return {f: full_feature_dict.get(f, 0.0) for f in feature_names}


def extract_features_df(
    profile_groups: set,
    candidate_group: int,
    user_based_scores: dict,
    item_based_scores: dict,
    group_to_users: dict,
    profile_members: set,
) -> pd.DataFrame:
    feature_dict = extract_feature_dict(
        profile_groups=profile_groups,
        candidate_group=candidate_group,
        user_based_scores=user_based_scores,
        item_based_scores=item_based_scores,
        group_to_users=group_to_users,
        profile_members=profile_members,
    )
    return pd.DataFrame([feature_dict], columns=feature_names)


# -----------------------------
# 4. Считаем признаки для твоего профиля
# -----------------------------
print("Computing user-based scores for my_groups...")
t0 = time.time()
user_based_scores = get_user_based_scores(
    profile_groups=my_groups,
    all_user_groups=user_to_groups,
    group_popularity=group_popularity,
)
print(f"user-based candidates: {len(user_based_scores)} in {time.time() - t0:.2f} sec")

if ENABLE_ITEM_BASED:
    print("Computing item-based scores for my_groups...")
    t0 = time.time()
    item_based_scores = get_item_based_scores(
        profile_groups=my_groups,
        group_to_users=group_to_users,
        user_to_groups=user_to_groups,
    )
    print(f"item-based candidates: {len(item_based_scores)} in {time.time() - t0:.2f} sec")
else:
    item_based_scores = {}
    print("item-based disabled")

profile_members = build_profile_members(my_groups, group_to_users)
print(f"profile_members: {len(profile_members)}")

# Кандидаты для предсказания
candidates = (
    set(sorted(user_based_scores, key=user_based_scores.get, reverse=True)[:TOP_CANDIDATES_FROM_USER_BASED]) |
    set(sorted(item_based_scores, key=item_based_scores.get, reverse=True)[:TOP_CANDIDATES_FROM_ITEM_BASED])
) - my_groups

print(f"Candidates for prediction: {len(candidates)}")

# -----------------------------
# 5. Получение имени группы
# -----------------------------
def get_group_name_safe(group_id: int) -> str:
    name = get_group_name_from_db(group_id)
    if name:
        return name

    try:
        import requests
        import os
        from dotenv import load_dotenv

        load_dotenv()
        token = os.getenv("VK_TOKEN")
        version = os.getenv("VK_API_VERSION", "5.131")

        url = "https://api.vk.com/method/groups.getById"
        params = {
            "group_id": group_id,
            "access_token": token,
            "v": version,
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if "response" in data and data["response"]:
            name = data["response"][0]["name"]
            save_group_name(group_id, name)
            time.sleep(0.34)
            return name
    except Exception:
        pass

    return "unknown"


# -----------------------------
# 6. Предсказание
# -----------------------------
def predict(group_id: int) -> float:
    features_df = extract_features_df(
        profile_groups=my_groups,
        candidate_group=group_id,
        user_based_scores=user_based_scores,
        item_based_scores=item_based_scores,
        group_to_users=group_to_users,
        profile_members=profile_members,
    )
    features_scaled = scaler.transform(features_df)
    return float(model.predict_proba(features_scaled)[0][1])


print("Predicting...")
preds = [(gid, predict(gid)) for gid in candidates]
preds.sort(key=lambda x: x[1], reverse=True)

all_probs = [p for _, p in preds]
if all_probs:
    print(f"Min prob: {min(all_probs):.6f}")
    print(f"Max prob: {max(all_probs):.6f}")
    print(f"Mean prob: {sum(all_probs)/len(all_probs):.6f}")

print("Fetching names for top-30...")
top_30 = []
for gid, prob in preds[:30]:
    name = get_group_name_safe(gid)
    top_30.append((gid, name, prob))

# -----------------------------
# 7. Вывод
# -----------------------------
print("\n" + "=" * 70)
print("🎯 TOP-30 RECOMMENDATIONS (ML Model)")
print("=" * 70)
for i, (gid, name, prob) in enumerate(top_30, 1):
    print(f"{i:2d}. {gid:10d} | {name[:45]:45} | {prob:.2%}")
print("=" * 70)

with open("ml_recommendations.txt", "w", encoding="utf-8") as f:
    f.write("ML Model Recommendations\n" + "=" * 70 + "\n")
    for i, (gid, name, prob) in enumerate(top_30, 1):
        f.write(f"{i:2d}. {gid} | {name} | {prob:.2%}\n")

print("\n✅ Saved to ml_recommendations.txt")