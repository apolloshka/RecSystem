import time
import math
import random
import joblib
import pandas as pd

from src.db.clickhouse_client import get_client, get_group_name_from_db, save_group_name
from src.recommenders.common import (
    build_user_to_groups,
    build_group_to_users,
    build_group_popularity,
    build_profile_members,
    get_user_based_scores_for_profile,
    get_item_based_scores_for_profile,
    extract_feature_dict_for_profile_candidate,
)

print("=== ML Predictor for my_groups ===")

client = get_client()

# -----------------------------
# Параметры
# -----------------------------
TOP_K_SIMILAR_USERS = 50
MIN_GROUP_SIZE = 3
MIN_USER_SIMILARITY = 0.01
MIN_COMMON_GROUPS = 1
MAX_GROUP_POPULARITY = 1_000_000

MIN_ITEM_GROUP_SIZE = 20
MIN_ITEM_SIMILARITY = 0.02
MIN_ITEM_SUPPORT = 1
MAX_ITEM_CANDIDATES = 5000

ENABLE_ITEM_BASED = True

TOP_CANDIDATES_FROM_USER_BASED = 1000
TOP_CANDIDATES_FROM_ITEM_BASED = 1000
TOP_CANDIDATES_FROM_BASELINE = 1000
TOP_CANDIDATES_RANDOM = 1000

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

rows = client.query("SELECT user_id, group_id FROM user_groups").result_rows
user_to_groups = build_user_to_groups(rows)
group_to_users = build_group_to_users(user_to_groups)
group_popularity = build_group_popularity(group_to_users)
all_group_ids = list(group_to_users.keys())

print(f"Loaded users: {len(user_to_groups)}")
print(f"Loaded groups: {len(group_to_users)}")

# -----------------------------
# 3. Считаем CF scores для твоего профиля
# -----------------------------
print("Computing user-based scores for my_groups...")
t0 = time.time()
user_based_scores = get_user_based_scores_for_profile(
    profile_groups=my_groups,
    all_user_groups=user_to_groups,
    group_popularity=group_popularity,
    top_k_users=TOP_K_SIMILAR_USERS,
    min_group_size=MIN_GROUP_SIZE,
    min_similarity=MIN_USER_SIMILARITY,
    min_common_groups=MIN_COMMON_GROUPS,
    max_group_popularity=MAX_GROUP_POPULARITY,
)
print(f"user-based candidates: {len(user_based_scores)} in {time.time() - t0:.2f} sec")

if ENABLE_ITEM_BASED:
    print("Computing item-based scores for my_groups...")
    t0 = time.time()
    item_based_scores = get_item_based_scores_for_profile(
        profile_groups=my_groups,
        group_to_users=group_to_users,
        user_to_groups=user_to_groups,
        min_item_group_size=MIN_ITEM_GROUP_SIZE,
        min_item_similarity=MIN_ITEM_SIMILARITY,
        min_item_support=MIN_ITEM_SUPPORT,
        max_item_candidates=MAX_ITEM_CANDIDATES,
    )
    print(f"item-based candidates: {len(item_based_scores)} in {time.time() - t0:.2f} sec")
else:
    item_based_scores = {}
    print("item-based disabled")

profile_members = build_profile_members(my_groups, group_to_users)
print(f"profile_members: {len(profile_members)}")

# -----------------------------
# 4. Расширенный candidate pool
# -----------------------------
baseline_candidates = [
    gid for gid, pop in sorted(group_popularity.items(), key=lambda x: x[1], reverse=True)
    if gid not in my_groups and pop >= MIN_GROUP_SIZE
][:TOP_CANDIDATES_FROM_BASELINE]

user_based_candidates = [
    gid for gid, _ in sorted(user_based_scores.items(), key=lambda x: x[1], reverse=True)
    if gid not in my_groups
][:TOP_CANDIDATES_FROM_USER_BASED]

item_based_candidates = [
    gid for gid, _ in sorted(item_based_scores.items(), key=lambda x: x[1], reverse=True)
    if gid not in my_groups
][:TOP_CANDIDATES_FROM_ITEM_BASED]

random_pool = [
    gid for gid in all_group_ids
    if gid not in my_groups and group_popularity.get(gid, 0) >= MIN_GROUP_SIZE
]
random_candidates = random.sample(random_pool, min(TOP_CANDIDATES_RANDOM, len(random_pool))) if random_pool else []

candidates = set(user_based_candidates) | set(item_based_candidates) | set(baseline_candidates) | set(random_candidates)
candidates = candidates - my_groups

print(f"Candidates for prediction: {len(candidates)}")

# -----------------------------
# 5. Имя группы
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
    feature_dict = extract_feature_dict_for_profile_candidate(
        profile_groups=my_groups,
        candidate_group=group_id,
        user_based_scores=user_based_scores,
        item_based_scores=item_based_scores,
        group_to_users=group_to_users,
        profile_members=profile_members,
    )
    feature_dict = {f: feature_dict.get(f, 0.0) for f in feature_names}
    features_df = pd.DataFrame([feature_dict], columns=feature_names)
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