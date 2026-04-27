import os
import time

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv

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
from src.recommenders.config import (
    TOP_K_SIMILAR_USERS,
    MIN_GROUP_SIZE,
    MIN_USER_SIMILARITY,
    MIN_COMMON_GROUPS,
    MAX_GROUP_POPULARITY,
    MIN_ITEM_GROUP_SIZE,
    MIN_ITEM_SIMILARITY,
    MIN_ITEM_SUPPORT,
    MAX_ITEM_CANDIDATES,
    ENABLE_ITEM_BASED,
    TOP_USER_BASED_CANDIDATES,
    TOP_ITEM_BASED_CANDIDATES,
    TOP_BASELINE_BLACKLIST,
    MAX_RECOMMENDED_GROUP_POPULARITY,
    MIN_CF_SCORE,
    MIN_MAX_GROUP_SIMILARITY,
    MODEL_BUNDLE_PATH,
    FEATURE_NAMES_PATH,
    TRAIN_USERS_PATH,
    TOP_K_RECOMMENDATIONS,
)

print("=== ML Predictor for my_groups ===")

load_dotenv()
VK_TOKEN = os.getenv("VK_TOKEN")
VK_API_VERSION = os.getenv("VK_API_VERSION", "5.131")

client = get_client()
train_users = joblib.load(TRAIN_USERS_PATH)
model_bundle = joblib.load(MODEL_BUNDLE_PATH)
pipeline = model_bundle["pipeline"]
feature_names = model_bundle.get("feature_names", joblib.load(FEATURE_NAMES_PATH))

print(f"Loaded feature_names: {feature_names}")

my_groups = {
    int(row[0])
    for row in client.query("SELECT group_id FROM my_groups").result_rows
}
print(f"My groups: {len(my_groups)}")

train_users_str = ",".join(map(str, train_users))

rows = client.query(f"""
    SELECT user_id, group_id
    FROM user_groups
    WHERE user_id IN ({train_users_str})
""").result_rows

print(f"[INFO] Loaded {len(rows)} rows for train users only")

user_to_groups = build_user_to_groups(rows)
group_to_users = build_group_to_users(user_to_groups)
group_popularity = build_group_popularity(group_to_users)

print(f"Loaded users: {len(user_to_groups)}")
print(f"Loaded groups: {len(group_to_users)}")

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
    target_user_id=None,
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
        target_user_id=None,
    )

    print(f"item-based candidates: {len(item_based_scores)} in {time.time() - t0:.2f} sec")
else:
    item_based_scores = {}
    print("item-based disabled")

profile_members = build_profile_members(
    profile_groups=my_groups,
    group_to_users=group_to_users,
    target_user_id=None,
)

print(f"profile_members: {len(profile_members)}")

baseline_blacklist = [
    gid
    for gid, pop in sorted(
        group_popularity.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    if gid not in my_groups and pop >= MIN_GROUP_SIZE
]

baseline_blacklist = set(baseline_blacklist[:TOP_BASELINE_BLACKLIST])

print(f"Baseline blacklist: {len(baseline_blacklist)}")

user_based_candidates = [
    gid
    for gid, _ in sorted(
        user_based_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    if gid not in my_groups
][:TOP_USER_BASED_CANDIDATES]

item_based_candidates = [
    gid
    for gid, _ in sorted(
        item_based_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    if gid not in my_groups
][:TOP_ITEM_BASED_CANDIDATES]

candidates = set(user_based_candidates) | set(item_based_candidates)
candidates = candidates - my_groups
candidates = candidates - baseline_blacklist

candidates = {
    gid
    for gid in candidates
    if group_popularity.get(gid, 0) <= MAX_RECOMMENDED_GROUP_POPULARITY
}

filtered_candidates = []

for gid in candidates:
    feature_dict = extract_feature_dict_for_profile_candidate(
        profile_groups=my_groups,
        candidate_group=gid,
        user_based_scores=user_based_scores,
        item_based_scores=item_based_scores,
        group_to_users=group_to_users,
        profile_members=profile_members,
        target_user_id=None,
    )

    has_cf_score = (
        feature_dict["user_based_score"] >= MIN_CF_SCORE
        or feature_dict["item_based_score"] >= MIN_CF_SCORE
    )

    has_group_similarity = (
        feature_dict["max_group_similarity"] >= MIN_MAX_GROUP_SIMILARITY
    )

    if has_cf_score and has_group_similarity:
        filtered_candidates.append(gid)

candidates = set(filtered_candidates)

print(f"Candidates for prediction: {len(candidates)}")


def get_group_names_batch(group_ids: list[int]) -> dict[str, str]:
    if not group_ids:
        return {}

    result = {}
    chunk_size = 500

    for i in range(0, len(group_ids), chunk_size):
        chunk = group_ids[i:i + chunk_size]
        chunk_str = ",".join(map(str, chunk))

        try:
            url = "https://api.vk.com/method/groups.getById"
            params = {
                "group_ids": chunk_str,
                "access_token": VK_TOKEN,
                "v": VK_API_VERSION,
            }

            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if "response" in data:
                for group in data["response"]:
                    gid = str(group["id"])
                    name = group["name"]
                    result[gid] = name
                    save_group_name(gid, name)
            else:
                print(f"  Error: {data.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"  Exception: {e}")

        if i + chunk_size < len(group_ids):
            time.sleep(0.34)

    return result


def get_group_name_safe(group_id: int) -> str:
    name = get_group_name_from_db(group_id)

    if name:
        return name

    names = get_group_names_batch([group_id])
    return names.get(str(group_id), "unknown")


def predict(group_id: int) -> float:
    feature_dict = extract_feature_dict_for_profile_candidate(
        profile_groups=my_groups,
        candidate_group=group_id,
        user_based_scores=user_based_scores,
        item_based_scores=item_based_scores,
        group_to_users=group_to_users,
        profile_members=profile_members,
        target_user_id=None,
    )

    feature_dict = {
        f: feature_dict.get(f, 0.0)
        for f in feature_names
    }

    features_df = pd.DataFrame([feature_dict], columns=feature_names)
    raw_prob = float(pipeline.predict_proba(features_df)[0][1])
    return raw_prob


print("Predicting...")

preds = []
for gid in candidates:
    raw_prob = predict(gid)
    preds.append((gid, raw_prob))

# Rank by raw model score to avoid calibration plateaus.
preds.sort(key=lambda x: x[1], reverse=True)

all_probs = [p for _, p in preds]

if all_probs:
    print(f"Min prob: {min(all_probs):.6f}")
    print(f"Max prob: {max(all_probs):.6f}")
    print(f"Mean prob: {sum(all_probs) / len(all_probs):.6f}")

print("Fetching names for top-30...")

top_30 = []
for gid, raw_prob in preds[:TOP_K_RECOMMENDATIONS]:
    name = get_group_name_safe(gid)
    top_30.append((gid, name, raw_prob))

print("\n" + "=" * 70)
print(f"TOP-{TOP_K_RECOMMENDATIONS} RECOMMENDATIONS (ML Model)")
print("=" * 70)

for i, (gid, name, raw_prob) in enumerate(top_30, 1):
    print(f"{i:2d}. {gid:10d} | {name[:45]:45} | {raw_prob:.2%}")

print("=" * 70)

with open("ml_recommendations.txt", "w", encoding="utf-8") as f:
    f.write("ML Model Recommendations\n" + "=" * 70 + "\n")

    for i, (gid, name, raw_prob) in enumerate(top_30, 1):
        f.write(f"{i:2d}. {gid} | {name} | {raw_prob:.2%}\n")

print("\nSaved to ml_recommendations.txt")