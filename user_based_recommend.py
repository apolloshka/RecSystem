import os
import math
import requests
from collections import defaultdict
from dotenv import load_dotenv
from src.db.clickhouse_client import (
    get_client,
    create_user_based_table,
    truncate_user_based,
    insert_user_based_recommendations,
    create_user_similarity_table,
    truncate_user_similarity,
    insert_user_similarity
)

print("=== UPDATED VERSION: softer filtered user-based ===")

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")
API_URL = "https://api.vk.com/method/"

MIN_GROUP_SIZE = 10
MAX_GROUP_POPULARITY = 10000000
MIN_SIMILARITY = 0.02
MIN_COMMON_GROUPS = 1
TOP_K_USERS = 100
TOP_N = 30


def vk_call(method, params=None):
    params = params or {}
    params["access_token"] = TOKEN
    params["v"] = V

    try:
        r = requests.get(API_URL + method, params=params, timeout=30)
        data = r.json()
    except Exception:
        return None

    if "error" in data:
        return None

    return data["response"]


def get_group_names(group_ids):
    group_names = {}

    for gid in group_ids:
        response = vk_call("groups.getById", {"group_id": gid})

        if response:
            group_names[str(gid)] = response[0]["name"]
        else:
            group_names[str(gid)] = "unknown"

    return group_names


def jaccard(a, b):
    a = set(a)
    b = set(b)

    inter = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0.0

    return inter / union


client = get_client()

create_user_based_table()
create_user_similarity_table()

truncate_user_based()
truncate_user_similarity()

my_groups_result = client.query("SELECT group_id FROM my_groups")
my_groups = set(row[0] for row in my_groups_result.result_rows)

print("My groups loaded from ClickHouse:", len(my_groups))

user_groups_result = client.query("""
    SELECT user_id, group_id
    FROM user_groups
""")

user_groups = defaultdict(list)

for user_id, group_id in user_groups_result.result_rows:
    user_groups[str(user_id)].append(group_id)

print("Users loaded from ClickHouse:", len(user_groups))

group_popularity = defaultdict(int)

for groups in user_groups.values():
    for g in set(groups):
        group_popularity[g] += 1

print("Unique groups loaded:", len(group_popularity))

user_group_counts = [len(set(groups)) for groups in user_groups.values()]
if user_group_counts:
    print(f"Avg groups per user: {sum(user_group_counts) / len(user_group_counts):.2f}")
    print(f"Max groups per user: {max(user_group_counts)}")

top_popular = sorted(group_popularity.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 most popular groups in dataset:")
for gid, cnt in top_popular:
    print(f"  {gid}: {cnt}")

similar_users = []

for uid, groups in user_groups.items():
    groups_set = set(groups)
    sim = jaccard(my_groups, groups_set)
    common_groups = len(my_groups & groups_set)

    if sim >= MIN_SIMILARITY and common_groups >= MIN_COMMON_GROUPS:
        similar_users.append((uid, sim, groups_set, common_groups))

similar_users.sort(key=lambda x: x[1], reverse=True)
top_users = similar_users[:TOP_K_USERS]

print("Similar users found after filtering:", len(similar_users))
print("Top users used:", len(top_users))

user_similarity_data = []
for uid, sim, groups_set, common_groups in top_users:
    user_similarity_data.append((uid, sim, common_groups))

if user_similarity_data:
    insert_user_similarity(user_similarity_data)

scores = defaultdict(float)

for uid, sim, groups_set, common_groups in top_users:
    for g in groups_set:
        if g in my_groups:
            continue

        if group_popularity[g] < MIN_GROUP_SIZE:
            continue

        if group_popularity[g] > MAX_GROUP_POPULARITY:
            continue

        scores[g] += sim

final_scores = {}

for g in scores:
    popularity_penalty = math.log(1 + group_popularity[g])
    final_scores[g] = scores[g] / popularity_penalty

recommendations = sorted(
    final_scores.items(),
    key=lambda x: x[1],
    reverse=True
)

candidate_recommendations = recommendations[:TOP_N * 5]
group_names = get_group_names([g for g, _ in candidate_recommendations])

filtered_recommendations = []
for g, score in candidate_recommendations:
    name = group_names.get(str(g), "unknown")
    if name == "unknown":
        continue
    filtered_recommendations.append((g, score, name))

top_recommendations = filtered_recommendations[:TOP_N]

print("\nTop user-based recommendations:\n")
for g, score, name in top_recommendations:
    print(f"{g} | {name} | {round(score, 6)}")

with open("user_based_recommendations.txt", "w", encoding="utf-8") as f:
    f.write("Top user-based recommendations\n\n")
    for g, score, name in top_recommendations:
        f.write(f"{g}\t{name}\t{score}\n")

print(f"✓ Saved {len(top_recommendations)} recommendations to user_based_recommendations.txt")

user_based_data = []
for group_id, score, group_name in top_recommendations:
    user_based_data.append((int(group_id), group_name, float(score)))

if user_based_data:
    insert_user_based_recommendations(user_based_data)