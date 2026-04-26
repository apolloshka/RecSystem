import os
import math
import requests
from collections import defaultdict
from dotenv import load_dotenv

from src.db.clickhouse_client import (
    get_client,
    create_item_based_table,
    truncate_item_based,
    insert_item_based_recommendations,
    create_group_similarity_table,
    truncate_group_similarity,
    insert_group_similarity,
)
from src.recommenders.common import (
    build_user_to_groups,
    build_group_to_users,
    get_item_based_scores_for_profile,
    jaccard,
)

print("=== item-based recommender ===")

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")
API_URL = "https://api.vk.com/method/"

MIN_GROUP_SIZE = 10
MAX_GROUP_SIZE = 1000000
MIN_COMMON_SUPPORT = 3
MIN_SIMILARITY = 0.01
TOP_N = 100
TOP_N_DISPLAY = 20
TOP_SIMILAR_PER_GROUP = 100

# для общего common.py
MAX_ITEM_CANDIDATES = 10000


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


def get_group_names_batch(group_ids):
    group_names = {}
    batch_size = 500

    group_ids_list = [str(gid) for gid in group_ids]

    for i in range(0, len(group_ids_list), batch_size):
        batch = group_ids_list[i:i + batch_size]
        batch_str = ",".join(batch)

        response = vk_call("groups.getById", {"group_ids": batch_str})

        if response and isinstance(response, list):
            for group in response:
                if "id" in group and "name" in group:
                    group_names[str(group["id"])] = group["name"]
        else:
            for gid in batch:
                if gid not in group_names:
                    group_names[gid] = "unknown"

    return group_names


client = get_client()

create_item_based_table()
create_group_similarity_table()

truncate_item_based()
truncate_group_similarity()

# -----------------------------
# Загружаем my_groups
# -----------------------------
my_groups_result = client.query("SELECT group_id FROM my_groups")
my_groups = set(int(row[0]) for row in my_groups_result.result_rows)

print("My groups loaded from ClickHouse:", len(my_groups))

# -----------------------------
# Загружаем user_groups
# -----------------------------
rows = client.query("SELECT user_id, group_id FROM user_groups").result_rows

user_to_groups = build_user_to_groups(rows)
group_to_users = build_group_to_users(user_to_groups)

print("Users loaded from ClickHouse:", len(user_to_groups))
print("Unique groups loaded:", len(group_to_users))

group_sizes = [len(users) for users in group_to_users.values()]
if group_sizes:
    print(f"Avg users per group: {sum(group_sizes) / len(group_sizes):.2f}")
    print(f"Max users per group: {max(group_sizes)}")

top_popular = sorted(group_to_users.items(), key=lambda x: len(x[1]), reverse=True)[:10]
print("Top 10 biggest groups in dataset:")
for gid, users in top_popular:
    print(f"  {gid}: {len(users)}")

# -----------------------------
# Считаем рекомендации через common.py
# -----------------------------
final_scores = get_item_based_scores_for_profile(
    profile_groups=my_groups,
    group_to_users=group_to_users,
    user_to_groups=user_to_groups,
    min_item_group_size=MIN_GROUP_SIZE,
    min_item_similarity=MIN_SIMILARITY,
    min_item_support=MIN_COMMON_SUPPORT,
    max_item_candidates=MAX_ITEM_CANDIDATES,
)

recommendations = sorted(
    final_scores.items(),
    key=lambda x: x[1],
    reverse=True
)

all_recommendations = recommendations[:TOP_N]

print(f"\nPrepared {len(all_recommendations)} recommendations for database")

# -----------------------------
# Сохраняем в ClickHouse
# -----------------------------
item_based_data = []
for group_id, score in all_recommendations:
    item_based_data.append((int(group_id), "", float(score)))

if item_based_data:
    insert_item_based_recommendations(item_based_data)
    print(f"Saved {len(item_based_data)} item-based recommendations to ClickHouse database")
else:
    print("No item-based recommendations to insert")

# -----------------------------
# Получаем названия для вывода
# -----------------------------
display_group_ids = [g for g, _ in all_recommendations[:TOP_N_DISPLAY]]

print(f"\nFetching names for {len(display_group_ids)} groups from VK API...")
group_names_dict = get_group_names_batch(display_group_ids)

display_recommendations = []
for group_id, score in all_recommendations[:TOP_N_DISPLAY]:
    name = group_names_dict.get(str(group_id), "unknown")
    display_recommendations.append((group_id, score, name))

print(f"\nTop {TOP_N_DISPLAY} item-based recommendations:\n")
for g, score, name in display_recommendations:
    print(f"{g} | {name} | {round(score, 6)}")

# -----------------------------
# Сохраняем в файл
# -----------------------------
with open("item_based_recommendations.txt", "w", encoding="utf-8") as f:
    f.write(f"Top {len(all_recommendations)} item-based recommendations (group IDs only)\n\n")
    f.write("Format: group_id\tscore\n\n")
    for group_id, score in all_recommendations:
        f.write(f"{group_id}\t{score}\n")

print(f"\nSaved {len(all_recommendations)} recommendations (IDs only) to item_based_recommendations.txt")

# -----------------------------
# Сохраняем похожие группы для каждой моей группы
# -----------------------------
group_similarity_data = []

for my_group in my_groups:
    users_a = group_to_users.get(my_group, set())

    if len(users_a) < MIN_GROUP_SIZE:
        continue
    if len(users_a) > MAX_GROUP_SIZE:
        continue

    similarities = []

    for group, users_b in group_to_users.items():
        if group == my_group or group in my_groups:
            continue

        if len(users_b) < MIN_GROUP_SIZE:
            continue
        if len(users_b) > MAX_GROUP_SIZE:
            continue

        sim = jaccard(users_a, users_b)
        if sim < MIN_SIMILARITY:
            continue

        common_users = len(users_a & users_b)

        similarities.append((group, sim, common_users))

    similarities.sort(key=lambda x: x[1], reverse=True)

    for target_group, sim, common_users in similarities[:TOP_SIMILAR_PER_GROUP]:
        group_similarity_data.append((my_group, target_group, sim, common_users))

if group_similarity_data:
    insert_group_similarity(group_similarity_data)
    print(f"Saved {len(group_similarity_data)} group similarity pairs to database")
else:
    print("No similar groups to insert")