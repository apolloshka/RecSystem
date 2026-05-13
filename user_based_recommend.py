import os
import requests
from dotenv import load_dotenv

from src.db.clickhouse_client import (
    get_client,
    create_user_based_table,
    truncate_user_based,
    insert_user_based_recommendations,
    create_user_similarity_table,
    truncate_user_similarity,
    insert_user_similarity,
)
from src.recommenders.common import (
    build_user_to_groups,
    build_group_to_users,
    build_group_popularity,
    jaccard,
    get_user_based_scores_for_profile,
)

print("=== user-based recommender ===")

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")
API_URL = "https://api.vk.com/method/"

MIN_GROUP_SIZE = 10
MAX_GROUP_POPULARITY = 1000000
MIN_SIMILARITY = 0.02
MIN_COMMON_GROUPS = 1
TOP_K_USERS = 100
TOP_N = 100
TOP_N_DISPLAY = 20


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

create_user_based_table()
create_user_similarity_table()

truncate_user_based()
truncate_user_similarity()

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
group_popularity = build_group_popularity(group_to_users)

print("Users loaded from ClickHouse:", len(user_to_groups))
print("Unique groups loaded:", len(group_popularity))

user_group_counts = [len(groups) for groups in user_to_groups.values()]
if user_group_counts:
    print(f"Avg groups per user: {sum(user_group_counts) / len(user_group_counts):.2f}")
    print(f"Max groups per user: {max(user_group_counts)}")

top_popular = sorted(group_popularity.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 most popular groups in dataset:")
for gid, cnt in top_popular:
    print(f"  {gid}: {cnt}")

# -----------------------------
# Ищем похожих пользователей для my_groups
# -----------------------------
similar_users = []

for uid, groups_set in user_to_groups.items():
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
    print(f"Saved {len(user_similarity_data)} similar users to database")
else:
    print("No similar users to insert")

# -----------------------------
# Считаем рекомендации через общий common.py
# -----------------------------
final_scores = get_user_based_scores_for_profile(
    profile_groups=my_groups,
    all_user_groups=user_to_groups,
    group_popularity=group_popularity,
    top_k_users=TOP_K_USERS,
    min_group_size=MIN_GROUP_SIZE,
    min_similarity=MIN_SIMILARITY,
    min_common_groups=MIN_COMMON_GROUPS,
    max_group_popularity=MAX_GROUP_POPULARITY,
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
user_based_data = []
for group_id, score in all_recommendations:
    user_based_data.append((int(group_id), "", float(score)))

if user_based_data:
    insert_user_based_recommendations(user_based_data)
    print(f"Saved {len(user_based_data)} user-based recommendations to ClickHouse database")
else:
    print("No user-based recommendations to insert")

# -----------------------------
# Получаем названия для вывода
# -----------------------------
display_group_ids = [g for g, _ in all_recommendations[:TOP_N_DISPLAY]]

if display_group_ids:
    print(f"\nFetching names for {len(display_group_ids)} groups from VK API...")
    group_names_dict = get_group_names_batch(display_group_ids)

    display_recommendations = []
    for group_id, score in all_recommendations[:TOP_N_DISPLAY]:
        name = group_names_dict.get(str(group_id), "unknown")
        display_recommendations.append((group_id, score, name))

    print(f"\nTop {TOP_N_DISPLAY} user-based recommendations:\n")
    for g, score, name in display_recommendations:
        print(f"{g} | {name} | {round(score, 6)}")
else:
    print("\nNo recommendations to display")

# -----------------------------
# Сохраняем в файл
# -----------------------------
with open("user_based_recommendations.txt", "w", encoding="utf-8") as f:
    f.write(f"Top {len(all_recommendations)} user-based recommendations (group IDs only)\n\n")
    f.write("Format: group_id\tscore\n\n")
    for group_id, score in all_recommendations:
        f.write(f"{group_id}\t{score}\n")

print(f"\nSaved {len(all_recommendations)} recommendations (IDs only) to user_based_recommendations.txt")