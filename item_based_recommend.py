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

print("user-based")

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")
API_URL = "https://api.vk.com/method/"

MIN_GROUP_SIZE = 10
MAX_GROUP_POPULARITY = 1000000
MIN_SIMILARITY = 0.02
MIN_COMMON_GROUPS = 1
TOP_K_USERS = 100
TOP_N = 1000                 # сколько рекомендаций сохранять в БД
TOP_N_DISPLAY = 20           # сколько рекомендаций показывать в терминале


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
    """Получить названия групп батчами по 500 для ускорения"""
    group_names = {}
    batch_size = 500
    
    # Преобразуем в список строк
    group_ids_list = [str(gid) for gid in group_ids]
    
    for i in range(0, len(group_ids_list), batch_size):
        batch = group_ids_list[i:i+batch_size]
        batch_str = ",".join(batch)
        
        response = vk_call("groups.getById", {"group_ids": batch_str})
        
        if response and isinstance(response, list):
            for group in response:
                if "id" in group and "name" in group:
                    group_names[str(group["id"])] = group["name"]
        else:
            # Если запрос не удался, помечаем все группы в батче как unknown
            for gid in batch:
                if gid not in group_names:
                    group_names[gid] = "unknown"
    
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
    print(f"Saved {len(user_similarity_data)} similar users to database")

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

# Все рекомендации для сохранения в БД (1000 штук)
all_recommendations = recommendations[:TOP_N]

print(f"\nPrepared {len(all_recommendations)} recommendations for database")

# -------- сохраняем в ClickHouse (только ID, без названий) --------

user_based_data = []
for group_id, score in all_recommendations:
    # В БД сохраняем без названия (или с пустой строкой)
    user_based_data.append((int(group_id), "", float(score)))

if user_based_data:
    insert_user_based_recommendations(user_based_data)
    print(f"Saved {len(user_based_data)} user-based recommendations to ClickHouse database")
else:
    print("No user-based recommendations to insert")

# -------- получаем названия ТОЛЬКО для первых TOP_N_DISPLAY групп --------

# Берем ID первых 20 групп для отображения
display_group_ids = [g for g, _ in all_recommendations[:TOP_N_DISPLAY]]

if display_group_ids:
    # Получаем их названия (батчами по 500 - быстро!)
    print(f"\nFetching names for {len(display_group_ids)} groups from VK API...")
    group_names_dict = get_group_names_batch(display_group_ids)

    # Формируем список для вывода
    display_recommendations = []
    for group_id, score in all_recommendations[:TOP_N_DISPLAY]:
        name = group_names_dict.get(str(group_id), "unknown")
        display_recommendations.append((group_id, score, name))

    # -------- выводим в терминал --------

    print(f"\nTop {TOP_N_DISPLAY} user-based recommendations:\n")
    for g, score, name in display_recommendations:
        print(f"{g} | {name} | {round(score, 6)}")
else:
    print("\nNo recommendations to display")

# -------- сохраняем в файл (только ID, без названий - экономим место) --------

with open("user_based_recommendations.txt", "w", encoding="utf-8") as f:
    f.write(f"Top {len(all_recommendations)} user-based recommendations (group IDs only)\n\n")
    f.write("Format: group_id\tscore\n\n")
    for group_id, score in all_recommendations:
        f.write(f"{group_id}\t{score}\n")

print(f"\nSaved {len(all_recommendations)} recommendations (IDs only) to user_based_recommendations.txt")