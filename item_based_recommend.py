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
    insert_group_similarity
)

print("=== UPDATED VERSION: filtered item-based ===")

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")
API_URL = "https://api.vk.com/method/"

MIN_GROUP_SIZE = 10          # минимальный размер группы в датасете
MAX_GROUP_SIZE = 1000000        # отсечь слишком массовые/шумные группы
MIN_COMMON_SUPPORT = 2       # группа должна быть похожа хотя бы на 2 мои группы
MIN_SIMILARITY = 0.01        # минимальная похожесть между группами
TOP_N = 30                   # сколько рекомендаций вернуть
TOP_SIMILAR_PER_GROUP = 100  # сколько похожих групп сохранять для каждой моей группы


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
    inter = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0.0

    return inter / union


client = get_client()

# Создаем таблицы
create_item_based_table()
create_group_similarity_table()

# Очищаем старые данные
truncate_item_based()
truncate_group_similarity()

# -------- загружаем мои группы --------

my_groups_result = client.query("SELECT group_id FROM my_groups")
my_groups = set(row[0] for row in my_groups_result.result_rows)

print("My groups loaded from ClickHouse:", len(my_groups))

# -------- загружаем user_groups --------

user_groups_result = client.query("""
    SELECT user_id, group_id
    FROM user_groups
""")

user_groups = defaultdict(set)

for user_id, group_id in user_groups_result.result_rows:
    user_groups[str(user_id)].add(group_id)

print("Users loaded from ClickHouse:", len(user_groups))

# -------- group -> users --------

group_users = defaultdict(set)

for user, groups in user_groups.items():
    for g in groups:
        group_users[g].add(user)

print("Unique groups loaded:", len(group_users))

group_sizes = [len(users) for users in group_users.values()]
if group_sizes:
    print(f"Avg users per group: {sum(group_sizes) / len(group_sizes):.2f}")
    print(f"Max users per group: {max(group_sizes)}")

top_popular = sorted(group_users.items(), key=lambda x: len(x[1]), reverse=True)[:10]
print("Top 10 biggest groups in dataset:")
for gid, users in top_popular:
    print(f"  {gid}: {len(users)}")

# -------- item-based scoring --------

scores = defaultdict(float)
counts = defaultdict(int)

for my_group in my_groups:
    users_a = group_users.get(my_group, set())

    if len(users_a) < MIN_GROUP_SIZE:
        continue

    if len(users_a) > MAX_GROUP_SIZE:
        continue

    for group, users_b in group_users.items():
        if group in my_groups:
            continue

        if len(users_b) < MIN_GROUP_SIZE:
            continue

        if len(users_b) > MAX_GROUP_SIZE:
            continue

        sim = jaccard(users_a, users_b)

        if sim < MIN_SIMILARITY:
            continue

        scores[group] += sim
        counts[group] += 1

final_scores = {}

for group in scores:
    if counts[group] < MIN_COMMON_SUPPORT:
        continue

    avg_score = scores[group] / counts[group]
    popularity_penalty = math.log(1 + len(group_users[group]))
    final_scores[group] = avg_score / popularity_penalty

recommendations = sorted(
    final_scores.items(),
    key=lambda x: x[1],
    reverse=True
)

# берем с запасом, чтобы после удаления unknown осталось около 30
candidate_recommendations = recommendations[:TOP_N * 5]
group_names = get_group_names([g for g, _ in candidate_recommendations])

filtered_recommendations = []
for g, score in candidate_recommendations:
    name = group_names.get(str(g), "unknown")
    if name == "unknown":
        continue
    filtered_recommendations.append((g, score, name))

top_recommendations = filtered_recommendations[:TOP_N]

print("\nTop item-based recommendations:\n")
for g, score, name in top_recommendations:
    print(f"{g} | {name} | {round(score, 6)}")

# -------- сохраняем похожие группы для каждой моей группы --------

group_similarity_data = []

for my_group in my_groups:
    users_a = group_users.get(my_group, set())

    if len(users_a) < MIN_GROUP_SIZE:
        continue

    if len(users_a) > MAX_GROUP_SIZE:
        continue

    similarities = []

    for group, users_b in group_users.items():
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
else:
    print("⚠ No similar groups to insert")

# -------- сохраняем в файл --------

with open("item_based_recommendations.txt", "w", encoding="utf-8") as f:
    f.write("Top item-based recommendations\n\n")
    for g, score, name in top_recommendations:
        f.write(f"{g}\t{name}\t{score}\n")

print(f"✓ Saved {len(top_recommendations)} recommendations to item_based_recommendations.txt")

# -------- сохраняем в ClickHouse --------

item_based_data = []
for group_id, score, group_name in top_recommendations:
    item_based_data.append((int(group_id), group_name, float(score)))

if item_based_data:
    insert_item_based_recommendations(item_based_data)
else:
    print("⚠ No item-based recommendations to insert")