import requests
import os
from dotenv import load_dotenv
from src.db.clickhouse_client import (
    get_client,
    create_baseline_table,
    truncate_baseline,
    insert_baseline_recommendations
)

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")
API_URL = "https://api.vk.com/method/"

MIN_GROUP_SIZE = 5
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
    """Получает названия групп по их ID"""
    group_names = {}
    
    # Разбиваем на чанки по 500 групп (ограничение VK API)
    chunk_size = 500
    for i in range(0, len(group_ids), chunk_size):
        chunk = group_ids[i:i + chunk_size]
        try:
            response = vk_call("groups.getById", {"group_ids": ",".join(map(str, chunk))})
            if response:
                for group in response:
                    group_names[str(group["id"])] = group["name"]
        except:
            pass
    
    return group_names


client = get_client()

# Создаем таблицу и очищаем её
create_baseline_table()
truncate_baseline()

# -------- получаем мои группы --------

my_groups_data = client.query("SELECT group_id FROM my_groups")
my_groups = set(row[0] for row in my_groups_data.result_rows)

print("My groups loaded from ClickHouse:", len(my_groups))

# -------- считаем популярность --------

query = """
SELECT group_id, count() as cnt
FROM user_groups
GROUP BY group_id
"""

result = client.query(query)
group_popularity = {row[0]: row[1] for row in result.result_rows}

print("Unique groups in user_groups:", len(group_popularity))

# -------- формируем рекомендации --------

recommendations = []

for group, popularity in group_popularity.items():
    if group in my_groups:
        continue

    if popularity < MIN_GROUP_SIZE:
        continue

    recommendations.append((group, popularity))

recommendations.sort(key=lambda x: x[1], reverse=True)
top = recommendations[:TOP_N]

print("Recommendations found:", len(top))

# -------- получаем названия групп --------

group_ids_for_names = [g for g, _ in top]
group_names = get_group_names(group_ids_for_names)

print("\nTop baseline recommendations:\n")
for g, score in top:
    name = group_names.get(str(g), "unknown")
    print(f"{g} | {name} | {score}")

# -------- сохраняем в файл --------

with open("baseline_recommendations.txt", "w", encoding="utf-8") as f:
    f.write("Top baseline recommendations\n\n")
    for g, score in top:
        name = group_names.get(str(g), "unknown")
        f.write(f"{g}\t{name}\t{score}\n")

# -------- сохраняем в ClickHouse --------
baseline_data = []
for group_id, members_count in top:
    group_name = group_names.get(str(group_id), "unknown")
    baseline_data.append((int(group_id), group_name, int(members_count)))

if baseline_data:
    insert_baseline_recommendations(baseline_data)
else:
    print("⚠ No baseline recommendations to insert")