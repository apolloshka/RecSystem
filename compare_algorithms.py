import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = "5.131"


def load_recommendations(file):
    recs = []

    with open(file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("Top") or not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) != 2:
                continue

            group, score = parts
            recs.append((group, float(score)))

    return recs


def get_group_name(group_id):

    url = "https://api.vk.com/method/groups.getById"

    params = {
        "group_id": group_id,
        "access_token": TOKEN,
        "v": V
    }

    r = requests.get(url, params=params).json()

    if "response" in r:
        g = r["response"][0]
        return g["name"]

    return "unknown"


user_recs = load_recommendations("user_based_recommendations.txt")
item_recs = load_recommendations("item_based_recommendations.txt")

df_user = pd.DataFrame(user_recs, columns=["group", "score"])
df_item = pd.DataFrame(item_recs, columns=["group", "score"])


user_groups = set(df_user["group"])
item_groups = set(df_item["group"])

intersection = user_groups & item_groups
union = user_groups | item_groups

overlap = len(intersection) / len(union) if union else 0


print("\nСРАВНЕНИЕ АЛГОРИТМОВ")
print("---------------------")
print("User recommendations:", len(user_groups))
print("Item recommendations:", len(item_groups))
print("Общие рекомендации:", len(intersection))
print("Overlap:", round(overlap, 4))


print("\nПЕРЕСЕКАЮЩИЕСЯ ГРУППЫ:")
print("---------------------")

if intersection:

    for gid in sorted(intersection):

        name = get_group_name(gid)

        print(f"{gid}  |  {name}")

else:
    print("Нет пересечений")