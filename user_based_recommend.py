import os
import json
import math
import requests
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")
API_URL = "https://api.vk.com/method/"

MIN_GROUP_SIZE = 5
TOP_K_USERS = 100
TOP_N = 30


def vk_call(method, params=None):
    params = params or {}
    params["access_token"] = TOKEN
    params["v"] = V

    r = requests.get(API_URL + method, params=params)
    data = r.json()

    if "error" in data:
        return None

    return data["response"]


def get_group_names(group_ids):
    group_names = {}
    group_ids = list(group_ids)

    for gid in group_ids:
        response = vk_call("groups.getById", {"group_id": gid})

        if response:
            group_names[str(gid)] = response[0]["name"]
        else:
            group_names[str(gid)] = "unknown"

    return group_names


with open("my_groups.json", "r", encoding="utf-8") as f:
    my_groups = set(json.load(f))

with open("user_groups.json", "r", encoding="utf-8") as f:
    user_groups = json.load(f)


group_popularity = defaultdict(int)

for groups in user_groups.values():
    for g in groups:
        group_popularity[g] += 1


def jaccard(a, b):
    a = set(a)
    b = set(b)

    inter = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0.0

    return inter / union


similar_users = []

for uid, groups in user_groups.items():
    sim = jaccard(my_groups, groups)

    if sim > 0:
        similar_users.append((uid, sim, groups))


similar_users.sort(key=lambda x: x[1], reverse=True)
top_users = similar_users[:TOP_K_USERS]


scores = defaultdict(float)

for uid, sim, groups in top_users:
    for g in groups:
        if g in my_groups:
            continue

        if group_popularity[g] < MIN_GROUP_SIZE:
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

top_recommendations = recommendations[:TOP_N]
group_names = get_group_names([g for g, _ in top_recommendations])

print("Top user-based recommendations:\n")
for g, score in top_recommendations:
    print(f"{g} | {group_names.get(str(g), 'unknown')} | {round(score, 6)}")

with open("user_based_recommendations.txt", "w", encoding="utf-8") as f:
    f.write("Top user-based recommendations\n\n")
    for g, score in top_recommendations:
        name = group_names.get(str(g), "unknown")
        f.write(f"{g}\t{name}\t{score}\n")