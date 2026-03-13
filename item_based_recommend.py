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
MIN_COMMON_SUPPORT = 2
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


group_users = defaultdict(set)

for user, groups in user_groups.items():
    for g in groups:
        group_users[g].add(user)


def jaccard(a, b):
    inter = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0.0

    return inter / union


scores = defaultdict(float)
counts = defaultdict(int)

for my_group in my_groups:
    users_a = group_users.get(my_group, set())

    if len(users_a) < MIN_GROUP_SIZE:
        continue

    for group, users_b in group_users.items():
        if group in my_groups:
            continue

        if len(users_b) < MIN_GROUP_SIZE:
            continue

        sim = jaccard(users_a, users_b)

        if sim <= 0:
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

top_recommendations = recommendations[:TOP_N]
group_names = get_group_names([g for g, _ in top_recommendations])

print("Top item-based recommendations:\n")
for g, score in top_recommendations:
    print(f"{g} | {group_names.get(str(g), 'unknown')} | {round(score, 6)}")

with open("item_based_recommendations.txt", "w", encoding="utf-8") as f:
    f.write("Top item-based recommendations\n\n")
    for g, score in top_recommendations:
        name = group_names.get(str(g), "unknown")
        f.write(f"{g}\t{name}\t{score}\n")