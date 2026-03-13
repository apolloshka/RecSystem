import os
import json
import time
import random
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")

API_URL = "https://api.vk.com/method/"


def vk_call(method, params=None):
    params = params or {}
    params["access_token"] = TOKEN
    params["v"] = V

    try:
        r = requests.get(API_URL + method, params=params)
        data = r.json()
    except Exception:
        return None

    if "error" in data:
        return None

    return data["response"]


# ---------- загрузка пользователей ----------

with open("members.txt", "r", encoding="utf-8") as f:
    users = [line.strip() for line in f]

print("Users loaded:", len(users))


# ---------- случайная выборка ----------

MAX_USERS = 1500

random.seed(42)      # фиксируем случайность (для воспроизводимости)
random.shuffle(users)

users = users[:MAX_USERS]



user_groups = {}

ok = 0
fail = 0

for i, uid in enumerate(users):

    response = vk_call("groups.get", {
        "user_id": uid,
        "count": 1000
    })

    if response is None:
        fail += 1
    else:
        groups = response.get("items", [])
        user_groups[uid] = groups
        ok += 1

    if i % 50 == 0:
        print(f"processed {i}/{MAX_USERS} | ok={ok} fail={fail}")

    time.sleep(0.35)


# ---------- сохранение ----------

with open("user_groups.json", "w", encoding="utf-8") as f:
    json.dump(user_groups, f)

print("\nDataset saved")
print("Users with data:", ok)
print("Users failed:", fail)