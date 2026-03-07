import os
import json
import time
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

    r = requests.get(API_URL + method, params=params)
    data = r.json()

    if "error" in data:
        return None

    return data["response"]


user_groups = {}

with open("members.txt", "r") as f:
    users = [line.strip() for line in f]

print("Users loaded:", len(users))

MAX_USERS = 800

ok = 0
fail = 0

for i, uid in enumerate(users[:MAX_USERS]):

    response = vk_call("groups.get", {
        "user_id": uid,
        "count": 1000
    })

    if response is None:
        fail += 1
    else:
        groups = response["items"]
        user_groups[uid] = groups
        ok += 1

    if i % 50 == 0:
        print(f"processed {i}/{MAX_USERS} | ok={ok} fail={fail}")

    time.sleep(0.35)


with open("user_groups.json", "w") as f:
    json.dump(user_groups, f)

print("Dataset saved")
print("Users with data:", ok)
print("Users failed:", fail)