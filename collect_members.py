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
        print("VK ERROR:", data["error"])
        return None

    return data["response"]


# -----------------------------
# читаем группы пользователя
# -----------------------------

with open("my_groups.json", "r", encoding="utf-8") as f:
    seed_groups = json.load(f)

print("All user groups:", seed_groups)

seed_groups = seed_groups

print("Seed groups used:", seed_groups)


members = set()

for gid in seed_groups:

    print("\nCollecting members from group:", gid)

    offset = 0

    while offset < 5000:  # ограничение для эксперимента

        response = vk_call("groups.getMembers", {
            "group_id": gid,
            "count": 1000,
            "offset": offset
        })

        if response is None:
            break

        users = response["items"]

        if not users:
            break

        for u in users:
            members.add(u)

        offset += 1000

        print(f"group {gid} offset {offset} total_members {len(members)}")

        time.sleep(0.34)  # защита от rate limit



with open("members.txt", "w", encoding="utf-8") as f:
    for uid in members:
        f.write(str(uid) + "\n")

print("\nSaved members:", len(members))
print("File: members.txt")