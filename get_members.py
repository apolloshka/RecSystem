import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")

GROUP_ID = 211224002  # твоя группа
MAX_MEMBERS = 5000    # стартуем с 5000

def vk_call(method, params=None):
    params = params or {}
    params["access_token"] = TOKEN
    params["v"] = V
    r = requests.get(f"https://api.vk.com/method/{method}", params=params, timeout=20)
    return r.json()

members = []
offset = 0

while len(members) < MAX_MEMBERS:
    batch = min(1000, MAX_MEMBERS - len(members))
    data = vk_call("groups.getMembers", {
        "group_id": GROUP_ID,
        "count": batch,
        "offset": offset
    })

    if "error" in data:
        print("ERROR:", data["error"])
        break

    items = data["response"]["items"]
    if not items:
        break

    members.extend(items)
    offset += len(items)

    print(f"members: {len(members)}")
    time.sleep(0.35)

print("TOTAL:", len(members))

with open("members.txt", "w", encoding="utf-8") as f:
    for uid in members:
        f.write(str(uid) + "\n")

print("Saved to members.txt")