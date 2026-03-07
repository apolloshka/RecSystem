import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")

def vk_call(method, params=None):
    params = params or {}
    params["access_token"] = TOKEN
    params["v"] = V
    r = requests.get(f"https://api.vk.com/method/{method}", params=params)
    return r.json()

data = vk_call("groups.get", {
    "extended": 1,
    "count": 1000
})

groups = data["response"]["items"]

my_groups = [g["id"] for g in groups]

print("My groups:", len(my_groups))

for g in groups:
    print(g["id"], g["name"])

# сохраняем
with open("my_groups.json", "w", encoding="utf-8") as f:
    json.dump(my_groups, f)

print("Saved to my_groups.json")