import os
import requests
from dotenv import load_dotenv
from src.db.clickhouse_client import insert_my_groups, truncate_my_groups

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")
API_URL = "https://api.vk.com/method/"


def vk_call(method, params=None):
    params = params or {}
    params["access_token"] = TOKEN
    params["v"] = V

    try:
        r = requests.get(API_URL + method, params=params, timeout=30)
        data = r.json()
    except Exception as e:
        print("Request error:", e)
        return None

    if "error" in data:
        print("VK API error:", data["error"])
        return None

    return data["response"]


response = vk_call("groups.get", {
    "extended": 1,
    "count": 1000
})

if response is None:
    print("Failed to load my groups")
    raise SystemExit(1)

groups = response.get("items", [])
my_groups = [g["id"] for g in groups]

print("My groups:", len(my_groups))

for g in groups:
    print(g["id"], g["name"])

try:
    truncate_my_groups()
    insert_my_groups(my_groups)
    print("Saved to ClickHouse: my_groups")
except Exception as e:
    print("ClickHouse error:", e)