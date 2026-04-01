import os
import time
import requests
from dotenv import load_dotenv
from src.db.clickhouse_client import get_client, truncate_group_members, insert_group_members

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")
API_URL = "https://api.vk.com/method/"

DELAY_BETWEEN_REQUESTS = 1
DELAY_ON_FLOOD = 120
MAX_MEMBERS_PER_GROUP = 2000  # максимум участников с группы


def vk_call(method, params=None, retry=3):
    params = params or {}
    params["access_token"] = TOKEN
    params["v"] = V

    for attempt in range(retry):
        try:
            r = requests.get(API_URL + method, params=params, timeout=30)
            data = r.json()
        except Exception as e:
            print(f"  Request error: {e}")
            if attempt < retry - 1:
                time.sleep(DELAY_ON_FLOOD)
            continue

        if "error" in data:
            error_code = data["error"].get("error_code")
            if error_code == 9:
                print(f"  Flood control, waiting {DELAY_ON_FLOOD} seconds...")
                if attempt < retry - 1:
                    time.sleep(DELAY_ON_FLOOD)
                    continue
                else:
                    return None
            else:
                print(f"  VK API error: {data['error']}")
                return None

        return data["response"]

    return None


def collect_group_members(group_id):
    members = []
    offset = 0
    count = 1000

    while offset < MAX_MEMBERS_PER_GROUP:  # ← ограничение
        remaining = MAX_MEMBERS_PER_GROUP - offset
        current_count = min(count, remaining)
        
        print(f"  Fetching members with offset {offset}...")
        response = vk_call("groups.getMembers", {
            "group_id": group_id,
            "offset": offset,
            "count": current_count,
            "fields": ""
        })

        if not response:
            break

        items = response.get("items", [])
        members.extend(items)
        print(f"  Got {len(items)} members, total so far: {len(members)}")

        if len(items) < current_count:
            break

        offset += current_count
        time.sleep(DELAY_BETWEEN_REQUESTS)

    return members


client = get_client()

# Получаем мои группы
my_groups_result = client.query("SELECT group_id FROM my_groups")
my_groups = [row[0] for row in my_groups_result.result_rows]

print("My groups loaded from ClickHouse:", len(my_groups))
print("Seed groups used:", my_groups)

# Очищаем таблицу
truncate_group_members()

total_members = 0

for idx, group_id in enumerate(my_groups, 1):
    print(f"\n[{idx}/{len(my_groups)}] Collecting members from group: {group_id}")

    members = collect_group_members(group_id)

    if members:
        insert_group_members(group_id, members)
        total_members += len(members)
        print(f"  Collected members: {len(members)} (max {MAX_MEMBERS_PER_GROUP})")
    else:
        print(f"  No members collected for group {group_id}")

    # Задержка между группами
    if idx < len(my_groups):
        time.sleep(1)

print(f"\nFinished collecting group members")
print(f"Total members collected: {total_members}")

# Сохраняем в файл для бэкапа
with open("members.txt", "w", encoding="utf-8") as f:
    for uid in members:
        f.write(str(uid) + "\n")

print("File: members.txt")