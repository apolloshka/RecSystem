import os
import time
import requests
from dotenv import load_dotenv
from src.db.clickhouse_client import get_client, insert_user_groups, truncate_user_groups

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")
API_URL = "https://api.vk.com/method/"

USERS_PER_GROUP = 350    # сколько брать из каждой группы
SLEEP_SECONDS = 1.5      # задержка между запросами


def vk_call(method, params=None, retry=3):
    params = params or {}
    params["access_token"] = TOKEN
    params["v"] = V

    for attempt in range(retry):
        try:
            r = requests.get(API_URL + method, params=params, timeout=30)
            data = r.json()
        except Exception as e:
            print(f"Request error: {e}")
            if attempt < retry - 1:
                time.sleep(SLEEP_SECONDS * 2)
            continue

        if "error" in data:
            error_code = data["error"].get("error_code")
            if error_code in (6, 9):  # Rate limit or Flood control
                wait_time = SLEEP_SECONDS * (attempt + 1) * 2
                print(f"  Rate limit (error {error_code}), waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"VK API error: {data['error']}")
                return None

        return data["response"]

    print(f"  Failed after {retry} attempts")
    return None


def load_balanced_users_from_clickhouse(users_per_group: int):
    client = get_client()

    stats_result = client.query(f"""
        SELECT source_group_id, count() AS cnt
        FROM
        (
            SELECT
                source_group_id,
                member_id,
                row_number() OVER (
                    PARTITION BY source_group_id
                    ORDER BY cityHash64(member_id)
                ) AS rn
            FROM group_members
        )
        WHERE rn <= {users_per_group}
        GROUP BY source_group_id
        ORDER BY source_group_id
    """)

    per_group_stats = [(row[0], row[1]) for row in stats_result.result_rows]

    members_result = client.query(f"""
        SELECT DISTINCT member_id
        FROM
        (
            SELECT
                source_group_id,
                member_id,
                row_number() OVER (
                    PARTITION BY source_group_id
                    ORDER BY cityHash64(member_id)
                ) AS rn
            FROM group_members
        )
        WHERE rn <= {users_per_group}
    """)

    users = [str(row[0]) for row in members_result.result_rows]
    return users, per_group_stats


def main():
    print("Loading balanced users from ClickHouse...")
    users, per_group_stats = load_balanced_users_from_clickhouse(USERS_PER_GROUP)

    print("\nSelected users per group:")
    for group_id, cnt in per_group_stats:
        print(f"  source_group_id={group_id}: {cnt}")

    print(f"\nUnique users loaded for processing: {len(users)}")

    user_groups = {}
    ok = 0
    fail = 0

    for i, uid in enumerate(users, start=1):
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

        if i % 50 == 0 or i == len(users):
            print(f"processed {i}/{len(users)} | ok={ok} fail={fail}")

        # Сохраняем промежуточные результаты каждые 100 пользователей
        if i % 100 == 0 and user_groups:
            truncate_user_groups()
            insert_user_groups(user_groups)
            print(f"  [Checkpoint] Saved {len(user_groups)} users so far")

        time.sleep(SLEEP_SECONDS)

    print("\nCollection finished")
    print("Users with data:", ok)
    print("Users failed:", fail)

    if user_groups:
        truncate_user_groups()
        insert_user_groups(user_groups)
        print("Saved to ClickHouse: user_groups")
    else:
        print("No data to save")


if __name__ == "__main__":
    main()