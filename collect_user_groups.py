import os
import time
import requests
from dotenv import load_dotenv
from src.db.clickhouse_client import get_client, insert_user_groups, truncate_user_groups

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")
API_URL = "https://api.vk.com/method/"

USERS_PER_GROUP = 250         # сколько брать из каждой группы
BATCH_SIZE = 15               # сколько пользователей в одном execute запросе
SLEEP_BETWEEN_BATCHES = 5.0    # задержка между батчами


def vk_call(method, params=None, retry=3):
    """Обычный вызов VK API"""
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
                time.sleep(10)
            continue

        if "error" in data:
            error_code = data["error"].get("error_code")
            if error_code in (6, 9):  
                wait_time = 30 * (attempt + 1)
                print(f"  Rate limit (error {error_code}), waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"VK API error: {data['error']}")
                return None

        return data["response"]

    return None


def execute_batch(user_ids):
    """
    Выполняет groups.get для нескольких пользователей за один запрос
    user_ids: список ID пользователей (до 25)
    """
    # Формируем код для execute
    code = "var result = [];"
    for i, uid in enumerate(user_ids):
        code += f"""
        var item{i} = API.groups.get({{
            "user_id": {uid},
            "count": 1000
        }});
        result.push({{
            "user_id": {uid},
            "groups": item{i}.items
        }});
        """
    code += "return result;"
    
    response = vk_call("execute", {"code": code})
    return response


def load_balanced_users_from_clickhouse(users_per_group: int):
    """Загружает сбалансированную выборку пользователей из group_members"""
    client = get_client()

    # Статистика по группам (для вывода)
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

    # Уникальные пользователи
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

    # Разбиваем пользователей на батчи
    batches = [users[i:i + BATCH_SIZE] for i in range(0, len(users), BATCH_SIZE)]
    print(f"Batches: {len(batches)} (batch size = {BATCH_SIZE})")

    user_groups = {}
    ok = 0
    fail = 0

    for batch_idx, batch in enumerate(batches, 1):
        print(f"\nProcessing batch {batch_idx}/{len(batches)}...")
        
        response = execute_batch(batch)
        
        if response is None:
            print(f"  Batch {batch_idx} failed completely")
            fail += len(batch)
            continue
        
        batch_ok = 0
        batch_fail = 0
        
        for item in response:
            uid = str(item["user_id"])
            groups = item.get("groups")
            
            # Защита от None и не-списков
            if groups is None or not isinstance(groups, list):
                groups = []
            
            if groups:
                user_groups[uid] = groups
                batch_ok += 1
            else:
                batch_fail += 1
        
        ok += batch_ok
        fail += batch_fail
        
        print(f"  Batch {batch_idx}: ok={batch_ok}, fail={batch_fail}")
        print(f"  Total so far: ok={ok}, fail={fail}")
        
        # Сохраняем промежуточные результаты каждые 10 батчей
        if batch_idx % 10 == 0 and user_groups:
            truncate_user_groups()
            insert_user_groups(user_groups)
            print(f"  [Checkpoint] Saved {len(user_groups)} users so far")
        
        # Задержка между батчами
        if batch_idx < len(batches):
            time.sleep(SLEEP_BETWEEN_BATCHES)

    print("\nCollection finished")
    print(f"Users with data: {ok}")
    print(f"Users failed: {fail}")

    if user_groups:
        truncate_user_groups()
        insert_user_groups(user_groups)
        print("Saved to ClickHouse: user_groups")
    else:
        print("No data to save")


if __name__ == "__main__":
    main()