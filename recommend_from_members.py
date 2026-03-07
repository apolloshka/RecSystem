import os
import time
from collections import Counter
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")

GROUP_ID_ME = 211224002  # группа, в которой ты состоишь (пока одна)
MEMBERS_FILE = "members.txt"

MAX_USERS = 800          # сколько участников обработать (стартуй с 300-800)
MIN_DELAY = 0.35

def vk_call(method, params=None):
    params = params or {}
    params["access_token"] = TOKEN
    params["v"] = V
    r = requests.get(f"https://api.vk.com/method/{method}", params=params, timeout=20)
    return r.json()

# 1) твои группы (чтобы исключить из рекомендаций)
my_groups_resp = vk_call("groups.get", {"extended": 0, "count": 1000})
if "error" in my_groups_resp:
    raise RuntimeError(my_groups_resp["error"])
my_groups = set(my_groups_resp["response"]["items"])
print("My groups:", len(my_groups))

# 2) читаем участников
with open(MEMBERS_FILE, "r", encoding="utf-8") as f:
    users = [int(line.strip()) for line in f if line.strip()]

users = users[:MAX_USERS]
print("Users to process:", len(users))

group_counter = Counter()
ok = 0
fail = 0

for i, uid in enumerate(users, 1):
    data = vk_call("groups.get", {"user_id": uid, "extended": 0, "count": 1000})

    if "error" in data:
        fail += 1
    else:
        ok += 1
        groups_u = data["response"]["items"]
        # считаем группы, которых у тебя нет
        for g in groups_u:
            if g not in my_groups:
                group_counter[g] += 1

    if i % 50 == 0:
        print(f"processed {i}/{len(users)} | ok={ok} fail={fail} unique_groups={len(group_counter)}")

    time.sleep(MIN_DELAY)

print("\nDONE")
print("ok:", ok, "fail:", fail)
print("candidate groups:", len(group_counter))

# 3) топ по частоте
top = group_counter.most_common(30)
print("\nTOP-30 recommended group IDs (by frequency among similar audience):")
for gid, cnt in top:
    print(gid, cnt)

# 4) (опционально) подтянем названия для топ-20
top_ids = [gid for gid, _ in top[:20]]
if top_ids:
    info = vk_call("groups.getById", {"group_ids": ",".join(map(str, top_ids))})
    if "error" in info:
        print("\nCould not load names:", info["error"])
    else:
        items = info["response"]
        print("\nTOP-20 with names:")
        by_id = {g["id"]: g for g in items}
        for gid, cnt in top[:20]:
            g = by_id.get(gid, {})
            print(f'{gid}\t{cnt}\t{g.get("name","")}  ({g.get("screen_name","")})')