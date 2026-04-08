import joblib
import math
import time
import numpy as np
from collections import defaultdict
from src.db.clickhouse_client import get_client, get_group_name_from_db, save_group_name

print("=== ML Predictor ===")

client = get_client()
model = joblib.load('recommendation_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Загружаем данные
group_to_users = defaultdict(set)
for user_id, group_id in client.query("SELECT user_id, group_id FROM user_groups").result_rows:
    group_to_users[int(group_id)].add(int(user_id))

my_groups = [row[0] for row in client.query("SELECT group_id FROM my_groups").result_rows]
print(f"My groups: {len(my_groups)}")

# Определяем признаки (из модели)
feature_names = [f"feature_{i}" for i in range(model.coef_.shape[1])]

# Функция извлечения признаков
def extract_features(group_id):
    group_users = group_to_users.get(group_id, set())
    group_pop = len(group_users)
    log_pop = math.log(1 + group_pop)
    
    # Item-based Jaccard
    total_jaccard = 0.0
    for my_group in my_groups:
        my_users = group_to_users.get(my_group, set())
        if not my_users:
            continue
        inter = len(group_users & my_users)
        union = len(group_users | my_users)
        if union > 0:
            total_jaccard += inter / union
    avg_jaccard = total_jaccard / len(my_groups) if my_groups else 0.0
    
    # User-based similar users
    similar_count = 0
    for user in list(group_users)[:50]:
        user_groups = client.query(f"SELECT group_id FROM user_groups WHERE user_id = {user} LIMIT 10").result_rows
        user_set = {row[0] for row in user_groups}
        if set(my_groups) & user_set:
            similar_count += 1
    
    return np.array([[
        len(my_groups),
        group_pop,
        log_pop,
        avg_jaccard,
        similar_count
    ]])

# Получение названия группы
def get_group_name_safe(group_id):
    name = get_group_name_from_db(group_id)
    if name:
        return name
    try:
        import requests
        import os
        from dotenv import load_dotenv
        load_dotenv()
        TOKEN = os.getenv("VK_TOKEN")
        V = os.getenv("VK_API_VERSION", "5.131")
        url = f"https://api.vk.com/method/groups.getById?group_id={group_id}&access_token={TOKEN}&v={V}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "response" in data and data["response"]:
            name = data["response"][0]["name"]
            save_group_name(group_id, name)
            time.sleep(0.34)
            return name
    except:
        pass
    return "unknown"

# Предсказание
def predict(group_id):
    features = extract_features(group_id)
    features_scaled = scaler.transform(features)
    return model.predict_proba(features_scaled)[0][1]

# Кандидаты
all_groups = list(group_to_users.keys())
candidates = [g for g in all_groups if g not in my_groups]

print("Predicting...")
preds = [(g, predict(g)) for g in candidates[:500]]
preds.sort(key=lambda x: x[1], reverse=True)

print("Fetching names for top-30...")
top_30 = []
for g, p in preds[:30]:
    name = get_group_name_safe(g)
    top_30.append((g, name, p))

print("\n" + "="*70)
print("🎯 TOP-30 RECOMMENDATIONS (ML Model)")
print("="*70)
for i, (g, name, p) in enumerate(top_30, 1):
    print(f"{i:2d}. {g:10d} | {name[:45]:45} | {p:.2%}")
print("="*70)

with open("ml_recommendations.txt", "w", encoding="utf-8") as f:
    f.write("ML Model Recommendations\n" + "="*70 + "\n")
    for i, (g, name, p) in enumerate(top_30, 1):
        f.write(f"{i:2d}. {g} | {name} | {p:.2%}\n")

print("\n✅ Saved to ml_recommendations.txt")