import joblib
import math
import time
import random
import numpy as np
from collections import defaultdict
from src.db.clickhouse_client import get_client, get_group_name_from_db, save_group_name

print("=== ML Predictor ===")

client = get_client()
model = joblib.load('recommendation_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

group_to_users = defaultdict(set)
for user_id, group_id in client.query("SELECT user_id, group_id FROM user_groups").result_rows:
    group_to_users[int(group_id)].add(int(user_id))

group_popularity = {g: len(users) for g, users in group_to_users.items()}

my_groups = [row[0] for row in client.query("SELECT group_id FROM my_groups").result_rows]
print(f"My groups: {len(my_groups)}")

# ----- ФИЛЬТРАЦИЯ КАНДИДАТОВ -----
# 1. Берём группы из user_based и item_based рекомендаций
good_groups = set()
for row in client.query("SELECT recommended_group_id FROM user_based_recommendations LIMIT 1000").result_rows:
    good_groups.add(row[0])
for row in client.query("SELECT recommended_group_id FROM item_based_recommendations LIMIT 1000").result_rows:
    good_groups.add(row[0])

# 2. Добавляем популярные группы для разнообразия
popular_candidates = [
    g for g, pop in group_popularity.items() 
    if pop > 100 and g not in good_groups and g not in my_groups
]
popular_sample = random.sample(popular_candidates, min(300, len(popular_candidates)))

# 3. Объединяем
candidates = list(good_groups) + popular_sample
candidates = [g for g in candidates if g not in my_groups]
print(f"Candidates for prediction: {len(candidates)}")

def extract_features(group_id):
    group_users = group_to_users.get(group_id, set())
    group_pop = len(group_users)
    log_pop = math.log(1 + group_pop)
    
    user_score_result = client.query(f"SELECT score FROM user_based_recommendations WHERE recommended_group_id = {group_id} LIMIT 1")
    user_based_score = user_score_result.result_rows[0][0] if user_score_result.result_rows else 0.0
    
    item_score_result = client.query(f"SELECT score FROM item_based_recommendations WHERE recommended_group_id = {group_id} LIMIT 1")
    item_based_score = item_score_result.result_rows[0][0] if item_score_result.result_rows else 0.0
    
    return np.array([[
        group_pop,
        log_pop,
        user_based_score,
        item_based_score
    ]])

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

def predict(group_id):
    features = extract_features(group_id)
    features_scaled = scaler.transform(features)
    return model.predict_proba(features_scaled)[0][1]

print("Predicting...")
preds = [(g, predict(g)) for g in candidates[:1000]]
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