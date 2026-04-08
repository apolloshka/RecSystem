import math
import random
from collections import defaultdict
from src.db.clickhouse_client import (
    get_client, create_ml_dataset_table, truncate_ml_dataset, insert_ml_dataset
)

print("Building ML Dataset")

client = get_client()

group_to_users = defaultdict(set)
user_groups_result = client.query("SELECT user_id, group_id FROM user_groups")
for user_id, group_id in user_groups_result.result_rows:
    group_to_users[int(group_id)].add(int(user_id))

my_groups_result = client.query("SELECT group_id FROM my_groups")
my_groups = [row[0] for row in my_groups_result.result_rows]
print(f"My groups: {len(my_groups)}")

user_based_result = client.query("SELECT recommended_group_id FROM user_based_recommendations LIMIT 300")
item_based_result = client.query("SELECT recommended_group_id FROM item_based_recommendations LIMIT 300")

positive_groups = set()
for row in user_based_result.result_rows:
    positive_groups.add(row[0])
for row in item_based_result.result_rows:
    positive_groups.add(row[0])
print(f"Positive groups: {len(positive_groups)}")

group_popularity = {g: len(users) for g, users in group_to_users.items()}
candidates_for_negative = [
    g for g, pop in group_popularity.items()
    if pop > 50 and g not in positive_groups and g not in my_groups
]
negative_groups = random.sample(candidates_for_negative, min(len(positive_groups) * 2, len(candidates_for_negative)))
print(f"Negative groups: {len(negative_groups)}")

def extract_features(group_id):
    group_users = group_to_users.get(group_id, set())
    group_pop = len(group_users)
    log_pop = math.log(1 + group_pop)
    
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
    
    similar_users_count = 0
    for user in list(group_users)[:50]:
        user_groups = client.query(f"SELECT group_id FROM user_groups WHERE user_id = {user} LIMIT 10").result_rows
        user_set = {row[0] for row in user_groups}
        if set(my_groups) & user_set:
            similar_users_count += 1
    
    return {
        "user_group_count": len(my_groups),
        "group_popularity": group_pop,
        "log_group_popularity": log_pop,
        "avg_jaccard": avg_jaccard,
        "similar_users_count": similar_users_count
    }

sample_features = extract_features(list(positive_groups)[0]) if positive_groups else {}
feature_names = list(sample_features.keys())
print(f"Features: {feature_names}")

rows = []
USER_ID = 0

for group_id in positive_groups:
    features = extract_features(group_id)
    rows.append([USER_ID, group_id, 1] + [features[f] for f in feature_names])

for group_id in negative_groups:
    features = extract_features(group_id)
    rows.append([USER_ID, group_id, 0] + [features[f] for f in feature_names])

print(f"Total rows: {len(rows)} (positive: {len(positive_groups)}, negative: {len(negative_groups)})")

create_ml_dataset_table(feature_names)
truncate_ml_dataset()
insert_ml_dataset(rows, feature_names)
print("Dataset saved")