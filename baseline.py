import json
from collections import defaultdict

MIN_GROUP_SIZE = 5


with open("my_groups.json", "r", encoding="utf-8") as f:
    my_groups = set(json.load(f))

with open("user_groups.json", "r", encoding="utf-8") as f:
    user_groups = json.load(f)


group_popularity = defaultdict(int)

for groups in user_groups.values():
    for g in groups:
        group_popularity[g] += 1


recommendations = []

for group, popularity in group_popularity.items():
    if group in my_groups:
        continue

    if popularity < MIN_GROUP_SIZE:
        continue

    recommendations.append((group, popularity))


recommendations.sort(key=lambda x: x[1], reverse=True)

print("Top baseline recommendations:\n")
for g, score in recommendations[:20]:
    print(g, score)