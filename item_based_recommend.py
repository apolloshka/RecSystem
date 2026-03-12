import json
from collections import defaultdict

# загрузка данных
with open("my_groups.json") as f:
    my_groups = set(json.load(f))

with open("user_groups.json") as f:
    user_groups = json.load(f)


# строим обратный индекс
# group -> users
group_users = defaultdict(set)

for user, groups in user_groups.items():
    for g in groups:
        group_users[g].add(user)


def jaccard(a, b):

    inter = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0

    return inter / union


scores = {}

# считаем похожесть групп
for my_group in my_groups:

    users_a = group_users.get(my_group, set())

    for group, users_b in group_users.items():

        if group in my_groups:
            continue

        sim = jaccard(users_a, users_b)

        if sim == 0:
            continue

        scores[group] = scores.get(group, 0) + sim


# сортируем
recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("Top item-based recommendations:")

for g, score in recommendations[:20]:
    print(g, score)


# сохраняем результат
with open("item_based_recommendations.txt", "w", encoding="utf-8") as f:

    f.write("Top item-based recommendations\n\n")

    for g, score in recommendations[:20]:
        f.write(f"{g} {score}\n")

print("\nСохранено в item_based_recommendations.txt")