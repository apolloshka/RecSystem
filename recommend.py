import json

# загрузка данных
with open("my_groups.json") as f:
    my_groups = set(json.load(f))

with open("user_groups.json") as f:
    user_groups = json.load(f)

def jaccard(a, b):
    a = set(a)
    b = set(b)

    inter = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0

    return inter / union


similar_users = []

for uid, groups in user_groups.items():

    sim = jaccard(my_groups, groups)

    if sim > 0:
        similar_users.append((uid, sim, groups))


# сортируем по похожести
similar_users.sort(key=lambda x: x[1], reverse=True)

top_users = similar_users[:50]


# считаем рекомендации
scores = {}

for uid, sim, groups in top_users:

    for g in groups:

        if g in my_groups:
            continue

        scores[g] = scores.get(g, 0) + sim


# сортируем
recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("Top recommendations:")

for g, score in recommendations[:20]:
    print(g, score)