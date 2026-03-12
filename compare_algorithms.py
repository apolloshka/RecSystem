import json
import matplotlib.pyplot as plt
import pandas as pd

# загрузка результатов
def load_recommendations(file):

    recs = []

    with open(file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("Top") or not line.strip():
                continue

            group, score = line.split()
            recs.append((group, float(score)))

    return recs


user_recs = load_recommendations("user_based_recommendations.txt")
item_recs = load_recommendations("item_based_recommendations.txt")


# превращаем в DataFrame
df_user = pd.DataFrame(user_recs, columns=["group", "score"])
df_item = pd.DataFrame(item_recs, columns=["group", "score"])


print("\nUSER BASED:")
print(df_user.head())

print("\nITEM BASED:")
print(df_item.head())


# -------------------------
# Сравнение метрик
# -------------------------

user_groups = set(df_user["group"])
item_groups = set(df_item["group"])

intersection = user_groups & item_groups
union = user_groups | item_groups

overlap = len(intersection) / len(union) if union else 0

print("\nСравнение алгоритмов")
print("---------------------")

print("User recommendations:", len(user_groups))
print("Item recommendations:", len(item_groups))

print("Общие рекомендации:", len(intersection))
print("Overlap:", overlap)


# -------------------------
# Таблица сравнения
# -------------------------

table = pd.DataFrame({
    "metric": [
        "count_recommendations",
        "avg_score",
        "max_score"
    ],
    "user_based": [
        len(df_user),
        df_user["score"].mean(),
        df_user["score"].max()
    ],
    "item_based": [
        len(df_item),
        df_item["score"].mean(),
        df_item["score"].max()
    ]
})

print("\nТаблица сравнения:")
print(table)


# -------------------------
# График распределения score
# -------------------------

plt.figure(figsize=(8,5))

plt.hist(df_user["score"], bins=10, alpha=0.6, label="user-based")
plt.hist(df_item["score"], bins=10, alpha=0.6, label="item-based")

plt.title("Распределение score рекомендаций")
plt.xlabel("score")
plt.ylabel("frequency")

plt.legend()

plt.savefig("images/score_distribution.png")

plt.show()


# -------------------------
# график топ рекомендаций
# -------------------------

top_user = df_user.head(10)
top_item = df_item.head(10)

plt.figure(figsize=(10,5))

plt.plot(top_user["score"].values, label="user-based")
plt.plot(top_item["score"].values, label="item-based")

plt.title("Top-10 recommendation scores")
plt.xlabel("rank")
plt.ylabel("score")

plt.legend()

plt.savefig("images/top_scores.png")

plt.show()