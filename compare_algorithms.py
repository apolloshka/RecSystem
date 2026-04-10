import os
import time
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from src.db.clickhouse_client import get_client, get_group_name_from_db, save_group_name


TOP_RECS_TO_LOAD = 30
TOP_RECS_TO_PRINT = 10
TOP_SIM_TO_PRINT = 10
SAVE_FULL_DETAILS_TO_FILE = True

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")
API_URL = "https://api.vk.com/method/"


def vk_call(method, params=None):
    params = params or {}
    params["access_token"] = TOKEN
    params["v"] = V

    try:
        r = requests.get(API_URL + method, params=params, timeout=30)
        data = r.json()
    except Exception:
        return None

    if "error" in data:
        return None

    return data["response"]


def get_group_names_batch(group_ids):
    group_names = {}

    missing_ids = []
    for gid in group_ids:
        name = get_group_name_from_db(int(gid))
        if name and str(name).strip():
            group_names[int(gid)] = name
        else:
            missing_ids.append(int(gid))

    batch_size = 500
    for i in range(0, len(missing_ids), batch_size):
        batch = missing_ids[i:i + batch_size]
        batch_str = ",".join(map(str, batch))

        response = vk_call("groups.getById", {"group_ids": batch_str})

        if response and isinstance(response, list):
            for group in response:
                if "id" in group and "name" in group:
                    gid = int(group["id"])
                    name = group["name"]
                    group_names[gid] = name
                    try:
                        save_group_name(gid, name)
                    except Exception:
                        pass
            time.sleep(0.34)
        else:
            for gid in batch:
                if gid not in group_names:
                    group_names[gid] = "unknown"

    return group_names


def fill_missing_names(df):
    if df.empty:
        return df

    df = df.copy()

    def is_missing(x):
        return pd.isna(x) or str(x).strip() == "" or str(x).strip().lower() == "unknown"

    missing_mask = df["name"].apply(is_missing)
    if not missing_mask.any():
        return df

    missing_ids = df.loc[missing_mask, "group"].astype(int).unique().tolist()
    fetched_names = get_group_names_batch(missing_ids)

    for idx, row in df.loc[missing_mask].iterrows():
        gid = int(row["group"])
        df.at[idx, "name"] = fetched_names.get(gid, "unknown")

    return df


def load_recommendations_from_clickhouse(limit=30):
    client = get_client()

    baseline_result = client.query(f"""
        SELECT group_id, group_name, members_count
        FROM baseline_recommendations
        ORDER BY members_count DESC
        LIMIT {limit}
    """)

    user_result = client.query(f"""
        SELECT recommended_group_id, group_name, score
        FROM user_based_recommendations
        ORDER BY score DESC
        LIMIT {limit}
    """)

    item_result = client.query(f"""
        SELECT recommended_group_id, group_name, score
        FROM item_based_recommendations
        ORDER BY score DESC
        LIMIT {limit}
    """)

    baseline_recs = [(row[0], row[1], row[2]) for row in baseline_result.result_rows]
    user_recs = [(row[0], row[1], row[2]) for row in user_result.result_rows]
    item_recs = [(row[0], row[1], row[2]) for row in item_result.result_rows]

    return baseline_recs, user_recs, item_recs


def load_ml_recommendations_from_file(filename="ml_recommendations.txt", limit=30):
    if not os.path.exists(filename):
        return []

    recs = []
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or "|" not in line:
            continue

        # ожидаем формат:
        #  1. 12345 | Group Name | 95.12%
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue

        left, name, prob_str = parts

        try:
            gid_part = left.split(".")[-1].strip()
            gid = int(gid_part)

            prob_str = prob_str.replace("%", "").strip()
            score = float(prob_str) / 100.0

            recs.append((gid, name, score))
        except Exception:
            continue

    return recs[:limit]


def load_similarity_from_clickhouse(limit_users=100, limit_groups=100):
    client = get_client()

    user_sim_result = client.query(f"""
        SELECT user_id, similarity, common_groups_count
        FROM user_similarity
        WHERE similarity < 0.9999
        ORDER BY similarity DESC
        LIMIT {limit_users}
    """)

    group_sim_result = client.query(f"""
        SELECT source_group_id, target_group_id, similarity, common_users_count
        FROM group_similarity
        ORDER BY similarity DESC
        LIMIT {limit_groups}
    """)

    user_sim = [(row[0], row[1], row[2]) for row in user_sim_result.result_rows]
    group_sim = [(row[0], row[1], row[2], row[3]) for row in group_sim_result.result_rows]

    return user_sim, group_sim


def compute_intersection_info(set_a, set_b):
    intersection = set_a & set_b
    union = set_a | set_b
    overlap = len(intersection) / len(union) if union else 0
    return intersection, overlap


def build_name_map(*dfs):
    names = {}
    for df in dfs:
        if df.empty:
            continue
        for _, row in df.iterrows():
            gid = row["group"]
            name = row["name"] if pd.notna(row["name"]) else "unknown"
            if gid not in names and str(name).strip():
                names[gid] = name
    return names


def print_short_table(title, df, n=10):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    if df.empty:
        print("No data")
        return

    print(df.head(n).to_string(index=False))

    if len(df) > n:
        print(f"... и еще {len(df) - n} строк(и) сохранены в comparison_results.txt")


def print_short_similarity_stats(title, df, common_col, n=10):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    if df.empty:
        print("No data")
        return

    print(f"\nTop {min(n, len(df))}:")
    print(df.head(n).to_string(index=False))

    print("\nStats:")
    print(f"  Max similarity: {df['similarity'].max():.4f}")
    print(f"  Avg similarity: {df['similarity'].mean():.4f}")
    print(f"  Median similarity: {df['similarity'].median():.4f}")
    print(f"  Max {common_col}: {df[common_col].max()}")
    print(f"  Avg {common_col}: {df[common_col].mean():.4f}")

    if len(df) > n:
        print("... полные данные сохранены в comparison_results.txt")


def print_intersection_summary(title, set_a, set_b, names_map, max_names=10):
    intersection, overlap = compute_intersection_info(set_a, set_b)

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Общие рекомендации: {len(intersection)}")
    print(f"Overlap (Jaccard): {round(overlap, 4)}")

    if not intersection:
        print("Нет пересечений")
        return

    print("Примеры пересечений:")
    for gid in sorted(list(intersection))[:max_names]:
        print(f"  {gid} | {names_map.get(gid, 'unknown')}")

    if len(intersection) > max_names:
        print(f"  ... и еще {len(intersection) - max_names}")


def plot_user_similarity(df_user_sim):
    if df_user_sim.empty:
        print("No user similarity data for plotting")
        return

    df_plot = df_user_sim.sort_values("similarity", ascending=False).reset_index(drop=True)
    df_plot["rank"] = df_plot.index + 1

    plt.figure(figsize=(14, 6))
    plt.bar(df_plot["rank"], df_plot["similarity"])
    plt.title("User-based CF: Top similar users")
    plt.xlabel("User rank")
    plt.ylabel("Similarity")
    plt.ylim(0, max(df_plot["similarity"].max() * 1.1, 0.05))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("user_similarity_top100.png", dpi=150)
    plt.close()

    print("✓ Saved: user_similarity_top100.png")


def plot_group_similarity(df_group_sim):
    if df_group_sim.empty:
        print("No group similarity data for plotting")
        return

    df_plot = df_group_sim.sort_values("similarity", ascending=False).reset_index(drop=True)
    df_plot["rank"] = df_plot.index + 1

    plt.figure(figsize=(14, 6))
    plt.bar(df_plot["rank"], df_plot["similarity"])
    plt.title("Item-based CF: Top similar group pairs")
    plt.xlabel("Group-pair rank")
    plt.ylabel("Similarity")
    plt.ylim(0, max(df_plot["similarity"].max() * 1.1, 0.05))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("group_similarity_top100.png", dpi=150)
    plt.close()

    print("✓ Saved: group_similarity_top100.png")


def plot_user_similarity_vs_common_groups(df_user_sim):
    if df_user_sim.empty:
        print("No user similarity scatter data for plotting")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(df_user_sim["common_groups"], df_user_sim["similarity"])
    plt.title("User-based CF: Similarity vs common groups count")
    plt.xlabel("Common groups count")
    plt.ylabel("Similarity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("user_similarity_vs_common_groups.png", dpi=150)
    plt.close()

    print("✓ Saved: user_similarity_vs_common_groups.png")


def plot_group_similarity_vs_common_users(df_group_sim):
    if df_group_sim.empty:
        print("No group similarity scatter data for plotting")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(df_group_sim["common_users"], df_group_sim["similarity"])
    plt.title("Item-based CF: Similarity vs common users count")
    plt.xlabel("Common users count")
    plt.ylabel("Similarity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("group_similarity_vs_common_users.png", dpi=150)
    plt.close()

    print("✓ Saved: group_similarity_vs_common_users.png")


def save_results_to_file(
    df_baseline,
    df_user,
    df_item,
    df_ml,
    df_user_sim,
    df_group_sim,
    summary_table,
    names_map,
    algorithm_sets,
    filename="comparison_results.txt"
):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("RECOMMENDATION ALGORITHMS COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        f.write("BASELINE RECOMMENDATIONS:\n")
        f.write(df_baseline.to_string(index=False) if not df_baseline.empty else "No data")
        f.write("\n\n")

        f.write("USER-BASED RECOMMENDATIONS:\n")
        f.write(df_user.to_string(index=False) if not df_user.empty else "No data")
        f.write("\n\n")

        f.write("ITEM-BASED RECOMMENDATIONS:\n")
        f.write(df_item.to_string(index=False) if not df_item.empty else "No data")
        f.write("\n\n")

        f.write("ML RECOMMENDATIONS:\n")
        f.write(df_ml.to_string(index=False) if not df_ml.empty else "No data")
        f.write("\n\n")

        f.write("USER-BASED SIMILARITY ANALYSIS:\n")
        f.write("=" * 80 + "\n")
        if not df_user_sim.empty:
            f.write(df_user_sim.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("No user similarity data\n\n")

        f.write("ITEM-BASED SIMILARITY ANALYSIS:\n")
        f.write("=" * 80 + "\n")
        if not df_group_sim.empty:
            f.write(df_group_sim.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("No group similarity data\n\n")

        f.write("INTERSECTIONS:\n")
        f.write("=" * 80 + "\n")

        keys = list(algorithm_sets.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a_name = keys[i]
                b_name = keys[j]
                a = algorithm_sets[a_name]
                b = algorithm_sets[b_name]

                intersection, overlap = compute_intersection_info(a, b)
                title = f"{a_name} vs {b_name}"

                f.write(f"\n{title}\n")
                f.write("-" * len(title) + "\n")
                f.write(f"Common recommendations: {len(intersection)}\n")
                f.write(f"Overlap (Jaccard): {round(overlap, 4)}\n")
                if intersection:
                    f.write("Intersecting groups:\n")
                    for gid in sorted(intersection):
                        f.write(f"  {gid} | {names_map.get(gid, 'unknown')}\n")
                else:
                    f.write("No intersections\n")

        f.write("\n\nSUMMARY TABLE:\n")
        f.write(summary_table.to_string(index=False))
        f.write("\n")


def main():
    try:
        print("Loading recommendations from ClickHouse...")
        baseline_recs, user_recs, item_recs = load_recommendations_from_clickhouse(limit=TOP_RECS_TO_LOAD)
        print("✓ Loaded recommendations from ClickHouse")

        print("Loading ML recommendations from file...")
        ml_recs = load_ml_recommendations_from_file(limit=TOP_RECS_TO_LOAD)
        print(f"✓ Loaded {len(ml_recs)} ML recommendations from ml_recommendations.txt")

        print("Loading similarity data from ClickHouse...")
        user_sim, group_sim = load_similarity_from_clickhouse(limit_users=100, limit_groups=100)
        print(f"✓ Loaded {len(user_sim)} similar users and {len(group_sim)} similar group pairs")

    except Exception as e:
        print(f"⚠ Could not load data: {e}")
        baseline_recs, user_recs, item_recs, ml_recs = [], [], [], []
        user_sim, group_sim = [], []

    df_baseline = pd.DataFrame(baseline_recs, columns=["group", "name", "score"])
    df_user = pd.DataFrame(user_recs, columns=["group", "name", "score"])
    df_item = pd.DataFrame(item_recs, columns=["group", "name", "score"])
    df_ml = pd.DataFrame(ml_recs, columns=["group", "name", "score"])

    df_baseline = fill_missing_names(df_baseline)
    df_user = fill_missing_names(df_user)
    df_item = fill_missing_names(df_item)
    df_ml = fill_missing_names(df_ml)

    df_user_sim = pd.DataFrame(user_sim, columns=["user_id", "similarity", "common_groups"])
    if not df_user_sim.empty:
        df_user_sim = df_user_sim[df_user_sim["similarity"] < 0.9999].copy()

    df_group_sim = pd.DataFrame(
        group_sim,
        columns=["source_group_id", "target_group_id", "similarity", "common_users"]
    )

    names_map = build_name_map(df_baseline, df_user, df_item, df_ml)

    print_short_table("BASELINE RECOMMENDATIONS (top preview)", df_baseline, TOP_RECS_TO_PRINT)
    print_short_table("USER-BASED RECOMMENDATIONS (top preview)", df_user, TOP_RECS_TO_PRINT)
    print_short_table("ITEM-BASED RECOMMENDATIONS (top preview)", df_item, TOP_RECS_TO_PRINT)
    print_short_table("ML RECOMMENDATIONS (top preview)", df_ml, TOP_RECS_TO_PRINT)

    print_short_similarity_stats(
        "USER-BASED SIMILARITY",
        df_user_sim,
        common_col="common_groups",
        n=TOP_SIM_TO_PRINT
    )

    print_short_similarity_stats(
        "ITEM-BASED SIMILARITY",
        df_group_sim,
        common_col="common_users",
        n=TOP_SIM_TO_PRINT
    )

    algorithm_dfs = {
        "baseline": df_baseline,
        "user_based": df_user,
        "item_based": df_item,
        "ml": df_ml,
    }

    algorithm_sets = {
        name: set(df["group"]) if not df.empty else set()
        for name, df in algorithm_dfs.items()
    }

    print("\n" + "=" * 60)
    print("COMPARISON OF ALGORITHMS")
    print("=" * 60)

    keys = list(algorithm_sets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_name = keys[i]
            b_name = keys[j]
            print_intersection_summary(
                f"{a_name.upper()} vs {b_name.upper()}",
                algorithm_sets[a_name],
                algorithm_sets[b_name],
                names_map,
                max_names=10
            )

    summary_table = pd.DataFrame({
        "algorithm": ["baseline", "user_based", "item_based", "ml"],
        "count_recommendations": [
            len(df_baseline),
            len(df_user),
            len(df_item),
            len(df_ml),
        ],
        "avg_score": [
            df_baseline["score"].mean() if len(df_baseline) > 0 else 0,
            df_user["score"].mean() if len(df_user) > 0 else 0,
            df_item["score"].mean() if len(df_item) > 0 else 0,
            df_ml["score"].mean() if len(df_ml) > 0 else 0,
        ],
        "max_score": [
            df_baseline["score"].max() if len(df_baseline) > 0 else 0,
            df_user["score"].max() if len(df_user) > 0 else 0,
            df_item["score"].max() if len(df_item) > 0 else 0,
            df_ml["score"].max() if len(df_ml) > 0 else 0,
        ]
    })

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(summary_table.to_string(index=False))

    if SAVE_FULL_DETAILS_TO_FILE:
        save_results_to_file(
            df_baseline,
            df_user,
            df_item,
            df_ml,
            df_user_sim,
            df_group_sim,
            summary_table,
            names_map,
            algorithm_sets,
        )
        print("\n✓ Full comparison results saved to comparison_results.txt")

    print("\n" + "=" * 60)
    print("PLOTTING USER-BASED AND ITEM-BASED SIMILARITY")
    print("=" * 60)

    plot_user_similarity(df_user_sim)
    plot_user_similarity_vs_common_groups(df_user_sim)
    plot_group_similarity(df_group_sim)
    plot_group_similarity_vs_common_users(df_group_sim)

    print("\n✓ Done")


if __name__ == "__main__":
    main()