import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.db.clickhouse_client import get_client


def load_recommendations_from_clickhouse():
    client = get_client()

    baseline_result = client.query("""
        SELECT group_id, group_name, members_count
        FROM baseline_recommendations
        ORDER BY members_count DESC
        LIMIT 30
    """)

    user_result = client.query("""
        SELECT recommended_group_id, group_name, score
        FROM user_based_recommendations
        ORDER BY score DESC
        LIMIT 30
    """)

    item_result = client.query("""
        SELECT recommended_group_id, group_name, score
        FROM item_based_recommendations
        ORDER BY score DESC
        LIMIT 30
    """)

    baseline_recs = [(row[0], row[1], row[2]) for row in baseline_result.result_rows]
    user_recs = [(row[0], row[1], row[2]) for row in user_result.result_rows]
    item_recs = [(row[0], row[1], row[2]) for row in item_result.result_rows]

    return baseline_recs, user_recs, item_recs


def load_similarity_from_clickhouse(limit_users=100, limit_groups=100):
    client = get_client()

    # Убираем self-match с similarity = 1.0
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


def print_intersection(title, set_a, set_b, df_a, df_b):
    intersection = set_a & set_b
    union = set_a | set_b
    overlap = len(intersection) / len(union) if union else 0

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Общие рекомендации: {len(intersection)}")
    print(f"Overlap (Jaccard): {round(overlap, 4)}")

    if not intersection:
        print("Нет пересечений")
        return

    names = {}

    for _, row in df_a.iterrows():
        if row["group"] in intersection:
            names[row["group"]] = row["name"]

    for _, row in df_b.iterrows():
        if row["group"] in intersection and row["group"] not in names:
            names[row["group"]] = row["name"]

    print("\nПересекающиеся группы:")
    for gid in sorted(intersection):
        print(f"  {gid} | {names.get(gid, 'unknown')}")


def plot_user_similarity(df_user_sim):
    if df_user_sim.empty:
        print("No user similarity data for plotting")
        return

    df_plot = df_user_sim.sort_values("similarity", ascending=False).reset_index(drop=True)
    df_plot["rank"] = df_plot.index + 1

    plt.figure(figsize=(14, 6))
    plt.bar(df_plot["rank"], df_plot["similarity"])
    plt.title("User-based CF: Top 100 similar users")
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
    plt.title("Item-based CF: Top 100 similar group pairs")
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
    df_user_sim,
    df_group_sim,
    summary_table,
    baseline_groups,
    user_groups,
    item_groups,
    filename="comparison_results.txt"
):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("RECOMMENDATION ALGORITHMS COMPARISON\n")
        f.write("=" * 60 + "\n\n")

        f.write("BASELINE RECOMMENDATIONS (Popular groups):\n")
        f.write(df_baseline.to_string(index=False) if not df_baseline.empty else "No data")
        f.write("\n\n")

        f.write("USER-BASED RECOMMENDATIONS:\n")
        f.write(df_user.to_string(index=False) if not df_user.empty else "No data")
        f.write("\n\n")

        f.write("ITEM-BASED RECOMMENDATIONS:\n")
        f.write(df_item.to_string(index=False) if not df_item.empty else "No data")
        f.write("\n\n")

        f.write("USER-BASED SIMILARITY ANALYSIS:\n")
        f.write("=" * 60 + "\n\n")
        if not df_user_sim.empty:
            f.write("TOP 10 SIMILAR USERS:\n")
            f.write(df_user_sim.head(10).to_string(index=False))
            f.write("\n\n")
            f.write(f"Max similarity: {df_user_sim['similarity'].max():.4f}\n")
            f.write(f"Avg similarity: {df_user_sim['similarity'].mean():.4f}\n")
            f.write(f"Median similarity: {df_user_sim['similarity'].median():.4f}\n")
            f.write(f"Max common groups: {df_user_sim['common_groups'].max()}\n")
            f.write(f"Avg common groups: {df_user_sim['common_groups'].mean():.4f}\n\n")
        else:
            f.write("No user similarity data\n\n")

        f.write("ITEM-BASED SIMILARITY ANALYSIS:\n")
        f.write("=" * 60 + "\n\n")
        if not df_group_sim.empty:
            f.write("TOP 10 SIMILAR GROUP PAIRS:\n")
            f.write(df_group_sim.head(10).to_string(index=False))
            f.write("\n\n")
            f.write(f"Max similarity: {df_group_sim['similarity'].max():.4f}\n")
            f.write(f"Avg similarity: {df_group_sim['similarity'].mean():.4f}\n")
            f.write(f"Median similarity: {df_group_sim['similarity'].median():.4f}\n")
            f.write(f"Max common users: {df_group_sim['common_users'].max()}\n")
            f.write(f"Avg common users: {df_group_sim['common_users'].mean():.4f}\n\n")
        else:
            f.write("No group similarity data\n\n")

        f.write("SUMMARY TABLE:\n")
        f.write(summary_table.to_string(index=False))
        f.write("\n\n")

        f.write(f"Common between Baseline & User-based: {len(baseline_groups & user_groups)}\n")
        f.write(f"Common between Baseline & Item-based: {len(baseline_groups & item_groups)}\n")
        f.write(f"Common between User-based & Item-based: {len(user_groups & item_groups)}\n")


def main():
    try:
        print("Loading recommendations from ClickHouse...")
        baseline_recs, user_recs, item_recs = load_recommendations_from_clickhouse()
        print("✓ Loaded recommendations from ClickHouse")

        print("\nLoading similarity data from ClickHouse...")
        user_sim, group_sim = load_similarity_from_clickhouse(limit_users=100, limit_groups=100)
        print(f"✓ Loaded {len(user_sim)} similar users and {len(group_sim)} similar group pairs")

    except Exception as e:
        print(f"⚠ Could not load from ClickHouse: {e}")
        baseline_recs, user_recs, item_recs = [], [], []
        user_sim, group_sim = [], []

    df_baseline = pd.DataFrame(baseline_recs, columns=["group", "name", "score"])
    df_user = pd.DataFrame(user_recs, columns=["group", "name", "score"])
    df_item = pd.DataFrame(item_recs, columns=["group", "name", "score"])

    df_user_sim = pd.DataFrame(user_sim, columns=["user_id", "similarity", "common_groups"])
    if not df_user_sim.empty:
        df_user_sim = df_user_sim[df_user_sim["similarity"] < 0.9999].copy()

    df_group_sim = pd.DataFrame(
        group_sim,
        columns=["source_group_id", "target_group_id", "similarity", "common_users"]
    )

    print("\n" + "=" * 60)
    print("BASELINE RECOMMENDATIONS (Popular groups)")
    print("=" * 60)
    print(df_baseline.to_string(index=False) if not df_baseline.empty else "No data")

    print("\n" + "=" * 60)
    print("USER-BASED RECOMMENDATIONS")
    print("=" * 60)
    print(df_user.to_string(index=False) if not df_user.empty else "No data")

    print("\n" + "=" * 60)
    print("ITEM-BASED RECOMMENDATIONS")
    print("=" * 60)
    print(df_item.to_string(index=False) if not df_item.empty else "No data")

    print("\n" + "=" * 60)
    print("USER-BASED SIMILARITY")
    print("=" * 60)
    if not df_user_sim.empty:
        print("\nTop 10 similar users:")
        print(df_user_sim.head(10).to_string(index=False))
        print("\nUser similarity stats:")
        print(f"  Max similarity: {df_user_sim['similarity'].max():.4f}")
        print(f"  Avg similarity: {df_user_sim['similarity'].mean():.4f}")
        print(f"  Median similarity: {df_user_sim['similarity'].median():.4f}")
        print(f"  Max common groups: {df_user_sim['common_groups'].max()}")
        print(f"  Avg common groups: {df_user_sim['common_groups'].mean():.4f}")
    else:
        print("No user similarity data")

    print("\n" + "=" * 60)
    print("ITEM-BASED SIMILARITY")
    print("=" * 60)
    if not df_group_sim.empty:
        print("\nTop 10 similar group pairs:")
        print(df_group_sim.head(10).to_string(index=False))
        print("\nGroup similarity stats:")
        print(f"  Max similarity: {df_group_sim['similarity'].max():.4f}")
        print(f"  Avg similarity: {df_group_sim['similarity'].mean():.4f}")
        print(f"  Median similarity: {df_group_sim['similarity'].median():.4f}")
        print(f"  Max common users: {df_group_sim['common_users'].max()}")
        print(f"  Avg common users: {df_group_sim['common_users'].mean():.4f}")
    else:
        print("No group similarity data")

    if not df_baseline.empty and not df_user.empty and not df_item.empty:
        baseline_groups = set(df_baseline["group"])
        user_groups = set(df_user["group"])
        item_groups = set(df_item["group"])

        print("\n" + "=" * 60)
        print("COMPARISON OF ALGORITHMS")
        print("=" * 60)

        print_intersection(
            "BASELINE vs USER-BASED",
            baseline_groups,
            user_groups,
            df_baseline,
            df_user
        )

        print_intersection(
            "BASELINE vs ITEM-BASED",
            baseline_groups,
            item_groups,
            df_baseline,
            df_item
        )

        print_intersection(
            "USER-BASED vs ITEM-BASED",
            user_groups,
            item_groups,
            df_user,
            df_item
        )

        summary_table = pd.DataFrame({
            "algorithm": ["baseline", "user_based", "item_based"],
            "count_recommendations": [
                len(df_baseline),
                len(df_user),
                len(df_item)
            ],
            "avg_score": [
                df_baseline["score"].mean() if len(df_baseline) > 0 else 0,
                df_user["score"].mean() if len(df_user) > 0 else 0,
                df_item["score"].mean() if len(df_item) > 0 else 0
            ],
            "max_score": [
                df_baseline["score"].max() if len(df_baseline) > 0 else 0,
                df_user["score"].max() if len(df_user) > 0 else 0,
                df_item["score"].max() if len(df_item) > 0 else 0
            ]
        })

        print("\n" + "=" * 60)
        print("SUMMARY TABLE")
        print("=" * 60)
        print(summary_table.to_string(index=False))

        save_results_to_file(
            df_baseline,
            df_user,
            df_item,
            df_user_sim,
            df_group_sim,
            summary_table,
            baseline_groups,
            user_groups,
            item_groups
        )

        print("\n✓ Comparison results saved to comparison_results.txt")

    else:
        print("\n⚠ Not enough recommendation data for full comparison")

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