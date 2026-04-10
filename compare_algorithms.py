import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.db.clickhouse_client import get_client


TOP_RECS_TO_LOAD = 30
TOP_RECS_TO_PRINT = 10
TOP_SIM_TO_PRINT = 10
SAVE_FULL_DETAILS_TO_FILE = True


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
        print(f"... полные данные сохранены в comparison_results.txt")


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
    df_user_sim,
    df_group_sim,
    summary_table,
    baseline_groups,
    user_groups,
    item_groups,
    names_map,
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

        f.write("USER-BASED SIMILARITY ANALYSIS:\n")
        f.write("=" * 80 + "\n")
        if not df_user_sim.empty:
            f.write(df_user_sim.to_string(index=False))
            f.write("\n\n")
            f.write(f"Max similarity: {df_user_sim['similarity'].max():.4f}\n")
            f.write(f"Avg similarity: {df_user_sim['similarity'].mean():.4f}\n")
            f.write(f"Median similarity: {df_user_sim['similarity'].median():.4f}\n")
            f.write(f"Max common groups: {df_user_sim['common_groups'].max()}\n")
            f.write(f"Avg common groups: {df_user_sim['common_groups'].mean():.4f}\n\n")
        else:
            f.write("No user similarity data\n\n")

        f.write("ITEM-BASED SIMILARITY ANALYSIS:\n")
        f.write("=" * 80 + "\n")
        if not df_group_sim.empty:
            f.write(df_group_sim.to_string(index=False))
            f.write("\n\n")
            f.write(f"Max similarity: {df_group_sim['similarity'].max():.4f}\n")
            f.write(f"Avg similarity: {df_group_sim['similarity'].mean():.4f}\n")
            f.write(f"Median similarity: {df_group_sim['similarity'].median():.4f}\n")
            f.write(f"Max common users: {df_group_sim['common_users'].max()}\n")
            f.write(f"Avg common users: {df_group_sim['common_users'].mean():.4f}\n\n")
        else:
            f.write("No group similarity data\n\n")

        f.write("INTERSECTIONS:\n")
        f.write("=" * 80 + "\n")

        for title, a, b in [
            ("BASELINE vs USER-BASED", baseline_groups, user_groups),
            ("BASELINE vs ITEM-BASED", baseline_groups, item_groups),
            ("USER-BASED vs ITEM-BASED", user_groups, item_groups),
        ]:
            intersection, overlap = compute_intersection_info(a, b)
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

        print("Loading similarity data from ClickHouse...")
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

    names_map = build_name_map(df_baseline, df_user, df_item)

    print_short_table("BASELINE RECOMMENDATIONS (top preview)", df_baseline, TOP_RECS_TO_PRINT)
    print_short_table("USER-BASED RECOMMENDATIONS (top preview)", df_user, TOP_RECS_TO_PRINT)
    print_short_table("ITEM-BASED RECOMMENDATIONS (top preview)", df_item, TOP_RECS_TO_PRINT)

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

    if not df_baseline.empty and not df_user.empty and not df_item.empty:
        baseline_groups = set(df_baseline["group"])
        user_groups = set(df_user["group"])
        item_groups = set(df_item["group"])

        print("\n" + "=" * 60)
        print("COMPARISON OF ALGORITHMS")
        print("=" * 60)

        print_intersection_summary(
            "BASELINE vs USER-BASED",
            baseline_groups,
            user_groups,
            names_map,
            max_names=10
        )

        print_intersection_summary(
            "BASELINE vs ITEM-BASED",
            baseline_groups,
            item_groups,
            names_map,
            max_names=10
        )

        print_intersection_summary(
            "USER-BASED vs ITEM-BASED",
            user_groups,
            item_groups,
            names_map,
            max_names=10
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

        if SAVE_FULL_DETAILS_TO_FILE:
            save_results_to_file(
                df_baseline,
                df_user,
                df_item,
                df_user_sim,
                df_group_sim,
                summary_table,
                baseline_groups,
                user_groups,
                item_groups,
                names_map,
            )
            print("\n✓ Full comparison results saved to comparison_results.txt")
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