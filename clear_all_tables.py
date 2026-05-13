from src.db.clickhouse_client import get_client

TABLES_TO_TRUNCATE = [
    "baseline_recommendations",
    "group_members",
    "group_similarity",
    "groups",
    "item_based_recommendations",
    "ml_dataset",
    "my_groups",
    "user_based_recommendations",
    "user_groups",
    "user_similarity",
]


def main():
    client = get_client()

    print("=== Clearing all project tables ===")

    for table in TABLES_TO_TRUNCATE:
        try:
            print(f"Truncating {table}...")
            client.command(f"TRUNCATE TABLE IF EXISTS {table}")
            print(f"  OK: {table}")
        except Exception as e:
            print(f"  ERROR: {table} -> {e}")

    print("=== Done ===")


if __name__ == "__main__":
    main()