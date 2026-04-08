import clickhouse_connect
import os
from dotenv import load_dotenv

load_dotenv('/opt/airflow/project/.env')

def get_client():
    return clickhouse_connect.get_client(
        host=os.getenv('CLICKHOUSE_HOST', 'clickhouse'),
        port=int(os.getenv('CLICKHOUSE_PORT', 8123)),
        username=os.getenv('CLICKHOUSE_USER', 'vk_user'),
        password=os.getenv('CLICKHOUSE_PASSWORD', 'vk_password'),
        database=os.getenv('CLICKHOUSE_DB', 'vk_diploma')
    )

# Основные таблицы
def truncate_my_groups():
    client = get_client()
    client.command("TRUNCATE TABLE my_groups")

def truncate_user_groups():
    client = get_client()
    client.command("TRUNCATE TABLE user_groups")

def truncate_group_members():
    client = get_client()
    client.command("TRUNCATE TABLE group_members")

def insert_my_groups(group_ids):
    client = get_client()
    rows = [(int(g),) for g in group_ids]
    if rows:
        client.insert("my_groups", rows, column_names=["group_id"])

def insert_user_groups(user_groups: dict):
    client = get_client()
    rows = []
    for user_id, groups in user_groups.items():
        for group_id in groups:
            rows.append((int(user_id), int(group_id)))
    if rows:
        client.insert("user_groups", rows, column_names=["user_id", "group_id"])

def insert_group_members(source_group_id, member_ids):
    client = get_client()
    rows = [(int(source_group_id), int(m)) for m in member_ids]
    if rows:
        client.insert("group_members", rows, column_names=["source_group_id", "member_id"])

# Таблицы рекомендаций
def create_user_based_table():
    client = get_client()
    try:
        client.command("""
            CREATE TABLE IF NOT EXISTS user_based_recommendations (
                recommended_group_id UInt64,
                group_name String,
                score Float64,
                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (score, created_at)
        """)
    except Exception as e:
        print(f"Error creating user_based table: {e}")

def create_item_based_table():
    client = get_client()
    try:
        client.command("""
            CREATE TABLE IF NOT EXISTS item_based_recommendations (
                recommended_group_id UInt64,
                group_name String,
                score Float64,
                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (score, created_at)
        """)
    except Exception as e:
        print(f"Error creating item_based table: {e}")

def create_baseline_table():
    client = get_client()
    try:
        client.command("""
            CREATE TABLE IF NOT EXISTS baseline_recommendations (
                group_id UInt64,
                group_name String,
                members_count UInt64,
                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (members_count, created_at)
        """)
    except Exception as e:
        print(f"Error creating baseline table: {e}")

def truncate_user_based():
    client = get_client()
    client.command("TRUNCATE TABLE IF EXISTS user_based_recommendations")

def truncate_item_based():
    client = get_client()
    client.command("TRUNCATE TABLE IF EXISTS item_based_recommendations")

def truncate_baseline():
    client = get_client()
    client.command("TRUNCATE TABLE IF EXISTS baseline_recommendations")

def insert_user_based_recommendations(recommendations):
    client = get_client()
    rows = [(int(g), str(n), float(s)) for g, n, s in recommendations]
    if rows:
        client.insert("user_based_recommendations", rows, column_names=["recommended_group_id", "group_name", "score"])

def insert_item_based_recommendations(recommendations):
    client = get_client()
    rows = [(int(g), str(n), float(s)) for g, n, s in recommendations]
    if rows:
        client.insert("item_based_recommendations", rows, column_names=["recommended_group_id", "group_name", "score"])

def insert_baseline_recommendations(recommendations):
    client = get_client()
    rows = [(int(g), str(n), int(c)) for g, n, c in recommendations]
    if rows:
        client.insert("baseline_recommendations", rows, column_names=["group_id", "group_name", "members_count"])

# Таблицы похожести
def create_user_similarity_table():
    client = get_client()
    try:
        client.command("""
            CREATE TABLE IF NOT EXISTS user_similarity (
                user_id UInt64,
                similarity Float64,
                common_groups_count UInt64,
                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (similarity, created_at)
        """)
    except Exception as e:
        print(f"Error creating user_similarity table: {e}")

def create_group_similarity_table():
    client = get_client()
    try:
        client.command("""
            CREATE TABLE IF NOT EXISTS group_similarity (
                source_group_id UInt64,
                target_group_id UInt64,
                similarity Float64,
                common_users_count UInt64,
                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (source_group_id, similarity, created_at)
        """)
    except Exception as e:
        print(f"Error creating group_similarity table: {e}")

def truncate_user_similarity():
    client = get_client()
    client.command("TRUNCATE TABLE IF EXISTS user_similarity")

def truncate_group_similarity():
    client = get_client()
    client.command("TRUNCATE TABLE IF EXISTS group_similarity")

def insert_user_similarity(similar_users):
    client = get_client()
    rows = [(int(u), float(s), int(c)) for u, s, c in similar_users]
    if rows:
        client.insert("user_similarity", rows, column_names=["user_id", "similarity", "common_groups_count"])

def insert_group_similarity(similar_groups):
    client = get_client()
    rows = [(int(s), int(t), float(sim), int(c)) for s, t, sim, c in similar_groups]
    if rows:
        client.insert("group_similarity", rows, column_names=["source_group_id", "target_group_id", "similarity", "common_users_count"])

# ML датасет
def create_ml_dataset_table(feature_names=None):
    client = get_client()
    if feature_names is None:
        feature_names = ["user_group_count", "group_popularity", "log_group_popularity", "avg_jaccard", "similar_users_count"]
    
    feature_columns = ",\n                ".join([f"{name} Float64" for name in feature_names])
    
    create_query = f"""
        CREATE TABLE IF NOT EXISTS ml_dataset (
            user_id UInt64,
            candidate_group_id UInt64,
            label UInt8,
            {feature_columns},
            created_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (user_id, candidate_group_id, created_at)
    """
    try:
        client.command(create_query)
    except Exception as e:
        print(f"Error creating ml_dataset table: {e}")

def truncate_ml_dataset():
    client = get_client()
    client.command("TRUNCATE TABLE IF EXISTS ml_dataset")

def insert_ml_dataset(rows, feature_names):
    client = get_client()
    if rows:
        column_names = ["user_id", "candidate_group_id", "label"] + feature_names
        client.insert("ml_dataset", rows, column_names=column_names)

def create_groups_table():
    client = get_client()
    try:
        client.command("""
            CREATE TABLE IF NOT EXISTS groups (
                group_id UInt64,
                name String,
                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY group_id
        """)
    except Exception as e:
        print(f"Error creating groups table: {e}")

def get_group_name_from_db(group_id):
    client = get_client()
    try:
        result = client.query(f"SELECT name FROM groups WHERE group_id = {group_id} LIMIT 1")
        if result.result_rows:
            return result.result_rows[0][0]
    except:
        pass
    return None

def save_group_name(group_id, name):
    client = get_client()
    try:
        client.insert("groups", [(int(group_id), name)], column_names=["group_id", "name"])
    except:
        pass