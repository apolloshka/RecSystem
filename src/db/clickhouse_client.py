import clickhouse_connect
import os
from dotenv import load_dotenv

load_dotenv('/opt/airflow/project/.env')  # путь к .env файлу

def get_client():
    return clickhouse_connect.get_client(
        host=os.getenv('CLICKHOUSE_HOST', 'clickhouse'),
        port=int(os.getenv('CLICKHOUSE_PORT', 8123)),
        username=os.getenv('CLICKHOUSE_USER', 'vk_user'),
        password=os.getenv('CLICKHOUSE_PASSWORD', 'vk_password'),
        database=os.getenv('CLICKHOUSE_DB', 'vk_diploma')
    )

# Функции для основных таблиц
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
    rows = []
    for group_id in group_ids:
        rows.append((int(group_id),))
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
    rows = []
    for member_id in member_ids:
        rows.append((int(source_group_id), int(member_id)))
    if rows:
        client.insert("group_members", rows, column_names=["source_group_id", "member_id"])


# Функции для таблиц рекомендаций
def create_user_based_table():
    """Создает таблицу для user-based рекомендаций, если её нет"""
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
        print("✓ User-based recommendations table ready")
    except Exception as e:
        print(f"⚠ Error creating user-based table: {e}")

def create_item_based_table():
    """Создает таблицу для item-based рекомендаций, если её нет"""
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
        print("✓ Item-based recommendations table ready")
    except Exception as e:
        print(f"⚠ Error creating item-based table: {e}")

def create_baseline_table():
    """Создает таблицу для baseline рекомендаций, если её нет"""
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
        print("✓ Baseline recommendations table ready")
    except Exception as e:
        print(f"⚠ Error creating baseline table: {e}")

def create_all_recommendation_tables():
    """Создает все таблицы для рекомендаций"""
    create_user_based_table()
    create_item_based_table()
    create_baseline_table()

def truncate_user_based():
    client = get_client()
    client.command("TRUNCATE TABLE IF EXISTS user_based_recommendations")

def truncate_item_based():
    client = get_client()
    client.command("TRUNCATE TABLE IF EXISTS item_based_recommendations")

def truncate_baseline():
    client = get_client()
    client.command("TRUNCATE TABLE IF EXISTS baseline_recommendations")

def truncate_recommendations():
    """Очищает все таблицы с рекомендациями перед новым запуском"""
    truncate_user_based()
    truncate_item_based()
    truncate_baseline()
    print("✓ Truncated all recommendation tables")

def insert_user_based_recommendations(recommendations):
    """Сохраняет user-based рекомендации в ClickHouse
    recommendations: список кортежей (recommended_group_id, group_name, score)
    """
    client = get_client()
    rows = []
    for group_id, group_name, score in recommendations:
        rows.append((int(group_id), str(group_name), float(score)))
    
    if rows:
        client.insert(
            "user_based_recommendations",
            rows,
            column_names=["recommended_group_id", "group_name", "score"]
        )
        print(f"✓ Inserted {len(rows)} user-based recommendations to ClickHouse")
    else:
        print("⚠ No user-based recommendations to insert")

def insert_item_based_recommendations(recommendations):
    """Сохраняет item-based рекомендации в ClickHouse
    recommendations: список кортежей (recommended_group_id, group_name, score)
    """
    client = get_client()
    rows = []
    for group_id, group_name, score in recommendations:
        rows.append((int(group_id), str(group_name), float(score)))
    
    if rows:
        client.insert(
            "item_based_recommendations",
            rows,
            column_names=["recommended_group_id", "group_name", "score"]
        )
        print(f"✓ Inserted {len(rows)} item-based recommendations to ClickHouse")
    else:
        print("⚠ No item-based recommendations to insert")

def insert_baseline_recommendations(recommendations):
    """Сохраняет baseline рекомендации (популярные группы) в ClickHouse
    recommendations: список кортежей (group_id, group_name, members_count)
    """
    client = get_client()
    rows = []
    for group_id, group_name, members_count in recommendations:
        rows.append((int(group_id), str(group_name), int(members_count)))
    
    if rows:
        client.insert(
            "baseline_recommendations",
            rows,
            column_names=["group_id", "group_name", "members_count"]
        )
        print(f"✓ Inserted {len(rows)} baseline recommendations to ClickHouse")
    else:
        print("⚠ No baseline recommendations to insert")


# Функции для таблиц похожести
def create_user_similarity_table():
    """Создает таблицу для хранения похожих пользователей"""
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
        print("✓ User similarity table ready")
    except Exception as e:
        print(f"⚠ Error creating user similarity table: {e}")

def create_group_similarity_table():
    """Создает таблицу для хранения похожих групп"""
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
        print("✓ Group similarity table ready")
    except Exception as e:
        print(f"⚠ Error creating group similarity table: {e}")

def create_all_similarity_tables():
    """Создает все таблицы для похожести"""
    create_user_similarity_table()
    create_group_similarity_table()

def truncate_user_similarity():
    """Очищает таблицу похожести пользователей"""
    client = get_client()
    client.command("TRUNCATE TABLE IF EXISTS user_similarity")
    print("✓ Truncated user_similarity table")

def truncate_group_similarity():
    """Очищает таблицу похожести групп"""
    client = get_client()
    client.command("TRUNCATE TABLE IF EXISTS group_similarity")
    print("✓ Truncated group_similarity table")

def truncate_all_similarity():
    """Очищает все таблицы похожести"""
    truncate_user_similarity()
    truncate_group_similarity()
    print("✓ Truncated all similarity tables")

def insert_user_similarity(similar_users):
    """Сохраняет похожих пользователей
    similar_users: список кортежей (user_id, similarity, common_groups_count)
    """
    client = get_client()
    rows = []
    for user_id, similarity, common_groups in similar_users:
        rows.append((int(user_id), float(similarity), int(common_groups)))
    
    if rows:
        client.insert("user_similarity", rows, column_names=["user_id", "similarity", "common_groups_count"])
        print(f"✓ Inserted {len(rows)} similar users to ClickHouse")
    else:
        print("⚠ No similar users to insert")

def insert_group_similarity(similar_groups):
    """Сохраняет похожие группы
    similar_groups: список кортежей (source_group_id, target_group_id, similarity, common_users_count)
    """
    client = get_client()
    rows = []
    for source, target, similarity, common_users in similar_groups:
        rows.append((int(source), int(target), float(similarity), int(common_users)))
    
    if rows:
        client.insert("group_similarity", rows, column_names=["source_group_id", "target_group_id", "similarity", "common_users_count"])
        print(f"✓ Inserted {len(rows)} similar group pairs to ClickHouse")
    else:
        print("⚠ No similar groups to insert")
# Функции для ML-датасета
def create_ml_dataset_table():
    """Создает таблицу для ML-датасета"""
    client = get_client()
    try:
        client.command("""
            CREATE TABLE IF NOT EXISTS ml_dataset (
                user_id UInt64,
                candidate_group_id UInt64,
                label UInt8,

                user_group_count UInt32,
                group_popularity UInt32,
                log_group_popularity Float64,

                similar_users_in_group_count UInt32,
                sum_similarity_to_group_members Float64,
                max_similarity_to_group_members Float64,
                user_cf_score Float64,

                max_item_similarity Float64,
                sum_item_similarity Float64,
                similar_user_groups_count UInt32,
                item_cf_score Float64,

                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (user_id, candidate_group_id, created_at)
        """)
        print("✓ ML dataset table ready")
    except Exception as e:
        print(f"⚠ Error creating ml_dataset table: {e}")


def truncate_ml_dataset():
    """Очищает таблицу ML-датасета"""
    client = get_client()
    client.command("TRUNCATE TABLE IF EXISTS ml_dataset")
    print("✓ Truncated ml_dataset table")


def insert_ml_dataset(rows):
    """
    Сохраняет строки ML-датасета в ClickHouse

    rows: список кортежей
    (
        user_id,
        candidate_group_id,
        label,
        user_group_count,
        group_popularity,
        log_group_popularity,
        similar_users_in_group_count,
        sum_similarity_to_group_members,
        max_similarity_to_group_members,
        user_cf_score,
        max_item_similarity,
        sum_item_similarity,
        similar_user_groups_count,
        item_cf_score
    )
    """
    client = get_client()
    prepared_rows = []

    for row in rows:
        prepared_rows.append((
            int(row[0]),   # user_id
            int(row[1]),   # candidate_group_id
            int(row[2]),   # label

            int(row[3]),   # user_group_count
            int(row[4]),   # group_popularity
            float(row[5]), # log_group_popularity

            int(row[6]),   # similar_users_in_group_count
            float(row[7]), # sum_similarity_to_group_members
            float(row[8]), # max_similarity_to_group_members
            float(row[9]), # user_cf_score

            float(row[10]), # max_item_similarity
            float(row[11]), # sum_item_similarity
            int(row[12]),   # similar_user_groups_count
            float(row[13])  # item_cf_score
        ))

    if prepared_rows:
        client.insert(
            "ml_dataset",
            prepared_rows,
            column_names=[
                "user_id",
                "candidate_group_id",
                "label",
                "user_group_count",
                "group_popularity",
                "log_group_popularity",
                "similar_users_in_group_count",
                "sum_similarity_to_group_members",
                "max_similarity_to_group_members",
                "user_cf_score",
                "max_item_similarity",
                "sum_item_similarity",
                "similar_user_groups_count",
                "item_cf_score"
            ]
        )
        print(f"✓ Inserted {len(prepared_rows)} rows into ml_dataset")
    else:
        print("⚠ No ML dataset rows to insert")
