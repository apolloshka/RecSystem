from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"

default_args = {
    "owner": "polina",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="vk_recommendation_pipeline",
    default_args=default_args,
    description="Pipeline for VK community recommendation system",
    start_date=datetime(2026, 3, 1),
    schedule=None,
    catchup=False,
    tags=["vk", "recommender"],
) as dag:

    get_my_groups = BashOperator(
        task_id="get_my_groups",
        bash_command=f"cd {PROJECT_DIR} && python get_my_groups.py",
    )

    collect_members = BashOperator(
        task_id="collect_members",
        bash_command=f"cd {PROJECT_DIR} && python collect_members.py",
    )

    collect_user_groups = BashOperator(
        task_id="collect_user_groups",
        bash_command=f"cd {PROJECT_DIR} && python collect_user_groups.py",
    )

    baseline = BashOperator(
        task_id="baseline",
        bash_command=f"cd {PROJECT_DIR} && python baseline.py",
    )

    user_based = BashOperator(
        task_id="user_based",
        bash_command=f"cd {PROJECT_DIR} && python user_based_recommend.py",
    )

    item_based = BashOperator(
        task_id="item_based",
        bash_command=f"cd {PROJECT_DIR} && python item_based_recommend.py",
    )

    compare_algorithms = BashOperator(
        task_id="compare_algorithms",
        bash_command=f"cd {PROJECT_DIR} && python compare_algorithms.py",
    )

    get_my_groups >> collect_members >> collect_user_groups
    collect_user_groups >> [baseline, user_based, item_based] >> compare_algorithms