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
    tags=["vk", "recommender", "ml"],
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

    build_ml_dataset = BashOperator(
        task_id="build_ml_dataset",
        bash_command=f"cd {PROJECT_DIR} && python build_ml_dataset.py",
    )

    train_logistic_regression = BashOperator(
        task_id="train_logistic_regression",
        bash_command=f"cd {PROJECT_DIR} && python train_logistic_regression.py",
    )

    predict_for_user = BashOperator(
        task_id="predict_for_user",
        bash_command=f"cd {PROJECT_DIR} && python predict_for_user.py",
    )

    compare_algorithms = BashOperator(
        task_id="compare_algorithms",
        bash_command=f"cd {PROJECT_DIR} && python compare_algorithms.py",
    )

    # Сбор данных
    get_my_groups >> collect_members >> collect_user_groups

    # Базовые алгоритмы
    collect_user_groups >> baseline
    collect_user_groups >> user_based
    collect_user_groups >> item_based

    # ML (evaluate_ranking удалён, поэтому train_logistic_regression теперь идёт сразу в predict_for_user)
    [collect_user_groups, user_based, item_based] >> build_ml_dataset
    build_ml_dataset >> train_logistic_regression >> predict_for_user

    # Сравнение всех алгоритмов
    [baseline, user_based, item_based, predict_for_user] >> compare_algorithms