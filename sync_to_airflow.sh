#!/bin/bash
echo "Syncing project files to Airflow container (excluding .venv)..."

docker cp dags airflow_webserver:/opt/airflow/
docker cp src airflow_webserver:/opt/airflow/project/
docker cp *.py airflow_webserver:/opt/airflow/project/
docker cp .env airflow_webserver:/opt/airflow/project/
docker cp schema.sql airflow_webserver:/opt/airflow/project/
docker cp requirements.txt airflow_webserver:/opt/airflow/project/ 2>/dev/null

echo "Done! Files synced to Airflow container"