## 1. Получить токен VK

Перейди по ссылке и скопируй токен из адресной строки после авторизации:

👉 [Получить токен VK](https://oauth.vk.com/authorize?client_id=2685278&display=page&redirect_uri=https://oauth.vk.com/blank.html&scope=groups&response_type=token&v=5.131)

Добавь токен в файл `.env` исходя из описания `.env.example`


## 2. Активация окружения 

.\.venv\Scripts\Activate.ps1   # Windows PowerShell

## 3. Запуск инфраструктуры

docker-compose up -d

## 4. Синхронизация файлов с контейнером
После обновления локальных файлов обнови их в контейнере:

docker cp . airflow_webserver:/opt/airflow/project/



Команды для запуска скриптов (внутри контейнера)
Все команды выполняются из PowerShell / терминала:

### 1. Получить мои группы
docker exec -it airflow_webserver bash -c "cd /opt/airflow/project && python get_my_groups.py"

### 2. Собрать участников моих групп 
docker exec -it airflow_webserver bash -c "cd /opt/airflow/project && python collect_members.py"

### 3. Собрать группы участников 
docker exec -it airflow_webserver bash -c "cd /opt/airflow/project && python collect_user_groups.py"

### 4. Baseline
docker exec -it airflow_webserver bash -c "cd /opt/airflow/project && python baseline.py"

### 5. User-based рекомендации
docker exec -it airflow_webserver bash -c "cd /opt/airflow/project && python user_based_recommend.py"

### 6. Item-based рекомендации
docker exec -it airflow_webserver bash -c "cd /opt/airflow/project && python item_based_recommend.py"

### 7. compare_algorithms (сравнивает алгоритмы)
docker exec -it airflow_webserver bash -c "cd /opt/airflow/project && python compare_algorithms.py"

### 8. build_ml_dataset 
docker exec -it airflow_webserver bash -c "cd /opt/airflow/project && python build_ml_dataset.py"

### 9. train_logistic_regression
docker exec -it airflow_webserver bash -c "cd /opt/airflow/project && python train_logistic_regression.py"

### 10. predict_for_user
docker exec -it airflow_webserver bash -c "cd /opt/airflow/project && python predict_for_user.py"

### 11. просмотр БД
docker exec -it vk_clickhouse clickhouse-client --user vk_user --password vk_password --database vk_diploma

### DAG (Directed Acyclic Graph) — это автоматизированный пайплайн, который запускает все твои скрипты в правильном порядке без ручного вмешательства.

Как запустить:
1. Открыть Airflow UI
В браузере перейди по адресу: http://localhost:8080

Логин: admin
Пароль: admin

2. Найти DAG
На главном экране найди vk_recommendation_pipeline в списке DAG.

3. Запустить DAG
Нажми на кнопку Play (треугольник) справа от названия DAG.









