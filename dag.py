from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# добавление текущей папки в sys.path, чтобы импортировать pipeline
# если файл лежит в той же директории, что и DAG
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import train

default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'titanic_ml_pipeline',
    default_args=default_args,
    description='DAG для обучения модели предсказания выживаемости на Титанике',
   # schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['ml', 'titanic'],
)

train_task = PythonOperator(task_id='train_model', python_callable=train,dag=dag,)

train_task

