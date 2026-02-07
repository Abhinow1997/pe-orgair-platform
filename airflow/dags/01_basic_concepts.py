"""
This DAG demonstrates basic Airflow concepts:
1. Different types of operators
2. Task dependencies
3. XCom for task communication
4. Using environment variables
"""
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta
import os
import json

# Default arguments (good practice)
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'basic_concepts',
    default_args=default_args,
    description='Basic Airflow Concepts DAG',
    schedule='@daily',
    catchup=False
)

# 1. Basic Python Task
def print_context(**context):
    """Shows how to use context variables in Airflow"""
    print(f"Execution date is {context['ds']}")
    print(f"Task instance: {context['task_instance']}")
    return "Hello from first task!"

task1 = PythonOperator(
    task_id='print_context',
    python_callable=print_context,
    dag=dag
)

# 2. Task demonstrating XCom
def return_value(**context):
    """Shows how to push data to XCom"""
    data = {
        'value': 123,
        'date': context['ds']
    }
    # Push to XCom
    context['task_instance'].xcom_push(key='sample_data', value=data)
    return "Data pushed to XCom"

task2 = PythonOperator(
    task_id='push_to_xcom',
    python_callable=return_value,
    dag=dag
)

# 3. Task using XCom value
def use_xcom_value(**context):
    """Shows how to pull data from XCom"""
    # Pull from XCom
    data = context['task_instance'].xcom_pull(
        task_ids='push_to_xcom',
        key='sample_data'
    )
    print(f"Retrieved data: {data}")
    return f"Successfully used XCom value: {data['value']}"

task3 = PythonOperator(
    task_id='use_xcom_value',
    python_callable=use_xcom_value,
    dag=dag
)

# 4. Bash task showing environment variables
bash_task = BashOperator(
    task_id='bash_env_task',
    bash_command='echo "Running on: $HOSTNAME, date: {{ ds }}"',
    env={'MY_VAR': 'my_value'},  # Set environment variable
    dag=dag
)

# Set task dependencies - showing different patterns
task1 >> task2 >> task3  # Linear dependency
task1 >> bash_task  # Another path
# This creates:
#   task1 >> task2 >> task3
#      |
#      └->> bash_task
