import sys
import os
import time
from datetime import datetime, timedelta

# ==============================================================================
# ABSOLUTE PATH INJECTION FOR YOUR PREPROCESSING FOLDER
# ==============================================================================
# This ensures that no matter where Airflow executes from, it can find your modules.
PROJECT_PREPROCESSING_PATH = '/home/azureuser/credit-risk-ews/preprocessing'

if PROJECT_PREPROCESSING_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PREPROCESSING_PATH)

# Add parent directory as a fallback
PARENT_DIR = '/home/azureuser/credit-risk-ews'
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# ==============================================================================
# SAFELY IMPORT YOUR MODULES FROM THE PREPROCESSING/ FOLDER
# ==============================================================================
from validation import download_and_validate
from preprocess import clean_and_preprocess
from eda import generate_eda_charts
from report import compile_execution_summary

# Import Airflow Structural Dependencies
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'DataOps_Group9',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    'intelligent_credit_risk_dataops_pipeline',
    default_args=default_args,
    description='Automated Risk Engineering System executing every 2 minutes',
    schedule_interval='*/2 * * * *',  # Exactly every 2 minutes as required by Cron phase
    catchup=False,
    max_active_runs=1
) as dag:

    def start_pipeline_callback(**context):
        context['ti'].xcom_push(key='start_time', value=time.time())

    def report_wrapper_task(**context):
        start_time = context['ti'].xcom_pull(key='start_time', task_ids='Start_Pipeline')
        if not start_time:
            start_time = time.time()
        compile_execution_summary(start_time)

    # Airflow Operational Task Definitions
    task_start = PythonOperator(
        task_id='Start_Pipeline',
        python_callable=start_pipeline_callback
    )

    task_validate = PythonOperator(
        task_id='Validate_Dataset',
        python_callable=download_and_validate
    )

    task_preprocess = PythonOperator(
        task_id='Preprocess_Data',
        python_callable=clean_and_preprocess
    )

    task_eda = PythonOperator(
        task_id='Generate_EDA',
        python_callable=generate_eda_charts
    )

    task_report = PythonOperator(
        task_id='Generate_Report',
        python_callable=report_wrapper_task
    )

    # PHASE 7 — DAG Flow Topology Linkage
    task_start >> task_validate >> task_preprocess >> task_eda >> task_report