# /home/azureuser/credit-risk-ews/airflow/dags/mlops_training_pipeline.py
import sys
import os
import subprocess
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Inject 'ml' directory into system paths so execution shells find the workspace context cleanly
PROJECT_ML_PATH = '/home/azureuser/credit-risk-ews/ml'
if PROJECT_ML_PATH not in sys.path:
    sys.path.insert(0, PROJECT_ML_PATH)

default_args = {
    'owner': 'MLOps_Group9',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

# Define the manual MLOps orchestration DAG
with DAG(
    'mlops_model_training_and_evaluation',
    default_args=default_args,
    description='Manually triggered MLOps Pipeline for Model Training and MLflow Evaluation',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    max_active_runs=1
) as dag:

    def run_training_script():
        """Executes train.py securely using the virtual environment's Python binary."""
        python_bin = '/home/azureuser/credit-risk-ews/venv/bin/python3'
        script_path = '/home/azureuser/credit-risk-ews/ml/train.py'
        
        result = subprocess.run([python_bin, script_path], check=True, capture_output=True, text=True)
        print(result.stdout)

    def run_evaluation_script():
        """Executes evaluate.py securely using the virtual environment's Python binary."""
        python_bin = '/home/azureuser/credit-risk-ews/venv/bin/python3'
        script_path = '/home/azureuser/credit-risk-ews/ml/evaluate.py'
        
        result = subprocess.run([python_bin, script_path], check=True, capture_output=True, text=True)
        print(result.stdout)

    # Define Airflow Execution Tasks
    task_train = PythonOperator(
        task_id='Train_ML_Models',
        python_callable=run_training_script
    )

    # Corrected the variable name reference here just in case
    task_evaluate = PythonOperator(
        task_id='Evaluate_and_Log_MLflow',
        python_callable=run_evaluation_script
    )

    # Define dependency flow
    task_train >> task_evaluate