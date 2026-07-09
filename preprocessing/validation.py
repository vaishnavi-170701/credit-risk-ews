# modules/validation.py
import os
import pandas as pd
import kaggle
from logger_config import get_pipeline_logger

logger = get_pipeline_logger("ValidationModule")

def download_and_validate():
    logger.info("Pipeline Started: Data Ingestion & Validation Stage")
    
    # 1. Download directly using Kaggle API
    dataset_id = 'laotse/credit-risk-dataset'
    raw_dir = 'data/raw'
    os.makedirs(raw_dir, exist_ok=True)
    
    logger.info(f"Downloading dataset from Kaggle: {dataset_id}")
    kaggle.api.dataset_download_files(dataset_id, path=raw_dir, unzip=True)
    
    csv_path = os.path.join(raw_dir, 'credit_risk_dataset.csv')
    if not os.path.exists(csv_path):
        logger.error(f"Dataset CSV file not found at {csv_path}")
        raise FileNotFoundError(f"Missing expected raw file.")
        
    # 2. Read Dataset
    df = pd.read_csv(csv_path)
    logger.info("Dataset Loaded Successfully")
    
    # 3. Gather Metrics
    rows, cols = df.shape
    columns_list = list(df.columns)
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    data_types = df.dtypes.to_dict()
    
    logger.info(f"Rows = {rows}")
    logger.info(f"Columns = {cols}")
    logger.info(f"Missing Values = {missing_values}")
    logger.info(f"Duplicate Rows = {duplicate_rows}")
    
    # 4. Generate Text Report Output
    report_path = 'reports/validation_report.txt'
    os.makedirs('reports', exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("=== DATA VALIDATION REPORT ===\n")
        f.write(f"Dataset Shape: {rows} rows, {cols} columns\n")
        f.write(f"Duplicate Rows: {duplicate_rows}\n")
        f.write(f"Total Missing Elements: {missing_values}\n\n")
        f.write("Column Names & Types:\n")
        for k, v in data_types.items():
            f.write(f"- {k}: {v}\n")
            
    logger.info(f"Validation Report Complete! Saved to {report_path}")
    return csv_path

if __name__ == "__main__":
    download_and_validate()