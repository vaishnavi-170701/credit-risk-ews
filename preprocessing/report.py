# modules/report.py
import os
import time
import pandas as pd
from logger_config import get_pipeline_logger

logger = get_pipeline_logger("ReportingModule")

def compile_execution_summary(start_time):
    logger.info("Compiling Automated Pipeline Execution Summary")
    
    raw_df = pd.read_csv('data/raw/credit_risk_dataset.csv')
    processed_df = pd.read_csv('data/processed/credit_risk_processed.csv')
    
    exec_duration = round(time.time() - start_time, 2)
    raw_rows, raw_cols = raw_df.shape
    processed_rows, processed_cols = processed_df.shape
    
    missing_count = int(raw_df.isnull().sum().sum())
    duplicate_count = int(raw_df.duplicated().sum())
    
    charts_generated = len(os.listdir('reports/figures'))
    
    summary_content = f"""==================================================
DATAOPS PIPELINE OPERATIONAL RUN SUMMARY
==================================================
Execution Time   : {exec_duration} seconds
Raw Dataset Rows : {raw_rows}
Raw Columns      : {raw_cols}
Missing Values   : {missing_count}
Duplicates Found : {duplicate_count}
Processed Rows   : {processed_rows}
Processed Columns: {processed_cols}
Charts Generated : {charts_generated} (Saved in reports/figures/)
Output File      : data/processed/credit_risk_processed.csv
Pipeline Status  : SUCCESS
=================================================="""

    with open('reports/summary.txt', 'w') as f:
        f.write(summary_content)
        
    logger.info("Pipeline Completed Successfully! Summary saved to reports/summary.txt")
    print(summary_content)

if __name__ == "__main__":
    compile_execution_summary(time.time())