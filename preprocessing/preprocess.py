# /home/azureuser/credit-risk-ews/preprocessing/preprocess.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # Explicitly required for MICE Imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from logger_config import get_pipeline_logger

logger = get_pipeline_logger("PreprocessingModule")

def clean_and_preprocess():
    logger.info("INFO — Preprocessing Stage Started")
    
    # Define absolute input and output paths to prevent Airflow directory shift bugs
    raw_path = '/home/azureuser/credit-risk-ews/data/raw/credit_risk_dataset.csv'
    processed_output_path = '/home/azureuser/credit-risk-ews/data/processed/credit_risk_processed.csv'
    
    if not os.path.exists(raw_path):
        logger.error(f"Raw data file path does not exist at {raw_path}")
        raise FileNotFoundError(f"Missing expected raw Kaggle file at: {raw_path}")
        
    # 1. Load Dataset
    df = pd.read_csv(raw_path)
    logger.info(f"Dataset Loaded for Preprocessing. Initial Shape: {df.shape}")
    
    # 2. Duplicate Row Removal
    initial_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Duplicate rows removed. Rows dropped: {initial_len - len(df)}")
    
    # 3. Anomaly Filtering & Business Rule Mitigation
    # Age values over 100 are anomalies; employment duration cannot exceed age minus working age boundary (16)
    df = df[df['person_age'] < 100].reset_index(drop=True)
    df = df[df['person_emp_length'] <= (df['person_age'] - 16)].reset_index(drop=True)
    logger.info("Structural cleaning & business logic anomaly filtering completed")

    # 4. Categorical Encoding (Ordinal & Nominal Mapping)
    # Ordinal Mapping for Loan Grade
    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    if 'loan_grade' in df.columns:
        df['loan_grade_encoded'] = df['loan_grade'].map(grade_mapping)
        df.drop(columns=['loan_grade'], inplace=True, errors='ignore')
    
    # Binary mapping for prior defaults
    if 'cb_person_default_on_file' in df.columns:
        df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
        
    # One-Hot Encoding for Nominal variables (Home Ownership & Loan Intent)
    nominal_cols = ['person_home_ownership', 'loan_intent']
    existing_nominal = [col for col in nominal_cols if col in df.columns]
    if existing_nominal:
        df = pd.get_dummies(df, columns=existing_nominal, drop_first=True, dtype=int)
        
    logger.info("Categorical encoding and hot-vector transformation complete")

    # 5. Data Leakage Prevention Boundary: Train-Test Stratified Split
    if 'loan_status' not in df.columns:
        logger.error("Target column 'loan_status' missing from the dataset.")
        raise KeyError("Target feature column 'loan_status' is required.")
        
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    # Perform stratified train-test split before imputation or scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Data Split Complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 6. MICE Imputation for Missing Numerical Variables
    num_scale_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    num_scale_features = [col for col in num_scale_features if col in X_train.columns]
    
    # Add missing indicator flags for features containing nulls to preserve structural variance information
    for col in ['loan_int_rate', 'person_emp_length']:
        if col in X_train.columns:
            X_train[f'{col}_was_missing'] = X_train[col].isnull().astype(int)
            X_test[f'{col}_was_missing'] = X_test[col].isnull().astype(int)
            
    # Apply MICE Imputation fitting strictly on training split data matrices
    logger.info("Executing MICE Imputation algorithm...")
    mice_imputer = IterativeImputer(random_state=42, max_iter=10)
    X_train[num_scale_features] = mice_imputer.fit_transform(X_train[num_scale_features])
    X_test[num_scale_features] = mice_imputer.transform(X_test[num_scale_features])
    logger.info("Imputation Successful")

    # 7. Normalization via RobustScaler
    # RobustScaler scales features using statistics that are robust to outliers (IQR)
    logger.info("Executing Robust Range Scaling Normalization...")
    scaler = RobustScaler()
    X_train[num_scale_features] = scaler.fit_transform(X_train[num_scale_features])
    X_test[num_scale_features] = scaler.transform(X_test[num_scale_features])
    logger.info("Scaling Completed")

    # 8. Re-combine Train Partition Matrix and Save Non-Empty File Output
    processed_df = pd.concat([X_train, y_train], axis=1)
    
    # Enforce directory existence path rules
    os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)
    
    # CRITICAL FIX: Write the actual populated dataframe down to the disk storage path
    processed_df.to_csv(processed_output_path, index=False)
    
    # Verify that the file was created successfully and contains columns/data rows
    if os.path.exists(processed_output_path) and os.path.getsize(processed_output_path) > 0:
        logger.info(f"Pipeline Step Successful: Processed dataset saved to {processed_output_path} (Size: {os.path.getsize(processed_output_path)} bytes)")
    else:
        logger.error(f"Critical System Failure: Empty file written out to {processed_output_path}")
        raise ValueError("Generated file contains 0 bytes data payloads.")
        
    return processed_output_path

if __name__ == "__main__":
    clean_and_preprocess()