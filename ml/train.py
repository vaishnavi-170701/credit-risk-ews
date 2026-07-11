# /home/azureuser/credit-risk-ews/ml/train.py
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Setup local imports
sys.path.insert(0, '/home/azureuser/credit-risk-ews/ml')
from ml_logger import get_ml_logger
from model_utils import save_pickle_model

logger = get_ml_logger("TrainingModule")

def train_models():
    logger.info("INFO — MLOps Step: Model Preparation and Training Initialized")
    
    processed_path = '/home/azureuser/credit-risk-ews/data/processed/credit_risk_processed.csv'
    if not os.path.exists(processed_path):
        logger.error("Processed data file missing for model training phase.")
        raise FileNotFoundError()
        
    df = pd.read_csv(processed_path)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    # Activity 2.2: 70% Train, 30% Test stratified partition split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    logger.info(f"Data partitioning successful. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Activity 2.1: Model Preparation (Logistic Regression & Random Forest)
    logger.info("Fitting Algorithm 1: Logistic Regression")
    lr_model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    lr_model.fit(X_train, y_train)
    save_pickle_model(lr_model, 'logistic_regression.pkl')
    
    logger.info("Fitting Algorithm 2: Random Forest Classifier")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    save_pickle_model(rf_model, 'random_forest.pkl')
    
    logger.info("Model Training Phase Complete.")
    return X_test, y_test, lr_model, rf_model

if __name__ == "__main__":
    train_models()

# Start the monitoring server dashboard inside your terminal session
# mlflow ui --host 0.0.0.0 --port 5000