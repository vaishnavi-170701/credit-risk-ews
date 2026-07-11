# /home/azureuser/credit-risk-ews/ml/evaluate.py
import os
import sys
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# Setup local imports
sys.path.insert(0, '/home/azureuser/credit-risk-ews/ml')
from ml_logger import get_ml_logger
from model_utils import save_pickle_model

logger = get_ml_logger("EvaluationModule")

def evaluate_and_log():
    logger.info("INFO - MLOps Step: Model Evaluation & Metric Logging Initialized")
    
    # 1. Re-generate identical 30% test split baseline mapping
    processed_path = '/home/azureuser/credit-risk-ews/data/processed/credit_risk_processed.csv'
    df = pd.read_csv(processed_path)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    
    models_dir = '/home/azureuser/credit-risk-ews/models'
    
    # Load serialized model objects
    with open(os.path.join(models_dir, 'logistic_regression.pkl'), 'rb') as f:
        lr_model = pickle.load(f)
    with open(os.path.join(models_dir, 'random_forest.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
        

    # ==============================================================================
    # PRODUCTION FIX FOR MLFLOW BACKEND
    # ==============================================================================
    # Connect tracking to a database file rather than maintenance-mode file stores
    mlflow.set_tracking_uri("sqlite:////home/azureuser/credit-risk-ews/mlflow.db")
    mlflow.set_experiment("Intelligent_Credit_Risk_Early_Warning_System")
    
    metrics_summary = {}
    
    # Loop evaluate models sequentially
    for name, model in [("Logistic_Regression", lr_model), ("Random_Forest", rf_model)]:
        with mlflow.start_run(run_name=f"{name}_Run"):
            y_pred = model.predict(X_test)
            
            # Compute Activity 2.4 requested performance logs
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            metrics_summary[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
            
            # MLOps Server Registry logs
            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"{name} Metrics -> Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # Determine best champion model architecture via highest F1 score metric
    best_name = "Random_Forest" if metrics_summary["Random_Forest"]["f1_score"] >= metrics_summary["Logistic_Regression"]["f1_score"] else "Logistic_Regression"
    best_model_obj = rf_model if best_name == "Random_Forest" else lr_model
    save_pickle_model(best_model_obj, 'best_model.pkl')
    logger.info(f"Champion candidate resolved: {best_name}")

    # Output metrics.json local summary file
    with open('/home/azureuser/credit-risk-ews/ml/metrics.json', 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    # Generate human readable text report summary artifact
    report_dir = '/home/azureuser/credit-risk-ews/reports'
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, 'model_report.txt'), 'w') as f:
        f.write("==================================================\n")
        f.write("MLOPS PIPELINE EVALUATION PERFORMANCE REPORT\n")
        f.write("==================================================\n")
        for m_name, metrics in metrics_summary.items():
            f.write(f"\nModel Configuration Name: {m_name}\n")
            for k, v in metrics.items():
                f.write(f" - {k.capitalize()}: {v:.4f}\n")
        f.write(f"\nCHAMPION SELECTION: {best_name} (Saved to models/best_model.pkl)\n")
        f.write("==================================================\n")

    logger.info("Evaluation Complete. Outputs written out cleanly.")

if __name__ == "__main__":
    evaluate_and_log()