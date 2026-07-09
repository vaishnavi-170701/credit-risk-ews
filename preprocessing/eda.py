# /home/azureuser/credit-risk-ews/preprocessing/eda.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from logger_config import get_pipeline_logger

logger = get_pipeline_logger("EDAModule")

def generate_eda_charts():
    logger.info("EDA Visualization Generation Started")
    
    # FIX: Use Absolute Paths instead of relative paths
    raw_df = pd.read_csv('/home/azureuser/credit-risk-ews/data/raw/credit_risk_dataset.csv')
    processed_df = pd.read_csv('/home/azureuser/credit-risk-ews/data/processed/credit_risk_processed.csv')
    
    # FIX: Use Absolute Path for output figure storage directory
    fig_dir = '/home/azureuser/credit-risk-ews/reports/figures'
    os.makedirs(fig_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")

    # Heatmap
    plt.figure(figsize=(10, 8))
    numerical_df = raw_df.select_dtypes(include=['number'])
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'heatmap.png'))
    plt.close()

    # Income Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(raw_df[raw_df['person_income'] < raw_df['person_income'].quantile(0.95)]['person_income'], bins=30, kde=True, color='skyblue')
    plt.title('Income Distribution')
    plt.savefig(os.path.join(fig_dir, 'income_distribution.png'))
    plt.close()

    # Age Distribution
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=raw_df['person_age'], color='salmon')
    plt.title('Age Profile Distribution')
    plt.savefig(os.path.join(fig_dir, 'age_distribution.png'))
    plt.close()

    # Loan Distribution Countplot
    plt.figure(figsize=(8, 5))
    sns.countplot(data=raw_df, x='person_home_ownership', hue='loan_status', palette='viridis')
    plt.title('Loan Status across Home Ownership')
    plt.savefig(os.path.join(fig_dir, 'loan_distribution.png'))
    plt.close()

    # Model Feature Importance Map
    X_train = processed_df.drop(columns=['loan_status'])
    y_train = processed_df['loan_status']
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feat_importances.values, y=feat_importances.index, palette="viridis")
    plt.title("Feature Importance Map")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_importance.png'))
    plt.close()
    
    logger.info("EDA Completed. Charts Saved.")