# /home/azureuser/credit-risk-ews/ml/model_utils.py
import os
import pickle
from ml_logger import get_ml_logger

logger = get_ml_logger("ModelUtils")

def save_pickle_model(model, filename):
    models_dir = '/home/azureuser/credit-risk-ews/models'
    os.makedirs(models_dir, exist_ok=True)
    target_path = os.path.join(models_dir, filename)
    
    with open(target_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved locally as pickle at: {target_path}")
    return target_path