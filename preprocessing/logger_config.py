# /home/azureuser/credit-risk-ews/preprocessing/logger.py
import os
import logging

def get_pipeline_logger(name):
    # Enforce absolute path to directory root logs folder
    log_dir = '/home/azureuser/credit-risk-ews/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'pipeline.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger