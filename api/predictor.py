# /home/azureuser/credit-risk-ews/api/predictor.py
import os
import pickle
import pandas as pd
import logging

logger = logging.getLogger("APIPredictor")

class RiskPredictor:
    def __init__(self):
        self.model_path = '/home/azureuser/credit-risk-ews/models/best_model.pkl'
        self.model = None
        self.feature_order = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            logger.error(f"Champion binary missing at path: {self.model_path}")
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        # Extract the exact feature names alignment from the trained model object dynamically
        if hasattr(self.model, "feature_names_in_"):
            self.feature_order = list(self.model.feature_names_in_)
            logger.info(f"Model feature blueprint recovered: {self.feature_order}")
        else:
            # Absolute hardcoded fallback matrix layout matching your exact preprocessing output structure
            self.feature_order = [
                "person_age", "person_income", "person_emp_length", "loan_amnt", 
                "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", 
                "cb_person_default_on_file", "loan_grade_encoded",
                "person_home_ownership_OTHER", "person_home_ownership_OWN", "person_home_ownership_RENT",
                "loan_intent_EDUCATION", "loan_intent_HOMEIMPROVEMENT", "loan_intent_MEDICAL", 
                "loan_intent_PERSONAL", "loan_intent_VENTURE",
                "loan_int_rate_was_missing", "person_emp_length_was_missing"
            ]
        logger.info("Successfully loaded champion model into API memory.")

    def predict_risk(self, input_data: dict) -> tuple:
        # Convert dictionary payload instantly into standard Pandas row matrix
        df_input = pd.DataFrame([input_data])
        
        # CRITICAL FIX: Re-index the dataframe columns to match the model fit alignment exactly
        # If any feature is missing in the payload, it fills it with 0 to prevent crashes
        df_input = df_input.reindex(columns=self.feature_order, fill_value=0)
        
        # Get raw binary prediction class and target classification probabilities
        pred_class = int(self.model.predict(df_input)[0])
        probabilities = self.model.predict_proba(df_input)[0]
        target_prob = float(probabilities[pred_class])
        
        prediction_label = "High Risk" if pred_class == 1 else "Low Risk"
        return prediction_label, round(target_prob, 2)

# Instantiate single global instance to be reused across requests
predictor_instance = RiskPredictor()