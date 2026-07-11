# /home/azureuser/credit-risk-ews/api/schemas.py
from pydantic import BaseModel, Field

class CreditRiskInput(BaseModel):
    person_age: float = Field(..., example=32.0)
    person_income: float = Field(..., example=55000.0)
    person_emp_length: float = Field(..., example=5.0)
    loan_amnt: float = Field(..., example=12000.0)
    loan_int_rate: float = Field(..., example=11.4)
    loan_percent_income: float = Field(..., example=0.22)
    cb_person_cred_hist_length: float = Field(..., example=3.0)
    loan_grade_encoded: float = Field(..., example=2.0)
    cb_person_default_on_file: float = Field(..., example=0.0)
    
    # One-hot encoded dummy configurations matching preprocess output features
    person_home_ownership_OTHER: int = Field(0, example=0)
    person_home_ownership_OWN: int = Field(0, example=0)
    person_home_ownership_RENT: int = Field(0, example=1)
    
    loan_intent_EDUCATION: int = Field(0, example=0)
    loan_intent_HOMEIMPROVEMENT: int = Field(0, example=0)
    loan_intent_MEDICAL: int = Field(0, example=1)
    loan_intent_PERSONAL: int = Field(0, example=0)
    loan_intent_VENTURE: int = Field(0, example=0)
    
    # Imputation flags matching preprocessing structure
    loan_int_rate_was_missing: int = Field(0, example=0)
    person_emp_length_was_missing: int = Field(0, example=0)

class PredictionResponse(BaseModel):
    prediction: str
    probability: float