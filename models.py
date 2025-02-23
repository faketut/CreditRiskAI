from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from datetime import datetime

# Define SQLAlchemy Base
Base = declarative_base()

# Define the raw_data table schema
class RawData(Base):
    __tablename__ = "raw_data"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True)
    month = Column(String)
    name = Column(String)
    age = Column(Integer)
    ssn = Column(String)
    occupation = Column(String)
    annual_income = Column(Float)
    monthly_inhand_salary = Column(Float)
    num_bank_accounts = Column(Integer)
    num_credit_card = Column(Integer)
    interest_rate = Column(Float)
    num_of_loan = Column(Integer)
    type_of_loan = Column(String)
    delay_from_due_date = Column(Integer)
    num_of_delayed_payment = Column(Integer)
    changed_credit_limit = Column(Float)
    num_credit_inquiries = Column(Integer)
    credit_mix = Column(String)
    outstanding_debt = Column(Float)
    credit_utilization_ratio = Column(Float)
    credit_history_age = Column(Integer)
    payment_of_min_amount = Column(String)
    total_emi_per_month = Column(Float)
    amount_invested_monthly = Column(Float)
    payment_behaviour = Column(String)
    monthly_balance = Column(Float)
    credit_score = Column(String)


# Pydantic model for data validation
class CustomerData(BaseModel):
    id: int
    customer_id: str
    month: str
    name: str
    age: int
    ssn: str
    occupation: str
    annual_income: float
    monthly_inhand_salary: float
    num_bank_accounts: int
    num_credit_card: int
    interest_rate: float
    num_of_loan: int
    type_of_loan: str
    delay_from_due_date: int
    num_of_delayed_payment: int
    changed_credit_limit: float
    num_credit_inquiries: int
    credit_mix: str
    outstanding_debt: float
    credit_utilization_ratio: float
    credit_history_age: int
    payment_of_min_amount: str
    total_emi_per_month: float
    amount_invested_monthly: float
    payment_behaviour: str
    monthly_balance: float
    credit_score: str

# Pydantic model for preprocessing response
class PreprocessResponse(BaseModel):
    message: str
    record_count: int

class EDAResponse(BaseModel):
    missing_values: dict
    correlations: dict
    visualizations: list

# Pydantic model for feature selection response
class FeatureSelectionResponse(BaseModel):
    message: str
    selected_features: list

# Pydantic model for training response
class ModelTrainingResponse(BaseModel):
    message: str
    model_version: str
    accuracy: float

# Pydantic model for validation response
class ModelValidationResponse(BaseModel):
    message: str
    roc_auc: float
    precision: float
    recall: float
    calibration_details: dict

# Pydantic model for prediction request
class PredictionRequest(BaseModel):
    customer_data: dict

# Pydantic model for prediction response
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_version: str