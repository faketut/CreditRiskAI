from flask_sqlalchemy import SQLAlchemy
from pydantic import BaseModel

db = SQLAlchemy()

# Define the raw_data table schema
class RawData(db.Model):
    __tablename__ = "raw_data"
    id = db.Column(db.Integer, primary_key=True, index=True)
    customer_id = db.Column(db.String, index=True)
    month = db.Column(db.String)
    name = db.Column(db.String)
    age = db.Column(db.Integer)
    ssn = db.Column(db.String)
    occupation = db.Column(db.String)
    annual_income = db.Column(db.Float)
    monthly_inhand_salary = db.Column(db.Float)
    num_bank_accounts = db.Column(db.Integer)
    num_credit_card = db.Column(db.Integer)
    interest_rate = db.Column(db.Float)
    num_of_loan = db.Column(db.Integer)
    type_of_loan = db.Column(db.String)
    delay_from_due_date = db.Column(db.Integer)
    num_of_delayed_payment = db.Column(db.Integer)
    changed_credit_limit = db.Column(db.Float)
    num_credit_inquiries = db.Column(db.Integer)
    credit_mix = db.Column(db.String)
    outstanding_debt = db.Column(db.Float)
    credit_utilization_ratio = db.Column(db.Float)
    credit_history_age = db.Column(db.String)
    payment_of_min_amount = db.Column(db.String)
    total_emi_per_month = db.Column(db.Float)
    amount_invested_monthly = db.Column(db.Float)
    payment_behaviour = db.Column(db.String)
    monthly_balance = db.Column(db.Float)
    credit_score = db.Column(db.String)

# Define other models similarly...

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

# Define the preprocess_response table schema
class PreprocessResponse(db.Model):
    __tablename__ = "preprocess_response"
    id = db.Column(db.Integer, primary_key=True, index=True)
    message = db.Column(db.String)
    record_count = db.Column(db.Integer)

# Define the eda_response table schema
class EDAResponse(db.Model):
    __tablename__ = "eda_response"
    id = db.Column(db.Integer, primary_key=True, index=True)
    missing_values = db.Column(db.JSON)
    correlations = db.Column(db.JSON)
    visualizations = db.Column(db.JSON)

# Define the feature_selection_response table schema
class FeatureSelectionResponse(db.Model):
    __tablename__ = "feature_selection_response"
    id = db.Column(db.Integer, primary_key=True, index=True)
    message = db.Column(db.String)
    selected_features = db.Column(db.JSON)

# Define the model_training_response table schema
class ModelTrainingResponse(db.Model):
    __tablename__ = "model_training_response"
    id = db.Column(db.Integer, primary_key=True, index=True)
    message = db.Column(db.String)
    model_version = db.Column(db.String)
    accuracy = db.Column(db.Float)

# Define the model_validation_response table schema
class ModelValidationResponse(db.Model):
    __tablename__ = "model_validation_response"
    id = db.Column(db.Integer, primary_key=True, index=True)
    message = db.Column(db.String)
    roc_auc = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    calibration_details = db.Column(db.JSON)

# Define the prediction_request table schema
class PredictionRequest(db.Model):
    __tablename__ = "prediction_request"
    id = db.Column(db.Integer, primary_key=True, index=True)
    customer_data = db.Column(db.JSON)

# Define the prediction_response table schema
class PredictionResponse(db.Model):
    __tablename__ = "prediction_response"
    id = db.Column(db.Integer, primary_key=True, index=True)
    prediction = db.Column(db.String)
    confidence = db.Column(db.Float)
    model_version = db.Column(db.String)