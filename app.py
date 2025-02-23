import os
from flask_api import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel, ValidationError
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Table, MetaData, select, insert
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from models import *
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy.stats import chi2_contingency
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.calibration import calibration_curve
import joblib
from category_encoders import WOEEncoder

# Initialize FastAPI app
app = FastAPI()

# Database configuration
DATABASE_URL = app.config.get('DATABASE_URI')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()

# Define table
raw_data = Table('raw_data', metadata, autoload_with=engine)
processed_data = Table('processed_data', metadata, autoload_with=engine)
preprocessing_logs = Table('preprocessing_logs', metadata, autoload_with=engine)
eda_results = Table('eda_results', metadata, autoload_with=engine)
selected_features = Table('selected_features', metadata, autoload_with=engine)
feature_selection_logs = Table('feature_selection_logs', metadata, autoload_with=engine)
model_training_logs = Table('model_training_logs', metadata, autoload_with=engine)
model_validation_logs = Table('model_validation_logs', metadata, autoload_with=engine)
predictions = Table('predictions', metadata, autoload_with=engine)
prediction_logs = Table('prediction_logs', metadata, autoload_with=engine)


# Endpoint to ingest raw data
@app.post("/ingest")
async def ingest_data(file: UploadFile = None):
    try:
        # Default file path
        default_file_path = "/data/train.csv"

        # Read the file into a DataFrame
        if file:
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file.file)
            elif file.filename.endswith(".json"):
                df = pd.read_json(file.file)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a CSV or JSON file.")
        else:
            df = pd.read_csv(default_file_path)

        # Validate the data against the schema's metadata
        try:
            validated_data = [RawData(**item) for item in df.to_dict(orient="records")]
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=f"Validation error: {e}")

        # Insert data into the raw_data table
        db = SessionLocal()
        try:
            for data in validated_data:
                db.add(RawData(**data.dict()))
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            db.close()

        return {"message": "Data ingested successfully", "record_count": len(validated_data)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_data():
    # Fetch data from the raw_data table
    with SessionLocal() as session:
        query = select(raw_data)
        result = session.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=raw_data.columns.keys())

    # Log: Start preprocessing
    log_entry = {
        "step": "start",
        "details": "Preprocessing started",
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(preprocessing_logs).values(log_entry))
        session.commit()

    # Handle missing values: Impute Annual_Income with median
    median_income = df['Annual_Income'].median()
    df['Annual_Income'].fillna(median_income, inplace=True)
    log_entry = {
        "step": "imputation",
        "details": f"Imputed Annual_Income with median: {median_income}",
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(preprocessing_logs).values(log_entry))
        session.commit()

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    log_entry = {
        "step": "remove_duplicates",
        "details": "Removed duplicate rows",
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(preprocessing_logs).values(log_entry))
        session.commit()

    # Remove outliers (e.g., values outside 1.5 * IQR)
    for col in ['Annual_Income', 'Monthly_Inhand_Salary']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    log_entry = {
        "step": "remove_outliers",
        "details": "Removed outliers using IQR method",
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(preprocessing_logs).values(log_entry))
        session.commit()

    # Encode categorical variables (e.g., Credit_Mix, Payment_Behaviour)
    df = pd.get_dummies(df, columns=['Credit_Mix', 'Payment_Behaviour'], drop_first=True)
    log_entry = {
        "step": "encode_categorical",
        "details": "Encoded categorical variables using one-hot encoding",
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(preprocessing_logs).values(log_entry))
        session.commit()

    # Store the cleaned data in the processed_data table
    with SessionLocal() as session:
        df.to_sql('processed_data', con=engine, if_exists='replace', index=False)
        session.commit()

    # Log: Preprocessing completed
    log_entry = {
        "step": "complete",
        "details": "Preprocessing completed",
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(preprocessing_logs).values(log_entry))
        session.commit()

    # Return response
    return PreprocessResponse(
        message="Data preprocessing completed successfully",
        record_count=len(df)
    )

@app.get("/eda", response_model=EDAResponse)
async def perform_eda():
    # Fetch data from the raw_data table
    with SessionLocal() as session:
        query = select(processed_data)
        result = session.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=processed_data.columns.keys())

    # Calculate missing values
    missing_values = df.isnull().sum().to_dict()

    # Calculate correlations
    correlations = df.corr().to_dict()

    # Generate visualizations
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    heatmap_buffer = io.BytesIO()
    plt.savefig(heatmap_buffer, format='png')
    heatmap_buffer.seek(0)
    heatmap_base64 = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
    plt.close()

    # Store EDA summary statistics in the eda_results table
    with SessionLocal() as session:
        session.execute(
            eda_results.insert().values(
                missing_values=missing_values,
                correlations=correlations,
                visualizations=[heatmap_base64]
            )
        )
        session.commit()

    # Return EDA results
    return EDAResponse(
        missing_values=missing_values,
        correlations=correlations,
        visualizations=[heatmap_base64]
    )

@app.post("/select_features", response_model=FeatureSelectionResponse)
async def select_features():
    # Fetch data from the processed_data table
    with SessionLocal() as session:
        query = select(processed_data)
        result = session.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=processed_data.columns.keys())

    # Log: Start feature selection
    log_entry = {
        "step": "start",
        "details": "Feature selection started",
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(feature_selection_logs).values(log_entry))
        session.commit()

    # Correlation analysis for numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    correlation_scores = df[numerical_features].corrwith(df['Credit_Score']).abs().to_dict()

    # Chi-square test for categorical features
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    chi2_scores = {}
    for feature in categorical_features:
        contingency_table = pd.crosstab(df[feature], df['Credit_Score'])
        chi2, _, _, _ = chi2_contingency(contingency_table)
        chi2_scores[feature] = chi2

    # Combine scores and select top features
    feature_scores = {**correlation_scores, **chi2_scores}
    top_features = sorted(feature_scores, key=feature_scores.get, reverse=True)[:5]  # Select top 5 features

    # Log: Selected features
    log_entry = {
        "step": "select_features",
        "details": f"Selected features: {top_features}",
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(feature_selection_logs).values(log_entry))
        session.commit()

    # Store selected features in the selected_features table
    with SessionLocal() as session:
        for feature in top_features:
            session.execute(insert(selected_features).values({"feature_name": feature, "importance_score": feature_scores[feature]}))
        session.commit()

    # Log: Feature selection completed
    log_entry = {
        "step": "complete",
        "details": "Feature selection completed",
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(feature_selection_logs).values(log_entry))
        session.commit()

    # Return response
    return FeatureSelectionResponse(
        message="Feature selection completed successfully",
        selected_features=top_features
    )

@app.post("/train", response_model=ModelTrainingResponse)
async def train_model():
    # Fetch data from the processed_data table
    with SessionLocal() as session:
        query = select(processed_data)
        result = session.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=processed_data.columns.keys())

    # Log: Start model training
    log_entry = {
        "step": "start",
        "details": "Model training started",
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(model_training_logs).values(log_entry))
        session.commit()

    # Split data into training and testing sets
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply WOE transformation to categorical variables
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    woe_encoder = WOEEncoder(cols=categorical_features)
    X_train = woe_encoder.fit_transform(X_train, y_train)
    X_test = woe_encoder.transform(X_test)

    # Train a logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the trained model
    model_version = "1.0.0"
    model_dir = "/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"credit_risk_model_{model_version}.pkl")
    joblib.dump(model, model_path)

    # Log: Model training completed
    log_entry = {
        "step": "complete",
        "details": f"Model training completed with accuracy: {accuracy}",
        "model_version": model_version,
        "hyperparameters": str(model.get_params()),
        "accuracy": accuracy,
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(model_training_logs).values(log_entry))
        session.commit()

    # Return response
    return ModelTrainingResponse(
        message="Model training completed successfully",
        model_version=model_version,
        accuracy=accuracy
    )

@app.post("/validate", response_model=ModelValidationResponse)
async def validate_model():
    # Fetch data from the processed_data table
    with SessionLocal() as session:
        query = select(processed_data)
        result = session.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=processed_data.columns.keys())

    # Log: Start model validation
    log_entry = {
        "step": "start",
        "details": "Model validation started",
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(model_validation_logs).values(log_entry))
        session.commit()

    # Load the trained model
    model_version = "1.0.0"
    model_path = f"/models/credit_risk_model_{model_version}.pkl"
    model = joblib.load(model_path)

    # Prepare data for validation
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    woe_encoder = WOEEncoder(cols=categorical_features)
    X = woe_encoder.fit_transform(X, y)

    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Calculate metrics
    roc_auc = roc_auc_score(y, y_pred_proba)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')

    # Perform calibration
    prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10)
    calibration_details = {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist()
    }

    # Log: Validation metrics and calibration details
    log_entry = {
        "step": "validation",
        "details": {
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "calibration_details": calibration_details
        },
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(model_validation_logs).values(log_entry))
        session.commit()

    # Log: Model validation completed
    log_entry = {
        "step": "complete",
        "details": "Model validation completed",
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(model_validation_logs).values(log_entry))
        session.commit()

    # Return response
    return ModelValidationResponse(
        message="Model validation completed successfully",
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        calibration_details=calibration_details
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Load the trained model
    model_version = "1.0.0"
    model_path = f"/models/credit_risk_model_{model_version}.pkl"
    model = joblib.load(model_path)

    # Prepare input data
    customer_data = request.customer_data
    df = pd.DataFrame([customer_data])

    # Apply WOE transformation to categorical variables
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    woe_encoder = WOEEncoder(cols=categorical_features)
    df = woe_encoder.fit_transform(df)

    # Make prediction
    prediction = model.predict(df)[0]
    prediction_proba = model.predict_proba(df)[0]
    confidence = max(prediction_proba)

    # Store prediction in the predictions table
    with SessionLocal() as session:
        session.execute(
            insert(predictions).values(
                customer_data=customer_data,
                prediction=prediction,
                confidence=confidence,
                model_version=model_version
            )
        )
        session.commit()

    # Log prediction request
    log_entry = {
        "customer_data": customer_data,
        "prediction": prediction,
        "confidence": confidence,
        "model_version": model_version,
        "timestamp": datetime.now()
    }
    with SessionLocal() as session:
        session.execute(insert(prediction_logs).values(log_entry))
        session.commit()

    # Return response
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        model_version=model_version
    )


if __name__ == "__main__":
    app.run(debug=True)