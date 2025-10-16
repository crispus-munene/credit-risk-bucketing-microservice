# src/api/main.py

from fastapi import FastAPI, HTTPException
from src.api.schemas import UserInput, PredictionResponse
from src.api.utils import load_kmeans_model, load_cluster_model, assign_cluster, predict_risk
import pandas as pd
import logging

app = FastAPI(title="Credit Risk Bucketing API", version="1.0")

# Load MLflow models once at startup
try:
    kmeans_model = load_kmeans_model()
    logging.info("✅ KMeans model loaded from MLflow.")
except Exception as e:
    logging.error(f"Failed to load KMeans model: {e}")
    kmeans_model = None


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: UserInput):
    """Predict cluster and risk probability based on user input."""
    if kmeans_model is None:
        raise HTTPException(status_code=500, detail="KMeans model not available")

    # Convert user input to DataFrame
    df = pd.DataFrame([input_data.dict()])

    # Step 1: Cluster assignment
    try:
        cluster = assign_cluster(kmeans_model, df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clustering user: {e}")

    # Step 2: Load cluster model
    try:
        cluster_model = load_cluster_model(cluster)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cluster model not found: {e}")

    # Step 3: Predict risk
    try:
        risk_label, probability = predict_risk(cluster_model, df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        cluster=cluster,
        predicted_risk=risk_label,
        probability=probability
    )


@app.get("/")
def root():
    return {"message": "Welcome to the Credit Risk Bucketing API!"}
