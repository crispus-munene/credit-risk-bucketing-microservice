from fastapi import FastAPI, HTTPException, Depends
from src.api.schemas import UserInput, PredictionResponse
import pandas as pd
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from fastapi_cache import FastAPICache
from src.api.utils import MlflowHandler
import logging

app = FastAPI(title="Credit Risk Bucketing API", version="1.0")

def get_handler():
    return MlflowHandler()

@app.on_event('startup')
async def on_startup():
    FastAPICache.init(InMemoryBackend(), prefix='credit-risk-bucketing-cache')

@app.get("/health")
@cache(expire=60)
def health(handler: MlflowHandler= Depends(get_handler)):
    result= handler.check_mlflow_health()
    return {'status', result}

@app.post("/predict", response_model=PredictionResponse)
@cache(expire=60)
def predict(input_data: UserInput, handler: MlflowHandler= Depends(get_handler)):
    kmeans_model= handler.load_kmeans_model()
    """Predict cluster and risk probability based on user input."""
    # Convert user input to DataFrame
    df = pd.DataFrame([input_data.model_dump()])

    # Step 1: Cluster assignment
    try:
        cluster = handler.assign_cluster(kmeans_model, df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clustering user: {e}")

    # Step 2: Load cluster model
    try:
        cluster_model = handler.load_cluster_model(cluster)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cluster model not found: {e}")

    # Step 3: Predict risk
    try:
        risk_label, probability = handler.predict_risk(cluster_model, df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        cluster=cluster,
        predicted_risk=risk_label,
        probability=probability
    )

@app.get("/")
def root():
    return {"message": "Credit Risk Clustering API"}
