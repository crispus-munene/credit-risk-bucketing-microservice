# src/api/utils.py
from mlflow.client import MlflowClient
import mlflow
import mlflow.pyfunc
import pandas as pd
import pprint
from sklearn.preprocessing import StandardScaler

class MlflowHandler:
    def __init__(self):
        tracking_uri= "sqlite:///mlflow.db"
        self.client= MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    def check_mlflow_health(self):
        try:
            experiments= self.client.search_experiments()
            for exp in experiments:
                pprint.pprint(dict(exp), indent=4)
                return 'Service is running and returning experiments'
        except Exception as e:
            return f'Error calling mlflow: {e}'

    # --- Load Models ---
    def load_kmeans_model(self):
        """Load KMeans clustering model from MLflow Production stage."""
        return mlflow.pyfunc.load_model("models:/kmeans-model/Production")

    def load_cluster_model(self, cluster_id: int):
        """Load appropriate LogisticRegression model for cluster."""
        model_name = f"model-cluster-{cluster_id}"
        return mlflow.sklearn.load_model(f"models:/{model_name}/Production")


    def assign_cluster(self, kmeans_model, input_data: pd.DataFrame):
        """Assign the cluster for the given input using the same scaler used in training."""
        numeric_columns= input_data.select_dtypes(exclude='O').columns.tolist()

        scaler = StandardScaler()
        scaled = scaler.fit_transform(input_data[numeric_columns])
        cluster = kmeans_model.predict(scaled)[0]
        return cluster


    def predict_risk(self, cluster_model, input_data: pd.DataFrame):
        """Predict the probability and classify the risk."""
        proba = cluster_model.predict_proba(input_data)[:, 1][0]
        risk_label = "High" if proba >= 0.5 else "Low"
        return risk_label, float(proba)
