# src/api/utils.py

import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.preprocessing import RobustScaler

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# --- Load Models ---
def load_kmeans_model():
    """Load KMeans clustering model from MLflow Production stage."""
    return mlflow.pyfunc.load_model("models:/kmeans-model/Production")

def load_cluster_model(cluster_id: int):
    """Load appropriate LogisticRegression model for cluster."""
    model_name = f"model-cluster-{cluster_id}"
    return mlflow.sklearn.load_model(f"models:/{model_name}/Production")


def assign_cluster(kmeans_model, input_data: pd.DataFrame):
    """Assign the cluster for the given input using the same scaler used in training."""
    numeric_columns = ['loan_amnt','funded_amnt','term','int_rate','installment','dti',
        'open_acc','revol_util','last_pymnt_amnt','acc_open_past_24mths','avg_cur_bal','bc_open_to_buy',
        'bc_util','mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op','mort_acc','num_actv_rev_tl',
        'log_annual_inc','fico_score', 'credit_age_months','credit_age_years']

    scaler = RobustScaler()
    scaled = scaler.fit_transform(input_data[numeric_columns])
    cluster = kmeans_model.predict(scaled)[0]
    return cluster


def predict_risk(cluster_model, input_data: pd.DataFrame):
    """Predict the probability and classify the risk."""
    proba = cluster_model.predict_proba(input_data)[:, 1][0]
    risk_label = "High" if proba >= 0.5 else "Low"
    return risk_label, float(proba)
