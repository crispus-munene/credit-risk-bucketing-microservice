import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score, precision_score, confusion_matrix, f1_score
import logging
import mlflow
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
logging.basicConfig(format=log_format, level=logging.INFO)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment('credit-risk-bucketing-experiment')

client = MlflowClient()

def cluster_model(train: pd.DataFrame, test: pd.DataFrame):
    train= train.copy()

    categorical_columns= train.select_dtypes(include='object').columns.tolist()
    numerical_columns= train.drop('charged_off', axis=1).select_dtypes(exclude='object').columns.tolist()

    train[categorical_columns]= train[categorical_columns].fillna('unknown')
    train[numerical_columns]= train[numerical_columns].fillna(train[numerical_columns].mean())

    X_train= train.drop('charged_off', axis=1)
    y_train= train['charged_off']

    numerical_data= X_train[numerical_columns].copy()
    scaler= StandardScaler()
    scaled_data= scaler.fit_transform(numerical_data)

    model= KMeans()
    visualizer= KElbowVisualizer(model, k=(2, 10), metric='calinski_harabasz', timings=False)
    visualizer.fit(scaled_data)
    best_k= visualizer.elbow_value_
    best_score= visualizer.k_scores_[best_k - 2]
    
    logging.info(f'Optimal number of clusters: {best_score}')
    logging.info(f'Clinski-Harabasz score at {best_k}: {best_score}')

    kmeans= KMeans(n_clusters=2, random_state=42)
    train_clusters= kmeans.fit_predict(scaled_data)

    numerical_data['cluster']= train_clusters

    X_train= X_train.copy()
    X_train['cluster']= numerical_data['cluster']

    categorical_cols= test.select_dtypes(include='object').columns.tolist()
    numerical_cols= test.drop('charged_off', axis=1).select_dtypes(exclude='object').columns.tolist()

    test[categorical_cols]= test[categorical_cols].fillna('unknown')
    test[numerical_cols]= test[numerical_cols].fillna(test[numerical_cols].mean())

    X_test= test.loc[:, test.columns != 'charged_off']
    y_test= test['charged_off']

    numerical_test= X_test[numerical_columns].copy()
    scaled_test= scaler.transform(numerical_test)

    X_test['cluster']= kmeans.predict(scaled_test)

    train= pd.concat([X_train, y_train], axis=1)
    test= pd.concat([X_test, y_test], axis=1)

    logging.info(f'Final clustered training data: {train.shape}')
    logging.info(f'Final clustered test data: {test.shape}')

    train.to_csv('data/processed/train.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)

    logging.info('Saved train and test dataset')

    with mlflow.start_run(run_name='kmeans-models'):
        mlflow.sklearn.log_model(kmeans, name='model')
        mlflow.log_param("n_clusters", 2)
        mlflow.set_tag("type", "clustering")

        run_id= mlflow.active_run().info.run_id
        model_uri= f'runs:/{run_id}/model'

        try:
            mlflow.register_model(model_uri, "kmeans-model")
        except Exception:
            pass

        latest = client.get_latest_versions("kmeans-model", stages=["None"])
        if latest:
            version = latest[0].version
            client.transition_model_version_stage(
                name="kmeans-model",
                version=version,
                stage="Production",
                archive_existing_versions=True
            )


def train_model(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)


    cluster_ids = [0, 1]
    model_names = ['model-cluster-0', 'model-cluster-1']



    for cluster_id, model_name in zip(cluster_ids, model_names):
        logging.info(f'Training model for cluster {cluster_id}')

        X_train = train.loc[train.cluster == cluster_id].drop(columns=['cluster', 'charged_off'], axis=1)
        y_train = train.loc[train.cluster == cluster_id]['charged_off']
        X_test = test.loc[test.cluster == cluster_id].drop(columns=['cluster', 'charged_off'], axis=1)
        y_test = test.loc[test.cluster == cluster_id]['charged_off']

        categorical_columns= X_train.select_dtypes(include='O').columns.tolist()
        numerical_columns= X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()

        preprocessor= ColumnTransformer(transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', TargetEncoder(cols=categorical_columns), categorical_columns)
        ])

        if cluster_id == 0:
            steps = [
                ('preprocess', preprocessor),
                ('sample', SMOTE(sampling_strategy=1.0, random_state=42)),
                ('model',  XGBClassifier(subsample=0.8, scale_pos_weight=1, reg_lambda=1.0, 
                            reg_alpha= 0.1, n_estimators= 600,min_child_weight=3, 
                            max_depth=7, learning_rate=0.1, gamma=0, colsample_bytree=1.0))
            ]
        else:
            steps = [
                ('preprocess', preprocessor),
                ('model', LGBMClassifier(subsample=0.8, scale_pos_weight=1, reg_lambda=1.0, 
                            reg_alpha= 0.1, num_leaves= 15, n_estimators= 600,min_child_samples=50, 
                            max_depth=5, learning_rate=0.1, colsample_bytree=1.0))
            ]

        pipeline = Pipeline(steps=steps)

        with mlflow.start_run(run_name=f'cluster-{cluster_id}-model'):
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)

            # Confusion matrix plot
            df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred),
                                 index=np.unique(y_test), columns=np.unique(y_test))
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'

            plt.figure(figsize=(5, 4))
            sns.heatmap(df_cm, annot=True, cmap='Blues')
            plot_path = f'src/visualization/confusion_matrix_cluster_{cluster_id}.png'
            plt.savefig(plot_path)
            plt.close()

            # Log metrics and artifacts
            mlflow.log_param('cluster-id', cluster_id)
            mlflow.log_param('model-type', 'LogisticRegression')
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('auc', auc)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('f1-score', f1)
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_artifact(plot_path)

            mlflow.sklearn.log_model(pipeline, artifact_path='model')

            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"

            try:
                client.get_registered_model(model_name)
                logging.info(f"Model '{model_name}' already registered.")
            except MlflowException:
                mlflow.register_model(model_uri=model_uri, name=model_name)
                logging.info(f"Registered new model '{model_name}'.")

            try:
                latest_versions = client.get_latest_versions(model_name, stages=['None'])
                if latest_versions:
                    version = latest_versions[0].version
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version,
                        stage='Production',
                        archive_existing_versions=True
                    )
                    logging.info(f"Promoted {model_name} version {version} to Production.")
            except Exception as e:
                logging.warning(f"Could not promote model {model_name}: {e}")
