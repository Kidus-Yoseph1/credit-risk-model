import os 
import sys
import pandas as pd
import numpy as np
import mlflow
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

from data_preprocess import load_and_clean_data, create_full_data_matrix

# Configure MLflow to log runs locally
mlflow.set_tracking_uri("http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "Credit_Risk_Scorecard_Model"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def evaluate_model(y_true, y_pred, y_pred_proba):
    """Calculates and logs standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }
    return metrics

def run_mlflow_experiment(model_class, model_params, param_grid, X_train, y_train, X_test, y_test, model_name):
    """
    Performs Grid Search for hyperparameter tuning and logs the best model to MLflow.
    """
    with mlflow.start_run(run_name=f"{model_name}_GridSearch") as run:
        
        print(f"\n--- Running Hyperparameter Tuning for {model_name} ---")
        
        # Log parameters before tuning
        mlflow.log_param("model_type", model_name)
        mlflow.log_params(model_params)
        
        # 1. Hyperparameter Tuning (Grid Search)
        grid_search = GridSearchCV(
            estimator=model_class(**model_params),
            param_grid=param_grid,
            cv=3, 
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # 2. Prediction and Evaluation
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)

        # 3. MLflow Logging
        mlflow.log_param("tuning_method", "GridSearch")
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Test AUC: {metrics['roc_auc']:.4f}")

        # 4. Log Model Artifact
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        return best_model, metrics['roc_auc']

def train_models_and_select_best(X_woe, y_target):
    """Orchestrates model training and selects the best model based on AUC."""
    
    # Data Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_woe, y_target, test_size=0.3, random_state=42, stratify=y_target
    )
    print(f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    # Define Models and Hyperparameters
    models_to_run = {
        "LogisticRegression": {
            "class": LogisticRegression,
            "params": {'solver': 'liblinear', 'random_state': 42},
            "grid": {'C': [0.01, 0.1, 1.0, 10.0], 'penalty': ['l1', 'l2']}
        },
        "RandomForest": {
            "class": RandomForestClassifier,
            "params": {'random_state': 42},
            "grid": {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        }
    }
    
    best_model_name = None
    best_auc = -1.0
    
    # Run Experiments
    for name, config in models_to_run.items():
        model, auc_score = run_mlflow_experiment(
            model_class=config["class"],
            model_params=config["params"],
            param_grid=config["grid"],
            X_train=X_train, y_train=y_train, 
            X_test=X_test, y_test=y_test,
            model_name=name
        )
        
        if auc_score > best_auc:
            best_auc = auc_score
            best_model_name = name
    
    print(f"\n--- Best Model Selected: {best_model_name} (AUC: {best_auc:.4f}) ---")
    
    # Register the Best Model as 'Production'
    if best_model_name:
        # Get the latest run ID for the best model name
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME).experiment_id],
            filter_string=f"tags.mlflow.runName = '{best_model_name}_GridSearch'"
        )
        if runs:
            latest_run = runs[0]
            model_uri = f"runs:/{latest_run.info.run_id}/model"
            
            # Register the model and transition the latest version to Production stage
            model_version = mlflow.register_model(model_uri=model_uri, name="Best_Credit_Risk_Model")
            client.transition_model_version_stage(
                name="Best_Credit_Risk_Model",
                version=model_version.version,
                stage="Production"
            )
            print(f"Registered version {model_version.version} of 'Best_Credit_Risk_Model' and set to Production.")


if __name__ == '__main__':
    data_file_path = 'Data/data.csv' 
    print("--- Starting Task 5: Model Training and MLflow Tracking ---")
    
    try:
        # Load and preprocess data
        raw_data = load_and_clean_data(data_file_path)
        X_woe, y_target, _ = create_full_data_matrix(raw_data)
        
        # Run training and tracking pipeline
        train_models_and_select_best(X_woe, y_target)
        
    except Exception as e:
        print(f"\nFATAL ERROR during model training and tracking: {e}")
