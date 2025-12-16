import os 
import sys
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import uvicorn
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

from .pydantic_models import CustomerFeatures, PredictionResponse

mlflow.set_tracking_uri("http://localhost:5000")
MODEL_NAME = "Best_Credit_Risk_Model"
MODEL = None

app = FastAPI(title="Credit Risk Scorecard API")

def load_production_model():
    """Loads the model marked as 'Production' in the MLflow Model Registry."""
    global MODEL
    client = MlflowClient()
    
    model_versions = client.get_latest_versions(name=MODEL_NAME, stages=["Production"])
    
    if not model_versions:
        raise RuntimeError(f"No Production model found for name: {MODEL_NAME}")
    
    latest_version = model_versions[0]
    model_uri = f"models:/{MODEL_NAME}/Production"
    
    try:
        MODEL = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded successfully: {MODEL_NAME} V{latest_version.version}")
        return latest_version.version
    except Exception as e:
        print(f"Error loading model from {model_uri}: {e}")
        raise RuntimeError(f"Could not load production model: {e}")

@app.on_event("startup")
async def startup_event():
    """Loads the model when the FastAPI application starts."""
    load_production_model()

@app.get("/")
def read_root():
    return {"status": "ok", "model_name": MODEL_NAME}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(features: CustomerFeatures):
    """Accepts WoE-transformed features and returns the risk probability."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded or available.")

    try:
        feature_dict = features.dict()
        data_df = pd.DataFrame([feature_dict])
        
        expected_cols = list(data_df.columns) 
        
        y_pred_proba = MODEL.predict_proba(data_df[expected_cols])[:, 1]
        
        risk_prob = y_pred_proba[0]
        risk_class = "High Risk" if risk_prob >= 0.5 else "Low Risk"
        
        client = MlflowClient()
        latest_version = client.get_latest_versions(name=MODEL_NAME, stages=["Production"])[0]

        return PredictionResponse(
            risk_probability=round(float(risk_prob), 4),
            risk_class=risk_class,
            model_version=f"v{latest_version.version}"
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input features: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
