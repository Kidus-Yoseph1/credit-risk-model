from pydantic import BaseModel
from typing import List

class CustomerFeatures(BaseModel):
    transaction_count_WOE: float
    total_amount_WOE: float
    average_amount_WOE: float
    std_amount_WOE: float
    avg_transaction_hour_WOE: float
    mode_transaction_month_WOE: float

    class Config:
        schema_extra = {
            "example": {
                "transaction_count_WOE": 0.52,
                "total_amount_WOE": -0.15,
                "average_amount_WOE": 0.33,
                "std_amount_WOE": -0.05,
                "avg_transaction_hour_WOE": 0.10,
                "mode_transaction_month_WOE": 0.08,
            }
        }

class PredictionResponse(BaseModel):
    risk_probability: float
    risk_class: str
    model_version: str
