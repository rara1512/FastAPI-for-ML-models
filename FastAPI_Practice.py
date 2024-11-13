from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load("xgboost_regressor_model.pkl")

# Define the request model to accept a dictionary of features
class PredictionRequest(BaseModel):
    features: dict

# Define the response model
class PredictionResponse(BaseModel):
    prediction: float

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Convert features dictionary to an array that matches the model's expected feature order
    feature_names = model.get_booster().feature_names  # Get feature names from the model
    if feature_names is None:
        raise HTTPException(status_code=500, detail="The model does not have feature names.")

    # Create an array of feature values in the correct order
    try:
        feature_values = [request.features.get(name, 0) for name in feature_names]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")

    # Convert to numpy array
    features_array = np.array([feature_values])

    # Predict using the model
    prediction = model.predict(features_array)[0]
    
    return PredictionResponse(prediction=prediction)