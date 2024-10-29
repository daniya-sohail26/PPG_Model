
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model


# Define your FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
ann_model = load_model('ann_model.keras')
rf_model = joblib.load('rf_model.pkl')
gb_model = joblib.load('gb_model.pkl')

# Load the scaler
with open('minmax_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

class InputData(BaseModel):
    temperature_2_m_above_gnd: float
    relative_humidity_2_m_above_gnd: float
    mean_sea_level_pressure_MSL: float
    total_precipitation_sfc: float
    snowfall_amount_sfc: float
    total_cloud_cover_sfc: float
    high_cloud_cover_high_cld_lay: int
    medium_cloud_cover_mid_cld_lay: int
    low_cloud_cover_low_cld_lay: int
    shortwave_radiation_backwards_sfc: float
    wind_speed_10_m_above_gnd: float
    wind_direction_10_m_above_gnd: float
    wind_speed_80_m_above_gnd: float
    wind_direction_80_m_above_gnd: float
    wind_gust_10_m_above_gnd: float
    angle_of_incidence: float
    zenith: float
    azimuth: float

# Prediction function
def predict(input_data, model):
    input_df = pd.DataFrame([{
        "temperature_2_m_above_gnd": input_data.temperature_2_m_above_gnd,
        "relative_humidity_2_m_above_gnd": input_data.relative_humidity_2_m_above_gnd,
        "mean_sea_level_pressure_MSL": input_data.mean_sea_level_pressure_MSL,
        "total_precipitation_sfc": input_data.total_precipitation_sfc,
        "snowfall_amount_sfc": input_data.snowfall_amount_sfc,
        "total_cloud_cover_sfc": input_data.total_cloud_cover_sfc,
        "high_cloud_cover_high_cld_lay": input_data.high_cloud_cover_high_cld_lay,
        "medium_cloud_cover_mid_cld_lay": input_data.medium_cloud_cover_mid_cld_lay,
        "low_cloud_cover_low_cld_lay": input_data.low_cloud_cover_low_cld_lay,
        "shortwave_radiation_backwards_sfc": input_data.shortwave_radiation_backwards_sfc,
        "wind_speed_10_m_above_gnd": input_data.wind_speed_10_m_above_gnd,
        "wind_direction_10_m_above_gnd": input_data.wind_direction_10_m_above_gnd,
        "wind_speed_80_m_above_gnd": input_data.wind_speed_80_m_above_gnd,
        "wind_direction_80_m_above_gnd": input_data.wind_direction_80_m_above_gnd,
        "wind_gust_10_m_above_gnd": input_data.wind_gust_10_m_above_gnd,
        "angle_of_incidence": input_data.angle_of_incidence,
        "zenith": input_data.zenith,
        "azimuth": input_data.azimuth
    }])

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    return prediction[0]

# API endpoints
@app.post("/predict/rf")
async def predict_rf(input_data: InputData):
    try:
        prediction = predict(input_data, rf_model)
        return {"predicted_power": float(prediction)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/gb")
async def predict_gb(input_data: InputData):
    try:
        prediction = predict(input_data, gb_model)
        return {"predicted_power": float(prediction)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/ann")
async def predict_ann(input_data: InputData):
    try:
        prediction = predict(input_data, ann_model)
        return {"predicted_power": float(prediction)}
    except Exception as e:
        return {"error": str(e)}
    