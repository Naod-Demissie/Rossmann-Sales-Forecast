from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime

# Initialize FastAPI app and logger
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Load the trained model
MODEL_PATH = "best_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load the model: %s", str(e))
    raise

# Initialize MinMaxScaler for preprocessing
scaler = MinMaxScaler(feature_range=(-1, 1))


# Define input schema using Pydantic
class InputData(BaseModel):
    Store: int
    DayOfWeek: int
    Date: str
    Sales: float
    Customers: int
    Open: int
    Promo: int
    StateHoliday: str
    SchoolHoliday: int


@app.post("/predict/")
async def predict(input_data: InputData):
    """
    Endpoint for predicting sales.
    Accepts input data as JSON with the required fields.
    """
    try:
        # Convert input data to DataFrame
        data = input_data.dict()
        input_df = pd.DataFrame([data])
        required_columns = [
            "Store",
            "DayOfWeek",
            "Date",
            "Sales",
            "Customers",
            "Open",
            "Promo",
            "StateHoliday",
            "SchoolHoliday",
        ]

        # Check if all required columns are present
        if not all(col in input_df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns. Required columns: {required_columns}",
            )

        # Preprocess data
        input_df["Date"] = pd.to_datetime(input_df["Date"])
        input_df["Year"] = input_df["Date"].dt.year
        input_df["Month"] = input_df["Date"].dt.month
        input_df["Day"] = input_df["Date"].dt.day
        input_df = input_df.drop(columns=["Date", "Sales"])  # Drop unnecessary columns

        # Handle categorical features
        input_df["StateHoliday"] = (
            input_df["StateHoliday"]
            .astype(str)
            .map({"0": 0, "a": 1, "b": 2, "c": 3})
            .fillna(0)
            .astype(int)
        )

        # Scale numerical features
        numerical_cols = [
            "Store",
            "DayOfWeek",
            "Customers",
            "Open",
            "Promo",
            "StateHoliday",
            "SchoolHoliday",
            "Year",
            "Month",
            "Day",
        ]
        scaled_data = scaler.fit_transform(input_df[numerical_cols])
        scaled_data = scaled_data.reshape(
            1, scaled_data.shape[0], scaled_data.shape[1]
        )  # Reshape for LSTM

        # Make predictions
        predictions = model.predict(scaled_data)
        predicted_sales = predictions[0][0]

        return {"predicted_sales": float(predicted_sales)}

    except Exception as e:
        logger.error("Error in prediction: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
