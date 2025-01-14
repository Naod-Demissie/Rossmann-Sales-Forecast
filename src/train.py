import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, acf, pacf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)
import os
import pickle


# Configure the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LSTMTrainer:
    def __init__(self, preprocessed_data_path):
        try:
            self.data = pd.read_csv(preprocessed_data_path, parse_dates=["Date"])
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.model = None
            logger.info("Data loaded successfully from %s", preprocessed_data_path)
        except Exception as e:
            logger.error("Error loading data: %s", str(e))
            raise

    def check_stationarity(self, store_id, column="Sales"):
        """Check if the time series data is stationary for a specific store."""
        try:
            store_data = self.data[self.data["Store"] == store_id]
            if store_data.empty:
                logger.warning("No data found for Store ID: %s", store_id)
                return

            result = adfuller(store_data[column].dropna())
            logger.info("ADF Statistic for Store %s: %f", store_id, result[0])
            logger.info("p-value: %f", result[1])

            if result[1] <= 0.05:
                logger.info("The %s data for Store %s is stationary.", column, store_id)
            else:
                logger.info(
                    "The %s data for Store %s is not stationary.", column, store_id
                )
        except Exception as e:
            logger.error("Error in check_stationarity: %s", str(e))

    def plot_acf_pacf(self, store_id, column="Sales"):
        """Plot ACF and PACF for a specific store."""
        try:
            store_data = self.data[self.data["Store"] == store_id]
            if store_data.empty:
                logger.warning("No data found for Store ID: %s", store_id)
                return

            series = store_data[column].dropna()
            lag_acf = acf(series, nlags=20)
            lag_pacf = pacf(series, nlags=20, method="ols")

            plt.figure(figsize=(12, 6))

            plt.subplot(121)
            plt.stem(lag_acf, use_line_collection=True)
            plt.title(f"Autocorrelation for Store {store_id}")

            plt.subplot(122)
            plt.stem(lag_pacf, use_line_collection=True)
            plt.title(f"Partial Autocorrelation for Store {store_id}")

            plt.tight_layout()
            plt.show()
            logger.info("ACF and PACF plots generated for Store ID: %s", store_id)
        except Exception as e:
            logger.error("Error in plot_acf_pacf: %s", str(e))

    def load_train_val_data(self, X_train_path, X_val_path, y_train_path, y_val_path):
        """Load pre-split training and validation data from pickle files."""
        try:
            with open(X_train_path, "rb") as f:
                self.X_train = pickle.load(f)
            with open(X_val_path, "rb") as f:
                self.X_val = pickle.load(f)
            with open(y_train_path, "rb") as f:
                self.y_train = pickle.load(f)
            with open(y_val_path, "rb") as f:
                self.y_val = pickle.load(f)

            logger.info("Train and validation data loaded successfully.")
            print(f"{self.X_train.shape=}")
            print(f"{self.X_val.shape=}")
            print(f"{self.y_train.shape=}")
            print(f"{self.y_val.shape=}")

        except Exception as e:
            logger.error("Error loading train/validation data: %s", str(e))

    def build_model(self):
        """Build the LSTM model."""
        try:
            # Reshape input data to 3D
            self.X_train = self.X_train.toarray().reshape(
                self.X_train.shape[0], self.X_train.shape[1], 1
            )
            self.X_val = self.X_val.toarray().reshape(
                self.X_val.shape[0], self.X_val.shape[1], 1
            )

            # Build the LSTM model
            self.model = Sequential(
                [
                    LSTM(
                        50,
                        activation="relu",
                        input_shape=(self.X_train.shape[1], 1),
                    ),
                    Dense(1),
                ]
            )

            optimizer = Adam(learning_rate=0.00001)
            self.model.compile(optimizer=optimizer, loss="mean_squared_error")

            logger.info("LSTM model built successfully.")
        except Exception as e:
            logger.error("Error building the model: %s", str(e))

    def train_model(self, epochs=20, batch_size=32):
        """Train the LSTM model."""
        try:
            if not self.model:
                logger.warning("Model not built. Please build the model first.")
                return

            # Create callbacks
            checkpoint_path = "./checkpoints/best_model"
            checkpoint = ModelCheckpoint(
                checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
            )
            early_stop = EarlyStopping(
                monitor="val_loss", patience=10, verbose=1, restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss", patience=2, factor=0.5, verbose=1
            )
            tensorboard = TensorBoard(log_dir="./logs/", histogram_freq=1)

            callbacks = [checkpoint, early_stop, reduce_lr, tensorboard]

            self.model.fit(
                self.X_train,
                self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.X_val, self.y_val),
                callbacks=callbacks,
                verbose=1,
            )
            logger.info("Model training completed successfully.")
        except Exception as e:
            logger.error("Error during model training: %s", str(e))

    def evaluate_model(self):
        """Evaluate the LSTM model on validation data."""
        try:
            if not self.model:
                logger.warning("Model not built. Please build the model first.")
                return

            predictions = self.model.predict(self.X_val)
            plt.figure(figsize=(10, 6))
            plt.plot(self.y_val, label="Actual")
            plt.plot(predictions, label="Predicted")
            plt.legend()
            plt.title("Actual vs Predicted Sales (Validation Set)")
            plt.show()
            logger.info(
                "Model evaluation completed. Actual vs Predicted plot generated."
            )
        except Exception as e:
            logger.error("Error during model evaluation: %s", str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM model for Rossmann Sales Prediction."
    )
    parser.add_argument(
        "--preprocessed_data",
        type=str,
        required=True,
        help="Path to the preprocessed data CSV.",
    )
    parser.add_argument(
        "--X_train",
        type=str,
        required=True,
        help="Path to the preprocessed training features (X_train).",
    )
    parser.add_argument(
        "--X_val",
        type=str,
        required=True,
        help="Path to the preprocessed validation features (X_val).",
    )
    parser.add_argument(
        "--y_train",
        type=str,
        required=True,
        help="Path to the preprocessed training labels (y_train).",
    )
    parser.add_argument(
        "--y_val",
        type=str,
        required=True,
        help="Path to the preprocessed validation labels (y_val).",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )

    args = parser.parse_args()

    trainer = LSTMTrainer(args.preprocessed_data)

    trainer.load_train_val_data(args.X_train, args.X_val, args.y_train, args.y_val)

    trainer.build_model()
    trainer.train_model(epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
