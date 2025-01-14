import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PreprocessData:
    def __init__(
        self,
        input_path=None,
        store_path=None,
        output_path=None,
        df=None,
        missing_threshold=0.6,
    ):
        """
        Initialize the PreprocessData class.

        Parameters:
        - input_path: Path to the primary input data file.
        - store_path: Path to the secondary input data file for merging.
        - output_path: Path to save the processed data.
        - df: Optional DataFrame provided directly.
        - missing_threshold: Threshold for dropping columns with missing values.
        """
        self.input_path = input_path
        self.store_path = store_path
        self.output_path = output_path
        self.missing_threshold = missing_threshold
        self.df = df
        self.categorical_columns = [
            "DayOfWeek",
            "Open",
            "Promo",
            "StateHoliday",
            "SchoolHoliday",
            "StoreType",
            "Assortment",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2",
            "Promo2SinceWeek",
            "Promo2SinceYear",
            "PromoInterval",
        ]

        self.numerical_columns = [
            "Sales",
            "Customers",
            "CompetitionDistance",
        ]

    def load_data(self):
        """Load data from the input files if DataFrame is not provided."""
        if self.df is not None:
            logger.info("DataFrame provided directly. Skipping file loading.")
            return

        if not self.input_path or not self.store_path:
            raise ValueError(
                "Both input_path and store_path must be provided if no DataFrame is passed."
            )

        try:
            logger.info("Loading data...")
            df = pd.read_csv(self.input_path)
            store_df = pd.read_csv(self.store_path)
            self.df = df.merge(store_df, on="Store", how="inner")
            logger.info(f"Data loaded and merged successfully. Shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    @staticmethod
    def display_unique_values(df):
        """Print the number of unique values in each column."""
        print("Number of Unique Values in Each Column:\n")
        print("_" * 70, f"{'Column Name':>45} |  Unique Values", "_" * 70, sep="\n")
        for col in df.columns:
            print(f"{col:>45} | {df[col].nunique()}")
        print("_" * 70)

    @staticmethod
    def missing_values_proportions(df):
        """Calculate and return missing values proportions."""
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]

        missing_proportions = (missing_values / len(df)) * 100
        missing_proportions = missing_proportions.round(2)

        return pd.DataFrame(
            {"Missing Values": missing_values, "Proportion (%)": missing_proportions}
        )

    def drop_missing_columns(self):
        """Drop columns with missing values exceeding the specified threshold."""
        logger.info("Dropping columns with excessive missing values...")
        initial_columns = self.df.shape[1]
        threshold = self.missing_threshold * self.df.shape[0]
        self.df = self.df.loc[:, self.df.isnull().sum() <= threshold]
        dropped_columns = initial_columns - self.df.shape[1]
        logger.info(f"Dropped {dropped_columns} columns with missing values.")

    def handle_missing_values(self):
        """Handle missing values in numerical and categorical columns."""
        logger.info("Handling missing values...")
        for col in self.numerical_columns:
            if self.df[col].isnull().any():
                mean_value = self.df[col].mean()
                self.df[col].fillna(mean_value, inplace=True)
                logger.info(
                    f"Filled missing values in numerical column '{col}' with mean: {mean_value:.2f}"
                )

        for col in self.categorical_columns:
            if self.df[col].isnull().any():
                mode_value = self.df[col].mode()[0]
                self.df[col].fillna(mode_value, inplace=True)
                logger.info(
                    f"Filled missing values in categorical column '{col}' with mode: '{mode_value}'"
                )

    def handle_outliers(self, method="iqr", plot=False):
        """Handle outliers using the specified method (IQR or Z-score)."""
        logger.info("Handling outliers...")

        if plot:
            plt.figure(figsize=(10, len(self.numerical_columns)))
            plt.subplot(1, 2, 1)
            self.df[self.numerical_columns].boxplot(vert=False)
            plt.title("Boxplot (Before Outlier Handling)")

        for col in self.numerical_columns:
            if method == "iqr":
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_count = outliers.sum()
                logger.info(
                    f"Column '{col}': Detected {outlier_count} outliers using IQR."
                )

                if outlier_count > 0:
                    mean_value = self.df[col].mean()
                    self.df.loc[outliers, col] = mean_value

            elif method == "zscore":
                z_scores = np.abs(zscore(self.df[col]))
                outliers = z_scores > 3
                outlier_count = outliers.sum()
                logger.info(
                    f"Column '{col}': Detected {outlier_count} outliers using Z-score."
                )

                if outlier_count > 0:
                    mean_value = self.df[col].mean()
                    self.df.loc[outliers, col] = mean_value
            else:
                raise ValueError("Invalid method. Use 'iqr' or 'zscore'.")

        if plot:
            plt.subplot(1, 2, 2)
            self.df[self.numerical_columns].boxplot(vert=False)
            plt.title("Boxplot (After Outlier Handling)")
            plt.tight_layout()
            plt.show()

    def prepare_data(self):
        """Prepare data for model training by splitting and preprocessing."""
        logger.info("Preparing data for training...")

        # Separate features and target
        X = self.df.drop(columns=["Sales"])
        y_sales = self.df["Sales"]

        # Filter columns based on availability
        self.numerical_columns = [
            col for col in self.numerical_columns if col in X.columns
        ]
        print(f"{self.numerical_columns=}")
        self.categorical_columns = [
            col for col in self.categorical_columns if col in X.columns
        ]
        print(f"{self.categorical_columns=}")

        # Convert categorical columns to strings to handle mixed types
        logger.info("Converting categorical columns to strings...")
        X[self.categorical_columns] = X[self.categorical_columns].astype(str)

        # Preprocess features using ColumnTransformer
        self.data_preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numerical_columns),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    self.categorical_columns,
                ),
            ]
        )

        logger.info("Applying preprocessing transformations...")
        X_transformed = self.data_preprocessor.fit_transform(X)

        # Scale target variable
        logger.info("Scaling target variable...")
        self.y_scaler = StandardScaler()
        y_sales_scaled = self.y_scaler.fit_transform(y_sales.values.reshape(-1, 1))

        # Split data for training and validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_transformed, y_sales_scaled, test_size=0.25, shuffle=False
        )

        print(
            self.X_train.shape, self.X_val.shape, self.y_train.shape, self.y_val.shape
        )

        # Ensure the output directory exists
        os.makedirs("./data/processed", exist_ok=True)

        # Save each array in pickle format
        logger.info("Saving processed data splits in pickle format...")
        with open("./data/processed/X_train.pkl", "wb") as f:
            pickle.dump(self.X_train, f)
        with open("./data/processed/X_val.pkl", "wb") as f:
            pickle.dump(self.X_val, f)
        with open("./data/processed/y_train.pkl", "wb") as f:
            pickle.dump(self.y_train, f)
        with open("./data/processed/y_val.pkl", "wb") as f:
            pickle.dump(self.y_val, f)

        logger.info("Data splits and scaler saved as pickle files.")

    def save_data(self):
        """Save the processed data to the specified output path."""
        if not self.output_path:
            logger.warning("No output path specified. Skipping save.")
            return

        logger.info("Saving processed data...")
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.df.to_csv(self.output_path, index=False)
            logger.info(f"Processed data saved to {self.output_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise

    def run(self):
        """Execute the complete preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline...")
        self.load_data()
        self.drop_missing_columns()
        self.handle_missing_values()
        self.handle_outliers()
        self.prepare_data()
        self.save_data()
        logger.info("Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a dataset for analysis.")
    parser.add_argument("--input_path", type=str, help="Path to the input data file.")
    parser.add_argument(
        "--store_path", type=str, help="Path to the secondary data file for merging."
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to save the processed data."
    )
    parser.add_argument(
        "--missing_threshold",
        type=float,
        default=0.6,
        help="Threshold to drop columns with missing values.",
    )
    args = parser.parse_args()

    preprocessor = PreprocessData(
        input_path=args.input_path,
        store_path=args.store_path,
        output_path=args.output_path,
        missing_threshold=args.missing_threshold,
    )
    preprocessor.run()
