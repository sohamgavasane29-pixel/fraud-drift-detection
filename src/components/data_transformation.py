import os
import sys
import pickle
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler

from src.logger import logger
from src.exception import CustomException


# Configuration class
@dataclass
class DataTransformationConfig:

    train_path = os.path.join("data", "windows", "window_1.csv")
    w2_path = os.path.join("data", "windows", "window_2.csv")
    w3_path = os.path.join("data", "windows", "window_3.csv")

    train_processed_path = os.path.join("data", "processed", "train_processed.csv")
    w2_processed_path = os.path.join("data", "processed", "w2_processed.csv")
    w3_processed_path = os.path.join("data", "processed", "w3_processed.csv")

    scaler_path = os.path.join("artifacts", "scaler.pkl")
    feature_columns_path = os.path.join("artifacts", "feature_columns.pkl")


# Data Transformation
class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()
        self.scaler = StandardScaler()

    # Load all window datasets
    def load_data(self):

        logger.info("Loading window datasets")

        train_df = pd.read_csv(self.config.train_path)
        w2_df = pd.read_csv(self.config.w2_path)
        w3_df = pd.read_csv(self.config.w3_path)

        logger.info("All datasets loaded successfully")

        return train_df, w2_df, w3_df

    # Create missing value indicators
    def create_missing_indicators(self, df):

        high_null_cols = [
            "D7",
            "D8",
            "V257",
            "V246",
            "V201",
            "V200",
            "V189",
            "V188",
            "R_emaildomain"
        ]

        for col in high_null_cols:
            df[col + "_missing"] = df[col].isnull().astype(int)

        return df

    # Numerical imputation
    def numerical_imputation(self, train_df, w2_df, w3_df):

        numerical_columns = train_df.select_dtypes(include=np.number).columns.tolist()

        numerical_columns = [
            col for col in numerical_columns
            if col not in ["TransactionID", "isFraud"]
        ]

        median_values = train_df[numerical_columns].median()

        train_df[numerical_columns] = train_df[numerical_columns].fillna(median_values)
        w2_df[numerical_columns] = w2_df[numerical_columns].fillna(median_values)
        w3_df[numerical_columns] = w3_df[numerical_columns].fillna(median_values)

        return train_df, w2_df, w3_df

    # Categorical imputation
    def categorical_imputation(self, train_df, w2_df, w3_df):

        categorical_columns = train_df.select_dtypes(include="object").columns.tolist()

        for col in categorical_columns:

            if col == "R_emaildomain":
                train_df[col] = train_df[col].fillna("Unknown")
                w2_df[col] = w2_df[col].fillna("Unknown")
                w3_df[col] = w3_df[col].fillna("Unknown")

            else:

                mode_value = train_df[col].mode()[0]

                train_df[col] = train_df[col].fillna(mode_value)
                w2_df[col] = w2_df[col].fillna(mode_value)
                w3_df[col] = w3_df[col].fillna(mode_value)

        return train_df, w2_df, w3_df

        # Encode M columns
    def encode_m_columns(self, train_df, w2_df, w3_df):

        m_columns = [
            "M1", "M2", "M3", "M4",
            "M5", "M6", "M7", "M8", "M9"
        ]

        mapping = {
            "T": 1,
            "F": 0
        }

        for col in m_columns:

            if col in train_df.columns:

                train_df[col] = train_df[col].map(mapping).fillna(0).astype(int)
                w2_df[col] = w2_df[col].map(mapping).fillna(0).astype(int)
                w3_df[col] = w3_df[col].map(mapping).fillna(0).astype(int)

        return train_df, w2_df, w3_df

    # Target encoding
    def target_encoding(self, train_df, w2_df, w3_df):

        categorical_columns = [
            "ProductCD",
            "card4",
            "card6",
            "P_emaildomain",
            "R_emaildomain"
        ]

        global_mean = train_df["isFraud"].mean()

        for col in categorical_columns:

            encoding_map = train_df.groupby(col)["isFraud"].mean()

            train_df[col] = train_df[col].map(encoding_map).fillna(global_mean)
            w2_df[col] = w2_df[col].map(encoding_map).fillna(global_mean)
            w3_df[col] = w3_df[col].map(encoding_map).fillna(global_mean)

        return train_df, w2_df, w3_df

    # Remove zero variance columns
    def remove_zero_variance(self, train_df, w2_df, w3_df):

        constant_columns = []

        for col in train_df.columns:

            if col == "isFraud":
                continue

            if train_df[col].nunique() == 1:
                constant_columns.append(col)

        train_df.drop(columns=constant_columns, inplace=True)
        w2_df.drop(columns=constant_columns, inplace=True)
        w3_df.drop(columns=constant_columns, inplace=True)

        logger.info(f"Removed zero variance columns: {constant_columns}")

        return train_df, w2_df, w3_df

    # Standard scaling
    def scale_features(self, train_df, w2_df, w3_df):

        feature_columns = [
            col for col in train_df.columns
            if col not in ["isFraud", "TransactionID"]
        ]

        x_train = train_df[feature_columns]
        x_w2 = w2_df[feature_columns]
        x_w3 = w3_df[feature_columns]

        y_train = train_df["isFraud"]
        y_w2 = w2_df["isFraud"]
        y_w3 = w3_df["isFraud"]

        x_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(x_train),
            columns=feature_columns,
            index=train_df.index
        )

        x_w2_scaled = pd.DataFrame(
            self.scaler.transform(x_w2),
            columns=feature_columns,
            index=w2_df.index
        )

        x_w3_scaled = pd.DataFrame(
            self.scaler.transform(x_w3),
            columns=feature_columns,
            index=w3_df.index
        )

        x_train_scaled["isFraud"] = y_train.values
        x_w2_scaled["isFraud"] = y_w2.values
        x_w3_scaled["isFraud"] = y_w3.values

        return (
            x_train_scaled,
            x_w2_scaled,
            x_w3_scaled,
            feature_columns
        )

    # Save transformed datasets and artifacts
    def save_artifacts(self, train_df, w2_df, w3_df, feature_columns):

        os.makedirs("artifacts", exist_ok=True)
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)

        train_df.to_csv(self.config.train_processed_path, index=False)
        w2_df.to_csv(self.config.w2_processed_path, index=False)
        w3_df.to_csv(self.config.w3_processed_path, index=False)

        with open(self.config.scaler_path, "wb") as file:
            pickle.dump(self.scaler, file)

        with open(self.config.feature_columns_path, "wb") as file:
            pickle.dump(feature_columns, file)

        logger.info("Processed datasets saved successfully.")
        logger.info("Scaler saved successfully.")
        logger.info("Feature columns saved successfully.")

    # Execute complete data transformation pipeline
    def initiate_data_transformation(self):

        try:

            logger.info("Starting Data Transformation")

            train_df, w2_df, w3_df = self.load_data()

            train_df = self.create_missing_indicators(train_df)
            w2_df = self.create_missing_indicators(w2_df)
            w3_df = self.create_missing_indicators(w3_df)

            train_df, w2_df, w3_df = self.numerical_imputation(
                train_df,
                w2_df,
                w3_df
            )

            train_df, w2_df, w3_df = self.categorical_imputation(
                train_df,
                w2_df,
                w3_df
            )

            train_df, w2_df, w3_df = self.encode_m_columns(
                train_df,
                w2_df,
                w3_df
            )

            train_df, w2_df, w3_df = self.target_encoding(
                train_df,
                w2_df,
                w3_df
            )

            train_df, w2_df, w3_df = self.remove_zero_variance(
                train_df,
                w2_df,
                w3_df
            )

            train_df, w2_df, w3_df, feature_columns = self.scale_features(
                train_df,
                w2_df,
                w3_df
            )

            self.save_artifacts(
                train_df,
                w2_df,
                w3_df,
                feature_columns
            )

            logger.info("Data Transformation Completed Successfully")

            return (
                train_df,
                w2_df,
                w3_df,
                feature_columns
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)


if __name__ == "__main__":

    transformer = DataTransformation()

    train_df, w2_df, w3_df, feature_columns = (
        transformer.initiate_data_transformation()
    )

    print("=" * 50)
    print("Data Transformation Completed Successfully")
    print("=" * 50)

    print(f"Train Shape : {train_df.shape}")
    print(f"W2 Shape    : {w2_df.shape}")
    print(f"W3 Shape    : {w3_df.shape}")
    print(f"Features    : {len(feature_columns)}")