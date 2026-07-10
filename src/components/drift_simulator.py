import os
import sys

import pandas as pd

from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException


# Configuration class
@dataclass
class DriftSimulatorConfig:

    window3_path = os.path.join(
        "data",
        "windows",
        "window_3.csv"
    )

    drifted_window_path = os.path.join(
        "data",
        "windows",
        "window_3_drifted.csv"
    )


# Drift configuration
@dataclass
class DriftConfig:

    transaction_multiplier: float = 2.5
    fraud_flip_percentage: float = 0.06
    v257_shift_factor: float = 1.5

    inject_transaction_drift: bool = True
    inject_fraud_drift: bool = True
    inject_v257_drift: bool = True


# Drift Simulator
class DriftSimulator:

    def __init__(self):

        self.config = DriftSimulatorConfig()
        self.drift_config = DriftConfig()

    # Load window 3 dataset
    def load_data(self):

        logger.info("Loading window_3 dataset")

        w3_df = pd.read_csv(self.config.window3_path)

        return w3_df

    # Inject transaction amount drift
    def shift_transaction_amount(self, w3_df):

        logger.info("Injecting Transaction Amount Drift")

        before_mean = w3_df["TransactionAmt"].mean()

        w3_df["TransactionAmt"] = w3_df["TransactionAmt"] * self.drift_config.transaction_multiplier

        after_mean = w3_df["TransactionAmt"].mean()

        logger.info(f"TransactionAmt Mean : {before_mean:.2f} -> {after_mean:.2f}")

        return w3_df

    # Inject fraud rate drift (Concept Drift)
    def increase_fraud_rate(self, w3_df):

        logger.info("Injecting Fraud Rate Drift")

        before_rate = w3_df["isFraud"].mean() * 100

        legitimate_index = w3_df[
            w3_df["isFraud"] == 0
        ].sample(
            frac=self.drift_config.fraud_flip_percentage,
            random_state=42
        ).index

        w3_df.loc[legitimate_index, "isFraud"] = 1

        after_rate = w3_df["isFraud"].mean() * 100

        logger.info(f"Fraud Rate : {before_rate:.2f}% -> {after_rate:.2f}%")

        return w3_df

    # Inject V257 distribution drift (Covariate Drift)
    def shift_v257(self, w3_df):

        logger.info("Injecting V257 Distribution Drift")

        before_mean = w3_df["V257"].mean()

        w3_df["V257"] = w3_df["V257"] + (w3_df["V257"].std() * self.drift_config.v257_shift_factor)

        after_mean = w3_df["V257"].mean()

        logger.info(f"V257 Mean : {before_mean:.4f} -> {after_mean:.4f}")

        return w3_df

    # Save drifted dataset
    def save_drifted_data(self, w3_df):

        logger.info("Saving drifted window_3 dataset")

        os.makedirs(
            os.path.dirname(self.config.drifted_window_path),
            exist_ok=True
        )

        w3_df.to_csv(
            self.config.drifted_window_path,
            index=False
        )

        logger.info("Drifted dataset saved successfully")

    # Execute complete drift simulation
    def inject_drift(self):

        try:

            logger.info("Starting Drift Simulation")

            w3_df = self.load_data()

            if self.drift_config.inject_transaction_drift:
                w3_df = self.shift_transaction_amount(w3_df)

            if self.drift_config.inject_fraud_drift:
                w3_df = self.increase_fraud_rate(w3_df)

            if self.drift_config.inject_v257_drift:
                w3_df = self.shift_v257(w3_df)

            self.save_drifted_data(w3_df)

            logger.info("Drift Simulation Completed Successfully")

            return w3_df

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)


if __name__ == "__main__":

    simulator = DriftSimulator()

    drifted_df = simulator.inject_drift()
    
    print("Drift Simulation Completed")

    print(f"Shape : {drifted_df.shape}")
    print(f"Fraud Rate : {drifted_df['isFraud'].mean() * 100:.2f}%")
        
