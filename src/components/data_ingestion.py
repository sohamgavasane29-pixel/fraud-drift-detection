import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    train_path = os.path.join("data", "processed", "train_processed.csv")
    w2_path = os.path.join("data", "processed", "w2_processed.csv")
    w3_path = os.path.join("data", "processed", "w3_processed.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion component.")

        try:
            train_df = pd.read_csv(self.ingestion_config.train_path)
            w2_df = pd.read_csv(self.ingestion_config.w2_path)
            w3_df = pd.read_csv(self.ingestion_config.w3_path)

            logger.info("Processed datasets loaded successfully.")
            logger.info(f"Train Shape : {train_df.shape}")
            logger.info(f"W2 Shape : {w2_df.shape}")
            logger.info(f"W3 Shape : {w3_df.shape}")

            return train_df, w2_df, w3_df

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)