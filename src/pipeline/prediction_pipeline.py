import os
import sys

from dataclasses import dataclass

from src.utils import load_pickle
from src.logger import logger
from src.exception import CustomException


# Configuration class
@dataclass
class PredictionPipelineConfig:

    model_path = os.path.join("artifacts", "fraud_model.pkl")
    threshold_path = os.path.join("artifacts", "threshold.pkl")


# Prediction Pipeline
class PredictionPipeline:

    def __init__(self):

        self.config = PredictionPipelineConfig()

    # Load trained model
    def load_model(self):

        logger.info("Loading trained model")

        return load_pickle(self.config.model_path)

    # Load prediction threshold
    def load_threshold(self):

        logger.info("Loading prediction threshold")

        return load_pickle(self.config.threshold_path)

    # Predict fraud probability
    def predict_probability(self, data):

        try:

            logger.info("Generating prediction probability")

            model = self.load_model()

            probability = model.predict_proba(data)[:, 1]

            return probability

        except Exception as e:

            logger.error(e)
            raise CustomException(e, sys)

    # Predict fraud label
    def predict(self, data):

        try:

            logger.info("Generating prediction")

            probability = self.predict_probability(data)
            threshold = self.load_threshold()

            prediction = (probability >= threshold).astype(int)

            return prediction

        except Exception as e:

            logger.error(e)
            raise CustomException(e, sys)
