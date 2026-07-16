import sys
import pandas as pd

from fastapi import FastAPI

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.entity.request_model import PredictionRequest
from src.exception import CustomException
from src.logger import logger

app = FastAPI()


@app.get("/")
def home():

    logger.info("Home endpoint accessed")

    return {"message": "Fraud Detection API is Running"}


@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        logger.info("Prediction request received")

        data = pd.DataFrame([request.features])

        pipeline = PredictionPipeline()

        probability = pipeline.predict_probability(data)[0]

        prediction = pipeline.predict(data)[0]

        logger.info("Prediction generated successfully")

        return {
            "prediction": int(prediction),
            "probability": float(probability)
        }

    except Exception as e:

        logger.error(e)

        raise CustomException(e, sys)