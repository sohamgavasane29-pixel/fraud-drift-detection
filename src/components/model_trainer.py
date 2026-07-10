import os
import sys
import json
import pickle

import pandas as pd

from dataclasses import dataclass

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

from src.logger import logger
from src.exception import CustomException

# Configuration class
@dataclass
class ModelTrainerConfig:

    train_path = os.path.join("data", "processed", "train_processed.csv")
    w2_path = os.path.join("data", "processed", "w2_processed.csv")

    model_path = os.path.join("artifacts", "fraud_model.pkl")
    threshold_path = os.path.join("artifacts", "threshold.pkl")
    metrics_path = os.path.join("artifacts", "model_metrics.json")

# Model Trainer
class ModelTrainer:

    def __init__(self):

        self.config = ModelTrainerConfig()

    # Execute complete model training pipeline
    def initiate_model_training(self):

        try:

            logger.info("Starting Model Training")

            train_df, w2_df = self.load_data()

            x_train,y_train,x_w2,y_w2=self.split_features_target(
                train_df,
                w2_df
            )
            x_train, y_train = self.apply_smote(
                x_train,
                y_train
            
            ) 
            model = self.train_model(
                x_train,
                y_train
            )

            y_probability, metrics = self.evaluate_model(
                model,
                x_w2,
                y_w2
            )

            best_threshold = self.find_best_threshold(
                y_w2,
                y_probability
            )
            self.save_artifacts(
                model,
                best_threshold,
                metrics
            )
            logger.info("Model Training Completed Successfully")

            return (metrics,best_threshold)
        
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
                
    # Load processed datasets
    def load_data(self):

        logger.info("Loading processed datasets")

        train_df = pd.read_csv(self.config.train_path)
        w2_df = pd.read_csv(self.config.w2_path)

        return train_df, w2_df
    # Split features and target
    def split_features_target(self, train_df, w2_df):

        x_train = train_df.drop(columns=["isFraud"])
        y_train = train_df["isFraud"]

        x_w2 = w2_df.drop(columns=["isFraud"])
        y_w2 = w2_df["isFraud"]

        return x_train, y_train, x_w2, y_w2
    
    # Apply SMOTE on training data
    def apply_smote(self, x_train, y_train):

        logger.info("Applying SMOTE on training data")

        smote = SMOTE(
            random_state=42
        )

        x_train_resampled, y_train_resampled = smote.fit_resample(
            x_train,
            y_train
        )

        return x_train_resampled, y_train_resampled
    
    # Train XGBoost model
    def train_model(self, x_train, y_train):

        logger.info("Training XGBoost model")

        model = XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="aucpr",
            verbosity=0,
            n_jobs=-1
        )

        model.fit(
            x_train,
            y_train
        )

        return model
    
    # Evaluate trained model
    def evaluate_model(self, model, x_w2, y_w2):

        logger.info("Evaluating model")

        y_probability = model.predict_proba(x_w2)[:, 1]

        y_prediction = model.predict(x_w2)

        pr_auc = average_precision_score(
            y_w2,
            y_probability
        )

        precision = precision_score(
            y_w2,
            y_prediction
        )

        recall = recall_score(
            y_w2,
            y_prediction
        )

        f1 = f1_score(
            y_w2,
            y_prediction
        )

        metrics = {
            "pr_auc": pr_auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        return (
            y_probability,
            metrics
        )
    
    # Find best classification threshold
    def find_best_threshold(self, y_w2, y_probability):

        logger.info("Finding best threshold")

        precision, recall, thresholds = precision_recall_curve(
            y_w2,
            y_probability
        )

        f1_scores = (
            2 * precision[:-1] * recall[:-1]
        ) / (
            precision[:-1] + recall[:-1] + 1e-10
        )

        best_index = f1_scores.argmax()

        best_threshold = thresholds[best_index]

        return best_threshold

    # Save model artifacts
    def save_artifacts(
        self,
        model,
        best_threshold,
        metrics
    ):

        logger.info("Saving model artifacts")

        os.makedirs("artifacts", exist_ok=True)

        with open(self.config.model_path, "wb") as file:
            pickle.dump(model, file)

        with open(self.config.threshold_path, "wb") as file:
            pickle.dump(best_threshold, file)

        with open(self.config.metrics_path, "w") as file:
            json.dump(
                metrics,
                file,
                indent=4
            )

if __name__ == "__main__":

        trainer = ModelTrainer()

        metrics, best_threshold = (trainer.initiate_model_training())

        print("=" * 50)
        print("Model Evaluation Completed")
        print("=" * 50)
        print()
        print(f"Best Threshold : {best_threshold:.4f}")

        print("Artifacts saved successfully.")

        print()
        
        for key, value in metrics.items():
            print(f"{key} : {value:.4f}")