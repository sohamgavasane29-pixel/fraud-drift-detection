import os
import sys

import pandas as pd

from dataclasses import dataclass

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.logger import logger
from src.exception import CustomException


# Configuration class
@dataclass
class DriftDetectorConfig:

    reference_path = os.path.join(
        "data",
        "windows",
        "window_1.csv"
    )

    current_path = os.path.join(
        "data",
        "windows",
        "window_3_drifted.csv"
    )

    report_path = os.path.join(
        "reports",
        "drift_report.html"
    )


# Drift Detector
class DriftDetector:

    def __init__(self):

        self.config = DriftDetectorConfig()

    # Execute complete drift detection
    def detect_drift(self):

        try:

            logger.info("Starting Drift Detection")

            reference_df = self.load_reference_data()
            current_df = self.load_current_data()

            report = self.generate_report(
                reference_df,
                current_df
            )

            self.save_report(report)

            logger.info("Drift Detection Completed Successfully")

            return report

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    # Load reference dataset
    def load_reference_data(self):

        logger.info("Loading reference dataset")

        reference_df = pd.read_csv(
            self.config.reference_path
        )

        return reference_df

    # Load current dataset
    def load_current_data(self):

        logger.info("Loading current dataset")

        current_df = pd.read_csv(
            self.config.current_path
        )

        return current_df

    # Generate drift report
    def generate_report(
        self,
        reference_df,
        current_df
    ):

        logger.info("Generating drift report")

        report = Report(
            metrics=[
                DataDriftPreset()
            ]
        )

        report.run(
            reference_data=reference_df,
            current_data=current_df
        )

        return report

    # Save drift report
    def save_report(
        self,
        report
    ):

        logger.info("Saving drift report")

        os.makedirs(
            "reports",
            exist_ok=True
        )

        report.save_html(
            self.config.report_path
        )

        logger.info(
            "Drift report saved successfully"
        )


if __name__ == "__main__":

    detector = DriftDetector()

    detector.detect_drift()

    print(detector.config.report_path)