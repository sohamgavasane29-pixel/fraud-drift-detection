import sys
import os
import pickle

from src.logger import logger
from src.exception import CustomException


# Save object as pickle
def save_pickle(file_path, obj):

    try:

        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True
        )

        with open(file_path, "wb") as file:
            pickle.dump(
                obj,
                file
            )

    except Exception as e:

        logger.error(e)
        raise CustomException(e,sys)


# Load pickle object
def load_pickle(file_path):

    try:

        with open(file_path, "rb") as file:
            obj = pickle.load(file)

        return obj

    except Exception as e:

        logger.error(e)
        raise CustomException(e,sys)
    
def get_sample_request(feature_names):

    return {
        "features": {
            feature: 0
            for feature in feature_names
        }
    }