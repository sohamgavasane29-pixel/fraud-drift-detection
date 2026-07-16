from src.utils import load_pickle

model = load_pickle("artifacts/fraud_model.pkl")

print(model.feature_names_in_)