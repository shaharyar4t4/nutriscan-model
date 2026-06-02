import warnings

import joblib

PKL_PATH = "health_nutrition_model.pkl"
JSON_PATH = "health_nutrition_model.json"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = joblib.load(PKL_PATH)


model.save_model(JSON_PATH)
print(f"✅ Native model saved -> {JSON_PATH}")
