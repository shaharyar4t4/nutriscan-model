"""
One-time converter: purane XGBoost se pickle hua model -> native JSON format.

Kyun: pickle/joblib XGBoost ke internal binary format ko version ke beech
guarantee nahi karta, isliye naye version (yahan xgboost==3.0.2) me load karte
waqt warning aati hai (aur kabhi-kabhi silently galat predictions). Native
`save_model`/`load_model` (.json) version-portable hai.

Kaise chalao (ek hi baar, usi env me jahan pkl load ho jaata hai):

    python convert_model.py

Phir bani hui `health_nutrition_model.json` ko commit/deploy kar do.
"""
import warnings

import joblib

PKL_PATH = "health_nutrition_model.pkl"
JSON_PATH = "health_nutrition_model.json"

# Aakhri baar pickle load (warning ignore — yeh expected hai).
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = joblib.load(PKL_PATH)

# XGBoost native format me save (XGBClassifier / Booster dono pe available).
model.save_model(JSON_PATH)
print(f"✅ Native model saved -> {JSON_PATH}")
print("   Ab ise commit karke deploy karo; app.py isi se load karega.")
