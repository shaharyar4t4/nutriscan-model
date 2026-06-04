import os
import warnings

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from fastapi.responses import JSONResponse
from xgboost import XGBClassifier

MODEL_JSON = "health_nutrition_model.json"
MODEL_PKL = "health_nutrition_model.pkl"

xgb_model = None
try:
    if os.path.exists(MODEL_JSON):
        xgb_model = XGBClassifier()
        xgb_model.load_model(MODEL_JSON)
        print("✅ Model loaded from native JSON.")
    else:
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xgb_model = joblib.load(MODEL_PKL)
        print("⚠️ Loaded legacy pickle. Run convert_model.py to remove the "
              "XGBoost version warning.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")

# ✅ Define input data structure
class NutritionData(BaseModel):
    Calories: int
    Protein: float
    Carbohydrates: float
    Fat: float
    Fiber: float
    Sugars: float
    Sodium: int
    Cholesterol: int
    Water_Intake: int
    Meal_Type_Dinner: bool
    Meal_Type_Lunch: bool
    Meal_Type_Snack: bool
    Category_Dairy: bool
    Category_Fruits: bool
    Category_Grains: bool
    Category_Meat: bool
    Category_Snacks: bool
    Category_Vegetables: bool

app = FastAPI()

def health_risks(nutrition):

    risks = []

    # --- Fat: cardiovascular load ---
    fat = nutrition['Fat']
    if fat > 35:
        risks.append("Very High Fat (>35g): Serious risk for heart disease and obesity. Avoid if you have a cardiac condition.")
    elif fat > 20:
        risks.append("High Fat (>20g): Not suitable for heart patients; may raise LDL cholesterol.")
    elif fat > 15:
        risks.append("Moderate Fat (>15g): Acceptable occasionally, but watch overall daily intake.")

    # --- Cholesterol: arterial health ---
    cholesterol = nutrition['Cholesterol']
    if cholesterol > 300:
        risks.append("High Cholesterol (>300mg): Increases risk of atherosclerosis and stroke.")
    elif cholesterol > 200:
        risks.append("Elevated Cholesterol (>200mg): Borderline; monitor if you have heart issues.")

    # --- Sodium: blood pressure / kidneys ---
    sodium = nutrition['Sodium']
    if sodium > 1000:
        risks.append("Very High Sodium (>1000mg): Strong risk of hypertension and kidney strain.")
    elif sodium > 500:
        risks.append("High Sodium (>500mg): Risk of hypertension; limit for blood-pressure patients.")

    # --- Sugars: diabetes / metabolic ---
    sugars = nutrition['Sugars']
    if sugars > 40:
        risks.append("Very High Sugar (>40g): High risk of diabetes, weight gain, and dental issues.")
    elif sugars > 25:
        risks.append("High Sugar (>25g): Risk for diabetes and blood-sugar spikes.")
    elif sugars > 15:
        risks.append("Moderate Sugar (>15g): Fine in moderation; avoid if pre-diabetic.")

    # --- Calories: overall energy balance ---
    calories = nutrition['Calories']
    if calories > 700:
        risks.append("Very High Calories (>700 kcal): Heavy meal; may contribute to weight gain if frequent.")
    elif calories > 500:
        risks.append("High Calories (>500 kcal): Calorie-dense; balance with activity.")

    # --- Fiber: digestive health (low fiber is the risk) ---
    fiber = nutrition['Fiber']
    if fiber < 2:
        risks.append("Low Fiber (<2g): Poor for digestion and gut health.")

    # --- Protein: very high protein strains kidneys ---
    protein = nutrition['Protein']
    if protein > 50:
        risks.append("Very High Protein (>50g): May strain kidneys; caution for renal patients.")

    # --- Combined / compound risk patterns ---
    if fiber < 2 and fat > 15:
        risks.append("Low Fiber + High Fat: Digestive risks and slower metabolism.")
    if sugars > 25 and fat > 20:
        risks.append("High Sugar + High Fat: Strong link to obesity and metabolic syndrome.")
    if sodium > 500 and cholesterol > 200:
        risks.append("High Sodium + High Cholesterol: Compounded cardiovascular risk.")
    if nutrition['Water_Intake'] < 200 and sodium > 500:
        risks.append("Low Water + High Sodium: Risk of dehydration and water retention.")

    # --- No risks detected ---
    if not risks:
        risks.append("No major nutritional risks detected. Balanced within healthy limits.")

    return risks


def _should_force_unhealthy(n):

    sugars = n['Sugars']
    fat = n['Fat']
    sodium = n['Sodium']
    calories = n['Calories']
    cholesterol = n['Cholesterol']
    fiber = n['Fiber']
    is_snack = n.get('Category_Snacks', False)

    # Very high absolute levels — kisi bhi item ke liye
    if fat > 20 or sodium > 600 or cholesterol > 250 or calories > 600:
        return True
    # Bahut zyada sugar — kuch bhi ho
    if sugars >= 18:
        return True
    # High sugar + (almost) no fiber => processed sugary (cola, candy, juice)
    if sugars >= 10 and fiber < 1.5:
        return True
    # Snack/beverage bucket ke liye sugar threshold strict
    if is_snack and sugars >= 8:
        return True
    return False


@app.post("/predict")
def predict(data: NutritionData):
    try:
        # ✅ Map data to match model columns
        mapped_data = {
            'Calories (kcal)': data.Calories,
            'Protein (g)': data.Protein,
            'Carbohydrates (g)': data.Carbohydrates,
            'Fat (g)': data.Fat,
            'Fiber (g)': data.Fiber,
            'Sugars (g)': data.Sugars,
            'Sodium (mg)': data.Sodium,
            'Cholesterol (mg)': data.Cholesterol,
            'Water_Intake (ml)': data.Water_Intake,
            'Meal_Type_Dinner': data.Meal_Type_Dinner,
            'Meal_Type_Lunch': data.Meal_Type_Lunch,
            'Meal_Type_Snack': data.Meal_Type_Snack,
            'Category_Dairy': data.Category_Dairy,
            'Category_Fruits': data.Category_Fruits,
            'Category_Grains': data.Category_Grains,
            'Category_Meat': data.Category_Meat,
            'Category_Snacks': data.Category_Snacks,
            'Category_Vegetables': data.Category_Vegetables,
        }

        input_data = pd.DataFrame([mapped_data])
        print("Input DataFrame columns:", input_data.columns.tolist())

        # ✅ Predict
        prediction = xgb_model.predict(input_data)[0]
        print("Raw prediction:", prediction)

        # ✅ Correct Mapping: 0 → Healthy, 1 → Unhealthy
        prediction_label = "Healthy" if prediction == 0 else "Unhealthy"

        # ✅ Deterministic guardrail: clearly unhealthy nutrients -> Unhealthy
        if prediction_label == "Healthy" and _should_force_unhealthy(data.dict()):
            prediction_label = "Unhealthy"
            print("Overridden to Unhealthy by nutrient rules.")

        # ✅ Health risks
        risks = health_risks(data.dict())

        # ✅ Final result
        result = {
            "Prediction": prediction_label,
            "Health Risks": risks
        }
        return result

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal Server Error: {e}"}
        )

@app.get("/")
def read_root():
    return {"message": "NutriScan API is running!"}
