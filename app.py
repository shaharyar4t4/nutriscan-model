import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
import pandas as pd
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("nutriscan")

MODEL_JSON = "health_nutrition_model.json"
MODEL_PKL = "health_nutrition_model.pkl"

# Column order the model was trained on. Built explicitly so a reordered
# payload can never silently shift features.
FEATURE_COLUMNS = [
    'Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)',
    'Fiber (g)', 'Sugars (g)', 'Sodium (mg)', 'Cholesterol (mg)',
    'Water_Intake (ml)', 'Meal_Type_Dinner', 'Meal_Type_Lunch',
    'Meal_Type_Snack', 'Category_Dairy', 'Category_Fruits', 'Category_Grains',
    'Category_Meat', 'Category_Snacks', 'Category_Vegetables',
]


def _load_model():
    """Load the classifier, preferring XGBoost's native format.

    Deliberately lets exceptions propagate: a service that cannot predict
    should fail at startup rather than accept traffic and return 500s while
    its health check still reports OK.
    """
    if os.path.exists(MODEL_JSON):
        model = XGBClassifier()
        model.load_model(MODEL_JSON)
        logger.info("Model loaded from native JSON.")
        return model, "native-json"

    import joblib
    model = joblib.load(MODEL_PKL)
    logger.warning(
        "Loaded legacy pickle. Run convert_model.py to produce %s and remove "
        "the XGBoost version warning.", MODEL_JSON
    )
    return model, "legacy-pickle"


xgb_model, MODEL_SOURCE = _load_model()


# --- Severity tiers --------------------------------------------------------
# MODERATE is advisory and never changes the verdict on its own. HIGH and
# VERY_HIGH force "Unhealthy".
#
# The verdict and the risk messages are derived from a single pass (_assess),
# so the two halves of a response cannot disagree: anything that forces
# "Unhealthy" always ships the HIGH message that explains why, and a response
# whose worst message is MODERATE is always "Healthy".
MODERATE, HIGH, VERY_HIGH = "moderate", "high", "very_high"
FORCES_UNHEALTHY = frozenset({HIGH, VERY_HIGH})

NO_RISK_MESSAGE = "No major nutritional risks detected. Balanced within healthy limits."

# The model can flag a meal that trips no individual threshold (it does this to
# whole grapes, for one). Returning "no risks detected" beside an "Unhealthy"
# verdict reads as a bug, so say where the verdict actually came from.
MODEL_ONLY_MESSAGE = (
    "Flagged as Unhealthy by the model, but no individual nutrient crossed a "
    "risk threshold. Treat this as a weak signal."
)

# --- Thresholds -----------------------------------------------------------
# Single source of truth. Previously these numbers were duplicated, with
# different values, across the risk text and the verdict override.
FAT_MODERATE, FAT_HIGH, FAT_VERY_HIGH = 15.0, 20.0, 35.0
SUGAR_MODERATE, SUGAR_HIGH, SUGAR_VERY_HIGH = 15.0, 25.0, 40.0
SODIUM_HIGH, SODIUM_VERY_HIGH = 500.0, 1000.0
CHOLESTEROL_MODERATE, CHOLESTEROL_HIGH = 200.0, 300.0
CALORIES_MODERATE, CALORIES_HIGH = 500.0, 700.0
FIBER_LOW = 2.0
PROTEIN_HIGH = 50.0
WATER_LOW = 200.0

# Intrinsic sugar (whole fruit, plain dairy) arrives packaged with fiber,
# water and protein, so it is only worth flagging at genuinely large portions.
SUGAR_INTRINSIC_HIGH = 40.0

# Free sugar with no fiber to slow it down — soft drinks, candy, juice.
ADDED_SUGAR_MIN, ADDED_SUGAR_MAX_FIBER = 10.0, 1.5

# A high-fiber plant food carries its fat as unsaturated fat (avocado, nuts,
# seeds). The payload has no saturated-fat field, so fiber plus category is
# the best available proxy for "whole food" rather than "fried or processed".
WHOLE_PLANT_MIN_FIBER = 5.0

# Salty and fried packaged snacks sit well below the general sodium and fat
# limits, so the snack bucket needs its own tighter bars.
SNACK_SODIUM_HIGH = 200.0
SNACK_FAT_HIGH, SNACK_FAT_MAX_FIBER = 10.0, 3.0

# Reporting low fiber only makes sense for something substantial; a glass of
# milk or an empty payload is not a fiber failure.
FIBER_MIN_CALORIES = 200.0


class NutritionData(BaseModel):
    Calories: float = Field(ge=0)
    Protein: float = Field(ge=0)
    Carbohydrates: float = Field(ge=0)
    Fat: float = Field(ge=0)
    Fiber: float = Field(ge=0)
    Sugars: float = Field(ge=0)
    Sodium: float = Field(ge=0)
    Cholesterol: float = Field(ge=0)
    Water_Intake: float = Field(ge=0)
    Meal_Type_Dinner: bool
    Meal_Type_Lunch: bool
    Meal_Type_Snack: bool
    Category_Dairy: bool
    Category_Fruits: bool
    Category_Grains: bool
    Category_Meat: bool
    Category_Snacks: bool
    Category_Vegetables: bool

    @model_validator(mode="after")
    def _one_hot_groups_stay_exclusive(self):
        """Reject rows the model never saw in training.

        Both flag groups are one-hot encoded with the first level dropped, so
        at most one may be true (all false is the dropped baseline).
        """
        groups = {
            "Meal_Type": ("Meal_Type_Dinner", "Meal_Type_Lunch", "Meal_Type_Snack"),
            "Category": ("Category_Dairy", "Category_Fruits", "Category_Grains",
                         "Category_Meat", "Category_Snacks", "Category_Vegetables"),
        }
        for name, fields in groups.items():
            selected = [f for f in fields if getattr(self, f)]
            if len(selected) > 1:
                raise ValueError(
                    f"{name} flags are mutually exclusive; got {selected}. "
                    f"Set at most one, or all false for the baseline level."
                )
        return self


app = FastAPI(title="NutriScan API")


def _assess(n):
    """Evaluate the nutrition profile once, returning (severity, message) pairs."""
    risks = []

    calories = n['Calories']
    protein = n['Protein']
    fat = n['Fat']
    fiber = n['Fiber']
    sugars = n['Sugars']
    sodium = n['Sodium']
    cholesterol = n['Cholesterol']
    water = n['Water_Intake']

    is_snack = n['Category_Snacks']
    # Whole fruit and plain dairy: sugar here is intrinsic, not added.
    intrinsic_sugar = n['Category_Fruits'] or n['Category_Dairy']
    # High-fiber plant food: fat here is unsaturated, not a cardiac risk.
    whole_plant = (
        (n['Category_Fruits'] or n['Category_Vegetables'])
        and fiber >= WHOLE_PLANT_MIN_FIBER
    )

    # --- Fat: cardiovascular load ---
    if fat > FAT_VERY_HIGH and not whole_plant:
        risks.append((VERY_HIGH, f"Very High Fat (>{FAT_VERY_HIGH:.0f}g): Serious risk "
                                 "for heart disease and obesity. Avoid if you have a "
                                 "cardiac condition."))
    elif fat > FAT_HIGH and not whole_plant:
        risks.append((HIGH, f"High Fat (>{FAT_HIGH:.0f}g): Not suitable for heart "
                            "patients; may raise LDL cholesterol."))
    elif fat > FAT_MODERATE:
        if whole_plant:
            risks.append((MODERATE, f"High Fat (>{FAT_MODERATE:.0f}g) from a "
                                    "high-fiber plant food: unsaturated fat, so "
                                    "calorie-dense rather than a cardiac risk."))
        else:
            risks.append((MODERATE, f"Moderate Fat (>{FAT_MODERATE:.0f}g): Acceptable "
                                    "occasionally, but watch overall daily intake."))

    # --- Sugars: diabetes / metabolic ---
    if intrinsic_sugar:
        if sugars > SUGAR_INTRINSIC_HIGH:
            risks.append((HIGH, f"Very High Sugar (>{SUGAR_INTRINSIC_HIGH:.0f}g): "
                                "intrinsic to fruit or dairy, but this portion is "
                                "large enough to spike blood sugar."))
    elif sugars > SUGAR_VERY_HIGH:
        risks.append((VERY_HIGH, f"Very High Sugar (>{SUGAR_VERY_HIGH:.0f}g): High risk "
                                 "of diabetes, weight gain, and dental issues."))
    elif sugars > SUGAR_HIGH:
        risks.append((HIGH, f"High Sugar (>{SUGAR_HIGH:.0f}g): Risk for diabetes and "
                            "blood-sugar spikes."))
    elif sugars >= ADDED_SUGAR_MIN and fiber < ADDED_SUGAR_MAX_FIBER:
        risks.append((HIGH, f"Added Sugar (>={ADDED_SUGAR_MIN:.0f}g with almost no "
                            "fiber): typical of soft drinks, candy and juice; causes "
                            "rapid blood-sugar spikes."))
    elif sugars > SUGAR_MODERATE:
        risks.append((MODERATE, f"Moderate Sugar (>{SUGAR_MODERATE:.0f}g): Fine in "
                                "moderation; avoid if pre-diabetic."))

    # --- Sodium: blood pressure / kidneys ---
    if sodium > SODIUM_VERY_HIGH:
        risks.append((VERY_HIGH, f"Very High Sodium (>{SODIUM_VERY_HIGH:.0f}mg): Strong "
                                 "risk of hypertension and kidney strain."))
    elif sodium > SODIUM_HIGH:
        risks.append((HIGH, f"High Sodium (>{SODIUM_HIGH:.0f}mg): Risk of hypertension; "
                            "limit for blood-pressure patients."))
    elif is_snack and sodium > SNACK_SODIUM_HIGH:
        risks.append((HIGH, f"Salty Snack (>{SNACK_SODIUM_HIGH:.0f}mg sodium): packaged "
                            "snacks concentrate salt; raises blood pressure."))

    # --- Processed snack fat: catches fried/packaged snacks that sit under
    #     the general fat bar. Fiber separates chips from nuts and seeds.
    if is_snack and fat > SNACK_FAT_HIGH and fiber < SNACK_FAT_MAX_FIBER:
        risks.append((HIGH, f"Processed Snack Fat (>{SNACK_FAT_HIGH:.0f}g fat with "
                            f"<{SNACK_FAT_MAX_FIBER:.0f}g fiber): typical of fried or "
                            "packaged snacks."))

    # --- Cholesterol: arterial health ---
    if cholesterol > CHOLESTEROL_HIGH:
        risks.append((HIGH, f"High Cholesterol (>{CHOLESTEROL_HIGH:.0f}mg): Increases "
                            "risk of atherosclerosis and stroke."))
    elif cholesterol > CHOLESTEROL_MODERATE:
        risks.append((MODERATE, f"Elevated Cholesterol (>{CHOLESTEROL_MODERATE:.0f}mg): "
                                "Borderline; monitor if you have heart issues."))

    # --- Calories: overall energy balance ---
    if calories > CALORIES_HIGH:
        risks.append((MODERATE, f"Very High Calories (>{CALORIES_HIGH:.0f} kcal): Heavy "
                                "meal; may contribute to weight gain if frequent."))
    elif calories > CALORIES_MODERATE:
        risks.append((MODERATE, f"High Calories (>{CALORIES_MODERATE:.0f} kcal): "
                                "Calorie-dense; balance with activity."))

    # --- Fiber: digestive health (low fiber is the risk) ---
    if fiber < FIBER_LOW and calories >= FIBER_MIN_CALORIES:
        risks.append((MODERATE, f"Low Fiber (<{FIBER_LOW:.0f}g): Poor for digestion and "
                                "gut health."))

    # --- Protein: very high protein strains kidneys ---
    if protein > PROTEIN_HIGH:
        risks.append((MODERATE, f"Very High Protein (>{PROTEIN_HIGH:.0f}g): May strain "
                                "kidneys; caution for renal patients."))

    # --- Combined / compound risk patterns ---
    if fiber < FIBER_LOW and fat > FAT_MODERATE and not whole_plant:
        risks.append((HIGH, "Low Fiber + High Fat: Digestive risks and slower "
                            "metabolism."))
    if sugars > SUGAR_HIGH and fat > FAT_HIGH and not intrinsic_sugar:
        risks.append((HIGH, "High Sugar + High Fat: Strong link to obesity and "
                            "metabolic syndrome."))
    if sodium > SODIUM_HIGH and cholesterol > CHOLESTEROL_MODERATE:
        risks.append((HIGH, "High Sodium + High Cholesterol: Compounded cardiovascular "
                            "risk."))
    if water < WATER_LOW and sodium > SODIUM_HIGH:
        risks.append((MODERATE, "Low Water + High Sodium: Risk of dehydration and water "
                                "retention."))

    return risks


def health_risks(nutrition):
    """Human-readable risk messages for a nutrition profile."""
    assessed = _assess(nutrition)
    return [message for _, message in assessed] or [NO_RISK_MESSAGE]


@app.post("/predict")
def predict(data: NutritionData):
    payload = data.model_dump()

    mapped_data = {
        'Calories (kcal)': payload['Calories'],
        'Protein (g)': payload['Protein'],
        'Carbohydrates (g)': payload['Carbohydrates'],
        'Fat (g)': payload['Fat'],
        'Fiber (g)': payload['Fiber'],
        'Sugars (g)': payload['Sugars'],
        'Sodium (mg)': payload['Sodium'],
        'Cholesterol (mg)': payload['Cholesterol'],
        'Water_Intake (ml)': payload['Water_Intake'],
        'Meal_Type_Dinner': payload['Meal_Type_Dinner'],
        'Meal_Type_Lunch': payload['Meal_Type_Lunch'],
        'Meal_Type_Snack': payload['Meal_Type_Snack'],
        'Category_Dairy': payload['Category_Dairy'],
        'Category_Fruits': payload['Category_Fruits'],
        'Category_Grains': payload['Category_Grains'],
        'Category_Meat': payload['Category_Meat'],
        'Category_Snacks': payload['Category_Snacks'],
        'Category_Vegetables': payload['Category_Vegetables'],
    }

    try:
        input_data = pd.DataFrame([mapped_data], columns=FEATURE_COLUMNS)
        prediction = xgb_model.predict(input_data)[0]
    except Exception:
        # Log the detail server-side; never hand internals to the caller.
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal server error")

    # 0 -> Healthy, 1 -> Unhealthy
    prediction_label = "Healthy" if prediction == 0 else "Unhealthy"

    assessed = _assess(payload)

    # Deterministic guardrail: a HIGH or worse nutrient finding overrides a
    # "Healthy" model prediction.
    if prediction_label == "Healthy" and any(s in FORCES_UNHEALTHY for s, _ in assessed):
        prediction_label = "Unhealthy"
        logger.info("Overridden to Unhealthy by nutrient rules.")

    risks = [message for _, message in assessed]
    if not risks:
        risks = [MODEL_ONLY_MESSAGE if prediction_label == "Unhealthy"
                 else NO_RISK_MESSAGE]

    return {
        "Prediction": prediction_label,
        "Health Risks": risks,
    }


@app.get("/")
def read_root():
    return {
        "message": "NutriScan API is running!",
        "model_source": MODEL_SOURCE,
    }
