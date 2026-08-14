# NutriScan Model API

A FastAPI service that classifies a meal as **Healthy** or **Unhealthy** from its
nutrition values, and returns a list of specific health risks alongside the
prediction.

Prediction comes from two layers stacked on top of each other: a trained XGBoost
classifier, and a hand-written rule guardrail that can override it. Understanding
that both layers exist is the key to understanding this project — see
[Step 5](#step-5-the-rule-guardrail-can-override-the-model).

---

## Table of contents

1. [Project files](#project-files)
2. [Setup, step by step](#setup-step-by-step)
3. [Running the API](#running-the-api)
4. [How a prediction works, step by step](#how-a-prediction-works-step-by-step)
5. [API reference](#api-reference)
6. [The rule thresholds](#the-rule-thresholds)
7. [Known issues](#known-issues)

---

## Project files

| File | What it does |
|---|---|
| `app.py` | The whole service: model loading, the two rule functions, and both endpoints. |
| `health_nutrition_model.pkl` | The trained XGBoost classifier, saved with joblib. 18 features. |
| `convert_model.py` | One-off script that re-saves the `.pkl` as XGBoost's native `.json` format. |
| `Procfile` | Railway/Heroku start command: `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| `requirements.txt` | Dependency pins. **See the warning in setup — these do not install on Python 3.14.** |
| `requirement.txt` | A stale duplicate of the above, missing `uvicorn`. Safe to delete. |
| `NutriScan-API.postman_collection.json` | Postman collection with a healthy and an unhealthy example request. |

---

## Setup, step by step

### Step 1 — Check your Python version

```bash
python --version
```

> **Warning — the pinned `requirements.txt` will not install on Python 3.14.**
> `numpy==2.3.1` and `xgboost==3.0.2` have no wheels built for Python 3.14, so
> pip falls back to compiling numpy from source and the build fails. The project
> was originally developed on **Python 3.13** (see the committed
> `__pycache__/app.cpython-313.pyc`).
>
> Two ways forward:
> - **Install Python 3.13** and use `requirements.txt` as-is, or
> - **Install unpinned** on 3.14, as shown in Step 3 below.

### Step 2 — Create a virtual environment

```bash
python -m venv myenv
```

`myenv` is already listed in `.gitignore`, so it will not be committed.

### Step 3 — Install dependencies

On **Python 3.13**, the pinned file works:

```bash
./myenv/Scripts/python.exe -m pip install -r requirements.txt
```

On **Python 3.14**, install unpinned so pip can resolve wheels that exist for
your interpreter:

```bash
./myenv/Scripts/python.exe -m pip install --only-binary :all: \
    fastapi uvicorn pandas numpy xgboost joblib scikit-learn
```

> **Note — `scikit-learn` is missing from `requirements.txt`.** The pickled model
> is an `XGBClassifier`, which is XGBoost's scikit-learn wrapper API, so
> scikit-learn belongs in the dependency list. Add it.

This has been verified working with much newer majors than the pins
(pandas 3.0.5, xgboost 3.4.0, pydantic 2.13.4), so the pins are stale rather
than load-bearing.

### Step 4 — (Optional) Convert the model to native format

`app.py` prefers `health_nutrition_model.json` if it exists, and falls back to
the `.pkl` otherwise. The `.json` is **not committed**, so out of the box the
service always loads the pickle and prints a warning. To silence it:

```bash
./myenv/Scripts/python.exe convert_model.py
```

This writes `health_nutrition_model.json` next to the pickle.

---

## Running the API

```bash
PYTHONIOENCODING=utf-8 ./myenv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

> **`PYTHONIOENCODING=utf-8` is required on Windows.** `app.py` prints emoji
> (`✅`, `⚠️`, `❌`) at import time. On a default `cp1252` Windows console this
> raises `UnicodeEncodeError` at `app.py:27` and the app never starts. Linux and
> Railway default to UTF-8, so this only bites local Windows development.

Once it is up:

| URL | What you get |
|---|---|
| <http://127.0.0.1:8000/> | `{"message":"NutriScan API is running!"}` |
| <http://127.0.0.1:8000/docs> | Interactive Swagger UI — easiest way to try `/predict` |

Add `--reload` during development to restart on file changes.

---

## How a prediction works, step by step

### Step 0 — The model loads once, at import

`app.py:14-27` runs when the module is imported, before any request is served.
It tries the native JSON first, then the pickle. If **both** fail, the exception
is caught, `xgb_model` stays `None`, and the app still starts — see
[Known issues](#known-issues).

### Step 1 — The request is validated

A `POST /predict` body is parsed into the `NutritionData` Pydantic model
(`app.py:30-48`). It has exactly **18 fields**: 9 numeric nutrition values, 3
`Meal_Type_*` booleans, and 6 `Category_*` booleans. Anything missing or of the
wrong type is rejected with a `422` before your code runs.

### Step 2 — Field names are mapped to the training column names

The API field names and the model's feature names are not the same. `app.py:151-170`
translates between them:

| API field | Model feature name |
|---|---|
| `Calories` | `Calories (kcal)` |
| `Protein` | `Protein (g)` |
| `Sodium` | `Sodium (mg)` |
| `Water_Intake` | `Water_Intake (ml)` |
| `Meal_Type_*`, `Category_*` | unchanged |

This mapping must match the model's feature names exactly, or XGBoost raises a
feature-mismatch error. The names in the shipped `.pkl` do line up with this
mapping.

### Step 3 — A one-row DataFrame is built

```python
input_data = pd.DataFrame([mapped_data])
```

XGBoost is given a single-row table because that is the shape it was trained on.

### Step 4 — The model predicts

```python
prediction = xgb_model.predict(input_data)[0]   # 0 or 1
prediction_label = "Healthy" if prediction == 0 else "Unhealthy"
```

`0` means Healthy, `1` means Unhealthy.

### Step 5 — The rule guardrail can override the model

This is the part that surprises people. `_should_force_unhealthy` (`app.py:122-144`)
is a set of hardcoded nutrient thresholds. If the model said **Healthy** but the
rules disagree, the answer is flipped to **Unhealthy**:

```python
if prediction_label == "Healthy" and _should_force_unhealthy(data.dict()):
    prediction_label = "Unhealthy"
```

Two consequences worth internalising:

- The override is **one-directional**. A model prediction of `Unhealthy` is never
  re-examined; only `Healthy` can be flipped.
- The thresholds are broad, so in practice **the rules decide most `Unhealthy`
  answers**, not the model. See [The rule thresholds](#the-rule-thresholds).

### Step 6 — The risk list is built independently

`health_risks` (`app.py:52-119`) walks the same nutrition values through its own
separate set of thresholds and appends a human-readable sentence for each one it
trips. It also checks four combination patterns (low fiber + high fat, high sugar
+ high fat, high sodium + high cholesterol, low water + high sodium). If nothing
trips, it returns a single "no major risks" message.

**This function does not look at the prediction, and the prediction does not look
at this function.** They are two independent passes over the same input, using
*different* threshold values — which is why the two halves of a response can
contradict each other.

### Step 7 — The response is assembled

```json
{ "Prediction": "Healthy" | "Unhealthy",
  "Health Risks": ["...", "..."] }
```

---

## API reference

### `GET /`

Liveness check.

```json
{"message": "NutriScan API is running!"}
```

### `POST /predict`

All 18 fields are required.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Calories": 120, "Protein": 8.0, "Carbohydrates": 20.0, "Fat": 5.0,
    "Fiber": 6.0, "Sugars": 8.0, "Sodium": 100, "Cholesterol": 10,
    "Water_Intake": 500,
    "Meal_Type_Dinner": false, "Meal_Type_Lunch": true, "Meal_Type_Snack": false,
    "Category_Dairy": false, "Category_Fruits": false, "Category_Grains": false,
    "Category_Meat": false, "Category_Snacks": false, "Category_Vegetables": true
  }'
```

```json
{
  "Prediction": "Healthy",
  "Health Risks": ["No major nutritional risks detected. Balanced within healthy limits."]
}
```

**Field types.** `Calories`, `Sodium`, `Cholesterol` and `Water_Intake` are typed
as `int`; the rest are `float`. Sending `"Calories": 105.5` returns a `422`, not a
rounded value.

**Boolean fields.** `Meal_Type_*` and `Category_*` are one-hot flags from training.
There is no `Meal_Type_Breakfast` field — all three `Meal_Type_*` set to `false`
is the implied baseline. Nothing enforces that at most one flag per group is
`true`.

| Status | Meaning |
|---|---|
| `200` | Prediction returned. |
| `422` | Pydantic validation failed — a field is missing or the wrong type. |
| `500` | Prediction raised. Body is `{"error": "Internal Server Error: ..."}`. |

---

## The rule thresholds

Both rule functions hardcode their own numbers, and **the numbers do not agree**.
This table is the single most useful thing to know when a prediction looks wrong.

| Nutrient | `health_risks` — adds a risk message at | `_should_force_unhealthy` — forces Unhealthy at |
|---|---|---|
| Fat (g) | `>15` moderate, `>20` high, `>35` very high | `>20` |
| Sodium (mg) | `>500` high, `>1000` very high | `>600` |
| Cholesterol (mg) | `>200` elevated, `>300` high | `>250` |
| Calories (kcal) | `>500` high, `>700` very high | `>600` |
| Sugars (g) | `>15` moderate, `>25` high, `>40` very high | `>=18`, **or** `>=10` when fiber `<1.5`, **or** `>=8` when `Category_Snacks` |
| Fiber (g) | `<2` low | only in combination with fat |
| Protein (g) | `>50` very high | no rule |

Note also that the comparison operators are inconsistent: sugar uses `>=` while
every other force rule uses `>`.

---

## Known issues

These are real, reproduced behaviours — not hypotheticals.

### Whole foods get forced to Unhealthy

The guardrail has no notion of intrinsic sugar (fruit, dairy) versus added sugar,
and treats unsaturated fat like saturated fat. It also never consults
`Category_Fruits` or `Category_Dairy` — only `Category_Snacks`.

| Input | Model said | API returns | Why |
|---|---|---|---|
| Avocado (Fat 29.5, Fiber 13.5) | Healthy | **Unhealthy** | `fat > 20` |
| Plain milk, 1 cup (Sugars 12, Fiber 0) | Healthy | **Unhealthy** | `sugars >= 10 and fiber < 1.5` |
| Large apple (Sugars 23.2) | Healthy | **Unhealthy** | `sugars >= 18` |

### Salty junk food slips through

| Input | Model said | API returns |
|---|---|---|
| Potato chips 50g (Fat 18, Sodium 290, Cal 274) | Healthy | **Healthy** |

Every threshold sits just under its limit. Note the model *itself* also predicted
Healthy here, so tightening the guardrail alone will not fix salty-snack
classification — there is no effective sodium rule below 600 mg.

### A response can contradict itself

Because Step 5 and Step 6 use different thresholds, `Sugars: 18` returns:

```json
{ "Prediction": "Unhealthy",
  "Health Risks": ["Moderate Sugar (>15g): Fine in moderation; avoid if pre-diabetic."] }
```

### No input range validation

Every numeric field accepts negative numbers. A body with `Calories: -500`,
`Fat: -50`, `Sugars: -30` returns `200` with `"Prediction": "Healthy"`. There are
no `Field(ge=0)` constraints.

### A failed model load is invisible

If loading raises, `app.py:26-27` catches it and leaves `xgb_model = None`. The
app still starts, `GET /` still reports `"NutriScan API is running!"`, and every
`/predict` call returns `500`. Health checks stay green while the API is dead.

### Other items

- **`data.dict()` is deprecated** (`app.py:183`, `app.py:188`). Pydantic v2 warns:
  *"The `dict` method is deprecated; use `model_dump` instead… to be removed in
  V3.0."*
- **Exception detail is leaked to clients** (`app.py:199-202`) — the raw exception
  string is returned in the response body.
- **No CORS middleware**, so browser-based frontends cannot call this directly.
- **No `runtime.txt` / `.python-version`**, so the deploy platform picks a Python
  version that the pins may not have wheels for.
- **`__pycache__/app.cpython-313.pyc` is committed.** `.gitignore` contains only
  `myenv`.
- **Per-request logging is noisy** — `app.py:173` prints the full column list on
  every call, and user nutrition values reach the logs.
- **No tests.** The threshold table above is a ready-made test matrix.
