# NutriScan Model API

A FastAPI service that classifies a meal as **Healthy** or **Unhealthy** from its
nutrition values, and returns a list of specific health risks alongside the
prediction.

Prediction comes from two layers stacked on top of each other: a trained XGBoost
classifier, and a deterministic rule guardrail that can override it. Understanding
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
7. [Known limitations](#known-limitations)

---

## Project files

| File | What it does |
|---|---|
| `app.py` | The whole service: model loading, the rule engine, and both endpoints. |
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
service always loads the pickle and logs a warning. To silence it:

```bash
./myenv/Scripts/python.exe convert_model.py
```

This writes `health_nutrition_model.json` next to the pickle. `GET /` reports
which one is in use via its `model_source` field.

---

## Running the API

```bash
PYTHONIOENCODING=utf-8 ./myenv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Once it is up:

| URL | What you get |
|---|---|
| <http://127.0.0.1:8000/> | `{"message":"NutriScan API is running!","model_source":"legacy-pickle"}` |
| <http://127.0.0.1:8000/docs> | Interactive Swagger UI — easiest way to try `/predict` |

Add `--reload` during development to restart on file changes.

If the model file cannot be loaded, **the app refuses to start** and the
traceback names the missing file. It does not boot into a state where `/`
reports healthy while `/predict` returns 500s.

---

## How a prediction works, step by step

### Step 0 — The model loads once, at import

`_load_model()` runs when the module is imported, before any request is served.
It tries the native JSON first, then the pickle, and records which one it used in
`MODEL_SOURCE`. Exceptions are deliberately **not** caught — see the note above.

### Step 1 — The request is validated

A `POST /predict` body is parsed into the `NutritionData` Pydantic model. It has
exactly **18 fields**: 9 numeric nutrition values, 3 `Meal_Type_*` booleans, and
6 `Category_*` booleans.

Validation rejects, with a `422`, any request that:

- omits a field or sends the wrong type,
- sends a **negative** number (every numeric field is `Field(ge=0)`), or
- sets **more than one flag** within either the `Meal_Type_*` or the
  `Category_*` group — they are one-hot encoded, so at most one may be true.

### Step 2 — Field names are mapped to the training column names

The API field names and the model's feature names are not the same:

| API field | Model feature name |
|---|---|
| `Calories` | `Calories (kcal)` |
| `Protein` | `Protein (g)` |
| `Sodium` | `Sodium (mg)` |
| `Water_Intake` | `Water_Intake (ml)` |
| `Meal_Type_*`, `Category_*` | unchanged |

### Step 3 — A one-row DataFrame is built

```python
input_data = pd.DataFrame([mapped_data], columns=FEATURE_COLUMNS)
```

`FEATURE_COLUMNS` pins the column order the model was trained on, so a
reordered payload cannot silently shift features.

### Step 4 — The model predicts

```python
prediction = xgb_model.predict(input_data)[0]   # 0 or 1
prediction_label = "Healthy" if prediction == 0 else "Unhealthy"
```

### Step 5 — The rule guardrail can override the model

`_assess()` walks the nutrition values through the thresholds in
[The rule thresholds](#the-rule-thresholds) and returns
`(severity, message)` pairs. Severity is one of `moderate`, `high` or
`very_high`.

If the model said **Healthy** but any finding is `high` or worse, the verdict is
flipped to **Unhealthy**. `moderate` findings are advisory and never change the
verdict.

The override is **one-directional**: a model prediction of `Unhealthy` is never
re-examined, only `Healthy` can be flipped.

### Step 6 — The risk messages come from the same pass

The messages in the response are the message halves of the very same
`_assess()` result that decided Step 5. This is what keeps the two halves of a
response consistent:

- anything that forces `Unhealthy` always ships the `high` message explaining why, and
- a response whose worst finding is `moderate` is always `Healthy`.

If `_assess()` finds nothing at all, the response carries one of two messages
depending on what the model said — see the grapes case in
[Known limitations](#known-limitations).

### Step 7 — The response is assembled

```json
{ "Prediction": "Healthy" | "Unhealthy",
  "Health Risks": ["...", "..."] }
```

---

## API reference

### `GET /`

Liveness check. `model_source` is `native-json` or `legacy-pickle`.

```json
{"message": "NutriScan API is running!", "model_source": "legacy-pickle"}
```

### `POST /predict`

All 18 fields are required. Every numeric field is a `float >= 0`.

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

**Boolean fields.** `Meal_Type_*` and `Category_*` are one-hot flags from training
with the first level dropped. There is no `Meal_Type_Breakfast` field — all three
`Meal_Type_*` set to `false` is the implied baseline. At most one flag per group
may be `true`.

| Status | Meaning |
|---|---|
| `200` | Prediction returned. |
| `422` | Validation failed — missing field, wrong type, negative value, or conflicting one-hot flags. |
| `500` | Prediction raised. Body is `{"detail": "Internal server error"}`; the real cause is in the server log. |

---

## The rule thresholds

All thresholds live in one constants block at the top of `app.py` and are used by
both the verdict and the risk text, so the two cannot drift apart.

### Base thresholds

| Nutrient | `moderate` | `high` | `very_high` |
|---|---|---|---|
| Fat (g) | `>15` | `>20` | `>35` |
| Sugars (g) | `>15` | `>25` | `>40` |
| Sodium (mg) | — | `>500` | `>1000` |
| Cholesterol (mg) | `>200` | `>300` | — |
| Calories (kcal) | `>500`, `>700` | — | — |
| Fiber (g) | `<2` (only when calories `>=200`) | — | — |
| Protein (g) | `>50` | — | — |

Only `high` and `very_high` force an `Unhealthy` verdict.

### Context rules

Total fat and total sugar alone misclassify whole foods, so four context rules
adjust them. The payload has no saturated-fat or added-sugar field, so fiber and
the category flags act as proxies.

| Rule | Condition | Effect |
|---|---|---|
| **Intrinsic sugar** | `Category_Fruits` or `Category_Dairy` | Sugar is only flagged above `40g`. Whole fruit and plain dairy carry sugar packaged with fiber, water and protein. |
| **Whole plant fat** | (`Category_Fruits` or `Category_Vegetables`) and fiber `>=5g` | Fat is downgraded to `moderate`. This fat is unsaturated — avocado, nuts, seeds. |
| **Added sugar** | sugar `>=10g` and fiber `<1.5g` and not intrinsic | `high`. Catches soft drinks, candy and juice. |
| **Snack bucket** | `Category_Snacks` and (sodium `>200mg`, or fat `>10g` with fiber `<3g`) | `high`. Packaged snacks concentrate salt and fat well below the general bars; the fiber test separates chips from nuts. |

### Worked examples

| Input | Verdict | Which rule decided |
|---|---|---|
| Avocado (Fat 29.5, Fiber 13.5, Vegetables) | Healthy | Whole plant fat → downgraded to `moderate` |
| Almonds 1oz unsalted (Fat 14.2, Fiber 3.5, Snacks) | Healthy | Under every bar; fiber `>=3` clears the snack fat rule |
| Plain milk (Sugars 12, Fiber 0, Dairy) | Healthy | Intrinsic sugar |
| Large apple (Sugars 23.2, Fruits) | Healthy | Intrinsic sugar |
| Coca-Cola (Sugars 35, Fiber 0, Snacks) | Unhealthy | Sugar `>25` |
| Sweetened iced tea (Sugars 22, Fiber 0, Snacks) | Unhealthy | Added sugar |
| Potato chips 50g (Fat 18, Sodium 290, Snacks) | Unhealthy | Snack bucket, both halves |
| Diet cola (Sugars 0, Snacks) | Healthy | Nothing trips |

---

## Known limitations

### The model misclassifies some whole foods on its own

The guardrail can only add `Unhealthy` verdicts, never remove them, so a wrong
`Unhealthy` from the model passes straight through. One reproducible case:

| Input | Model raw | Confidence | Rules found | API returns |
|---|---|---|---|---|
| Grapes, 1 cup (Sugars 23.4, Fiber 1.4, Fruits) | `1` Unhealthy | 99.77% | nothing | **Unhealthy** |

Because the rules find nothing to report here, the response is explicit about
where the verdict came from rather than claiming a nutrient problem that does not
exist:

```json
{ "Prediction": "Unhealthy",
  "Health Risks": ["Flagged as Unhealthy by the model, but no individual nutrient crossed a risk threshold. Treat this as a weak signal."] }
```

**This is not fixable without retraining**, and the training dataset is not in
this repo — it has never been committed, so there is nothing to retrain from.
Options are to obtain the original dataset, or to retire the model and run the
rule engine alone.

### Not yet addressed

- **No CORS middleware**, so browser-based frontends cannot call this directly.
- **No `runtime.txt` / `.python-version`**, so the deploy platform picks a Python
  version that the pins may not have wheels for.
- **Two requirements files**, one stale, and both encoded UTF-16.
  `scikit-learn` is missing from each.
- **`__pycache__/app.cpython-313.pyc` is committed.** `.gitignore` contains only
  `myenv`.
- **No automated tests.** The tables above are a ready-made test matrix.
