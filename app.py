
import os
import subprocess
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model

from flask import Flask, request, jsonify, render_template


# ==========================================
# 1. INITIALIZE FLASK APPLICATION
# ==========================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent


# ==========================================
# 2. LOAD ML AND DL MODELS
# ==========================================

# Existing Random Forest model
MODEL_PATH = BASE_DIR / "models" / "bank_churn_pipeline.pkl"

# Newly trained Random Forest model
ML_MODEL_PATH = BASE_DIR / "models" / "bank_churn_pipeline_ml.pkl"

# ANN model
ANN_MODEL_PATH = BASE_DIR / "models" / "bank_churn_ann.keras"

# ANN preprocessing pipeline
ANN_PREPROCESSOR_PATH = (
    BASE_DIR / "models" / "bank_churn_ann_preprocessor.pkl"
)

# Check model files
required_model_files = [
    MODEL_PATH,
    ML_MODEL_PATH,
    ANN_MODEL_PATH,
    ANN_PREPROCESSOR_PATH
]

for path in required_model_files:
    if not path.exists():
        raise FileNotFoundError(f"Required model file not found: {path}")

# Load models
model = joblib.load(MODEL_PATH)
ml_model = joblib.load(ML_MODEL_PATH)
ann_model = load_model(ANN_MODEL_PATH)
ann_preprocessor = joblib.load(ANN_PREPROCESSOR_PATH)

print("Existing Random Forest model loaded successfully!")
print("New Random Forest ML model loaded successfully!")
print("ANN Deep Learning model loaded successfully!")
print("ANN preprocessor loaded successfully!")


# ==========================================
# 3. LOAD ORIGINAL DATASET
# ==========================================

DATA_PATH = BASE_DIR / "dataset" / "Churn_Modelling.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

# Read-only loading. Original CSV is never modified.
analytics_df = pd.read_csv(DATA_PATH)

print("Original dataset loaded successfully!")


# ==========================================
# 4. REQUIRED CUSTOMER FEATURES
# ==========================================

REQUIRED_FIELDS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

NUMERIC_FIELDS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]


# ==========================================
# 5. HOMEPAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# 6. HEALTH CHECK API
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "Bank Customer Churn Prediction API is running!",
        "models": {
            "machine_learning": "Random Forest",
            "deep_learning": "Artificial Neural Network"
        }
    })


# ==========================================
# 7. DASHBOARD ANALYTICS API
# ==========================================

@app.route("/analytics", methods=["GET"])
def analytics():

    # Customer distribution
    churn_counts = (
        analytics_df["Exited"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    # Geography-wise churn percentage
    geography_churn = (
        analytics_df.groupby("Geography")["Exited"]
        .mean() * 100
    )

    # Gender-wise churn percentage
    gender_churn = (
        analytics_df.groupby("Gender")["Exited"]
        .mean() * 100
    )

    # Age-group-wise churn percentage
    age_groups = pd.cut(
        analytics_df["Age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"]
    )

    age_churn = (
        analytics_df.groupby(age_groups, observed=False)["Exited"]
        .mean() * 100
    )

    return jsonify({
        "status": "success",

        "customer_distribution": {
            "Stayed": int(churn_counts.loc[0]),
            "Exited": int(churn_counts.loc[1])
        },

        "geography_churn": {
            str(key): round(float(value), 2)
            for key, value in geography_churn.items()
        },

        "gender_churn": {
            str(key): round(float(value), 2)
            for key, value in gender_churn.items()
        },

        "age_churn": {
            str(key): round(float(value), 2)
            for key, value in age_churn.items()
        }
    })


# ==========================================
# 8. CUSTOMER INPUT VALIDATION
# ==========================================

def validate_customer(data):

    if not isinstance(data, dict):
        return None, "Please provide valid JSON customer data."

    # Check missing fields
    missing = [
        field for field in REQUIRED_FIELDS
        if field not in data
    ]

    if missing:
        return None, f"Missing required fields: {missing}"

    # Create customer DataFrame
    customer = pd.DataFrame([
        {field: data[field] for field in REQUIRED_FIELDS}
    ])

    # Convert numeric fields
    try:
        for field in NUMERIC_FIELDS:
            customer[field] = pd.to_numeric(
                customer[field],
                errors="raise"
            )
    except (ValueError, TypeError):
        return None, "Numeric fields must contain valid numbers."

    # Check finite numerical values
    if not np.isfinite(
        customer[NUMERIC_FIELDS].to_numpy(dtype=float)
    ).all():
        return None, "Numeric fields must contain finite values."

    # Check categorical values
    if customer["Geography"].iloc[0] not in [
        "France", "Germany", "Spain"
    ]:
        return None, "Geography must be France, Germany, or Spain."

    if customer["Gender"].iloc[0] not in [
        "Male", "Female"
    ]:
        return None, "Gender must be Male or Female."

    # Validate numeric ranges
    if not 300 <= customer["CreditScore"].iloc[0] <= 900:
        return None, "Credit score must be between 300 and 900."

    if not 18 <= customer["Age"].iloc[0] <= 100:
        return None, "Age must be between 18 and 100."

    if not 0 <= customer["Tenure"].iloc[0] <= 10:
        return None, "Tenure must be between 0 and 10."

    if customer["Balance"].iloc[0] < 0:
        return None, "Balance cannot be negative."

    if customer["NumOfProducts"].iloc[0] not in [1, 2, 3, 4]:
        return None, "Number of products must be between 1 and 4."

    if customer["HasCrCard"].iloc[0] not in [0, 1]:
        return None, "HasCrCard must be 0 or 1."

    if customer["IsActiveMember"].iloc[0] not in [0, 1]:
        return None, "IsActiveMember must be 0 or 1."

    if customer["EstimatedSalary"].iloc[0] < 0:
        return None, "Estimated salary cannot be negative."

    return customer, None


# ==========================================
# 9. CUSTOMER CHURN PREDICTION API - ML + DL
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True)

        customer, validation_error = validate_customer(data)

        if validation_error:
            return jsonify({
                "status": "error",
                "error": validation_error
            }), 400

        # --------------------------------------
        # MACHINE LEARNING: RANDOM FOREST
        # --------------------------------------

        ml_prediction = int(
            ml_model.predict(customer)[0]
        )

        ml_probability = float(
            ml_model.predict_proba(customer)[0][1]
        )

        # --------------------------------------
        # DEEP LEARNING: ANN
        # --------------------------------------

        ann_input = ann_preprocessor.transform(customer)

        if hasattr(ann_input, "toarray"):
            ann_input = ann_input.toarray()

        ann_input = np.asarray(
            ann_input,
            dtype=np.float32
        )

        ann_probability = float(
            ann_model.predict(
                ann_input,
                verbose=0
            ).ravel()[0]
        )

        ann_prediction = int(ann_probability >= 0.5)

        # --------------------------------------
        # RETURN BOTH PREDICTIONS
        # --------------------------------------

        return jsonify({
            "status": "success",

            "machine_learning": {
                "model": "Random Forest",
                "prediction": (
                    "Exited" if ml_prediction == 1 else "Stayed"
                ),
                "churn_probability": round(
                    ml_probability * 100, 2
                )
            },

            "deep_learning": {
                "model": "Artificial Neural Network",
                "prediction": (
                    "Exited" if ann_prediction == 1 else "Stayed"
                ),
                "churn_probability": round(
                    ann_probability * 100, 2
                )
            }
        })

    except (ValueError, TypeError) as error:
        return jsonify({
            "status": "error",
            "error": f"Invalid customer input: {str(error)}"
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "status": "error",
            "error": "Something went wrong while generating predictions."
        }), 500


# ==========================================
# 10. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    url = "http://127.0.0.1:5000/"

    def open_external_browser():

        browser_paths = [
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
            / "Google/Chrome/Application/chrome.exe",

            Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
            / "Google/Chrome/Application/chrome.exe",

            Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
            / "Microsoft/Edge/Application/msedge.exe",

            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Google/Chrome/Application/chrome.exe",

            Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
            / "Microsoft/Edge/Application/msedge.exe"
        ]

        browser = next(
            (path for path in browser_paths if path.exists()),
            None
        )

        if browser:
            subprocess.Popen([str(browser), url])
        else:
            os.startfile(url)

    # Open browser after Flask starts
    threading.Timer(
        1.5,
        open_external_browser
    ).start()

    app.run(
        debug=True,
        use_reloader=False
    )
