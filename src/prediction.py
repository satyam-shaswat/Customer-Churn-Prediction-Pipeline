import joblib
import pandas as pd


# -----------------------------
# Load trained artifacts
# -----------------------------

MODEL_PATH = "models/logistic_regression.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)


# -----------------------------
# Prepare customer input
# -----------------------------

def prepare_input(customer_data):
    """
    Convert raw customer information into the exact
    feature format expected by the trained model.
    """

    df = pd.DataFrame([customer_data])

    # Convert categorical variables into dummy variables
    df = pd.get_dummies(df, drop_first=True, dtype=int)

    # Match the exact columns and order used during training
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


# -----------------------------
# Predict churn
# -----------------------------

def predict_churn(customer_data):
    """
    Predict churn for a single customer.

    Returns:
        prediction
        churn_probability
        risk_level
    """

    # Prepare input
    features = prepare_input(customer_data)

    # Scale while preserving feature names
    features_scaled = pd.DataFrame(
        scaler.transform(features),
        columns=feature_columns,
        index=features.index
    )

    # Prediction
    prediction = model.predict(features_scaled)[0]

    # Probability of churn
    probabilities = model.predict_proba(features_scaled)[0]

    # Find probability corresponding to "Yes"
    class_names = list(model.classes_)

    if "Yes" in class_names:
        churn_index = class_names.index("Yes")
        churn_probability = probabilities[churn_index]
    else:
        # Fallback in case model uses numeric labels
        churn_probability = probabilities[1]

    return {
        "prediction": prediction,
        "churn_probability": float(churn_probability),
        "risk_level": get_risk_level(churn_probability)
    }


# -----------------------------
# Risk classification
# -----------------------------

def get_risk_level(probability):
    """
    Convert churn probability into a business-friendly
    risk category.
    """

    if probability < 0.30:
        return "Low Risk"

    elif probability < 0.60:
        return "Medium Risk"

    else:
        return "High Risk"