"""
prediction.py

Handles:
1. Loading saved model assets
2. Preprocessing user input
3. Prediction
4. Probability estimation
"""

import joblib
import pandas as pd


class ChurnPredictor:
    """
    Customer Churn Prediction Engine
    """

    def __init__(
        self,
        model_path="models/logistic_regression.pkl",
        scaler_path="models/scaler.pkl",
        columns_path="models/feature_columns.pkl",
    ):

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(columns_path)

    # ----------------------------------------------------

    def preprocess_input(self, user_input: dict):

        """
        Convert dictionary input
        into model-ready dataframe.
        """

        data = pd.DataFrame([user_input])

        # One-hot encoding
        data = pd.get_dummies(
            data,
            drop_first=True,
            dtype=int
        )

        # Match training columns
        data = data.reindex(
            columns=self.feature_columns,
            fill_value=0
        )

        # Scaling
        scaled = self.scaler.transform(data)

        scaled = pd.DataFrame(
            scaled,
            columns=self.feature_columns
        )

        return scaled

    # ----------------------------------------------------

    def predict(self, user_input: dict):

        processed = self.preprocess_input(user_input)

        prediction = self.model.predict(processed)[0]

        probability = self.model.predict_proba(processed)[0]

        churn_probability = probability[1]

        stay_probability = probability[0]

        return {

            "prediction": prediction,

            "churn_probability": round(
                churn_probability * 100,
                2
            ),

            "stay_probability": round(
                stay_probability * 100,
                2
            )

        }

    # ----------------------------------------------------

    def get_risk_level(self, probability):

        if probability >= 75:

            return "High Risk 🔴"

        elif probability >= 40:

            return "Medium Risk 🟠"

        else:

            return "Low Risk 🟢"

    # ----------------------------------------------------

    def get_recommendation(self, probability):

        if probability >= 75:

            return [

                "📞 Contact customer within 24 hours",

                "🎁 Offer loyalty discount",

                "📅 Recommend a long-term contract",

                "👨‍💼 Assign customer success executive"

            ]

        elif probability >= 40:

            return [

                "📧 Send engagement email",

                "🎁 Offer limited-period discount",

                "📊 Monitor customer activity"

            ]

        else:

            return [

                "✅ Customer appears stable",

                "📈 Continue regular engagement"

            ]