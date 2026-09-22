from src.prediction import predict_churn


customer = {
    "SeniorCitizen": 0,
    "tenure": 5,
    "MonthlyCharges": 80.0,
    "TotalCharges": 400.0,
    "gender": "Female",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
}


result = predict_churn(customer)

print("\nPrediction Result")
print("-----------------")
print("Prediction:", result["prediction"])
print("Churn Probability:", result["churn_probability"])
print("Risk Level:", result["risk_level"])