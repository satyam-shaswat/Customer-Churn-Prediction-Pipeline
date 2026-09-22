import streamlit as st
import pandas as pd

from src.prediction import predict_churn


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .risk-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }

    .metric-label {
        font-size: 14px;
        color: #666666;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict customer churn probability and identify high-risk customers.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Churn Prediction",
        "Business Insights"
    ]
)


# =========================================================
# LOAD DATA
# =========================================================

DATA_PATH = "data/telco_customer_churn_clean.csv"

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Dataset not found at: {DATA_PATH}"
    )
    st.stop()


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.header("Dashboard")

    total_customers = len(df)

    churned_customers = (
        df["Churn"]
        .value_counts()
        .get("Yes", 0)
    )

    churn_rate = (
        churned_customers / total_customers * 100
    )

    retained_customers = (
        df["Churn"]
        .value_counts()
        .get("No", 0)
    )

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "Churned Customers",
            f"{churned_customers:,}"
        )

    with col3:
        st.metric(
            "Retained Customers",
            f"{retained_customers:,}"
        )

    with col4:
        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

    st.divider()

    # -----------------------------------------------------
    # CHURN DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("Customer Churn Distribution")

    churn_counts = (
        df["Churn"]
        .value_counts()
        .rename_axis("Churn")
        .reset_index(name="Customers")
    )

    st.bar_chart(
        churn_counts.set_index("Churn")
    )

    # -----------------------------------------------------
    # CONTRACT VS CHURN
    # -----------------------------------------------------

    st.subheader("Contract Type vs Churn")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )

    st.bar_chart(contract_churn)

    # -----------------------------------------------------
    # TENURE VS CHURN
    # -----------------------------------------------------

    st.subheader("Tenure vs Churn")

    tenure_churn = (
        df.groupby("Churn")["tenure"]
        .mean()
        .round(2)
    )

    st.bar_chart(tenure_churn)


# =========================================================
# CHURN PREDICTION
# =========================================================

elif page == "Churn Prediction":

    st.header("Customer Churn Prediction")

    st.write(
        "Enter customer information below to estimate "
        "their probability of churn."
    )

    st.divider()

    # -----------------------------------------------------
    # CUSTOMER INFORMATION
    # -----------------------------------------------------

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )

    with col2:

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=10.0
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["No", "Yes"]
        )

    with col3:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.divider()

    # -----------------------------------------------------
    # SERVICES
    # -----------------------------------------------------

    st.subheader("Services")

    col1, col2, col3 = st.columns(3)

    with col1:

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

    with col3:

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )

    st.divider()

    # -----------------------------------------------------
    # PREDICTION BUTTON
    # -----------------------------------------------------

    predict_button = st.button(
        "🔮 Predict Churn",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        customer = {
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method
        }

        try:

            result = predict_churn(customer)

            prediction = result["prediction"]
            probability = result["churn_probability"]
            risk = result["risk_level"]

            st.divider()

            st.subheader("Prediction Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Predicted Churn",
                    prediction
                )

            with col2:
                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

            with col3:
                st.metric(
                    "Risk Level",
                    risk
                )

            # Risk message

            if risk == "High Risk":

                st.error(
                    "⚠️ This customer has a high estimated "
                    "risk of churn."
                )

            elif risk == "Medium Risk":

                st.warning(
                    "⚠️ This customer has a moderate estimated "
                    "risk of churn."
                )

            else:

                st.success(
                    "✅ This customer has a low estimated "
                    "risk of churn."
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

elif page == "Business Insights":

    st.header("Business Insights")

    st.write(
        "Key observations identified during exploratory "
        "data analysis."
    )

    st.divider()

    st.subheader("1. Contract Type")

    st.write(
        "Customers on month-to-month contracts show a "
        "substantially higher churn rate compared with "
        "customers on one-year and two-year contracts."
    )

    st.subheader("2. Customer Tenure")

    st.write(
        "Customers with shorter tenure are more likely "
        "to churn. Early-stage customers therefore "
        "represent an important retention segment."
    )

    st.subheader("3. Technical Support")

    st.write(
        "Customers with technical support show lower "
        "churn compared with customers without technical "
        "support."
    )

    st.subheader("4. Customer Charges")

    st.write(
        "Churned customers tend to have lower tenure and "
        "lower accumulated total charges."
    )

    st.divider()

    st.subheader("Potential Retention Strategies")

    st.markdown(
        """
        - Focus retention campaigns on new customers.
        - Encourage longer-term contracts where appropriate.
        - Identify customers with high monthly charges.
        - Promote technical support services.
        - Monitor month-to-month customers more closely.
        - Use churn probability to prioritize retention efforts.
        """
    )