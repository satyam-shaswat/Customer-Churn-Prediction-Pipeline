import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.prediction import ChurnPredictor

# ----------------------------------------------------

st.set_page_config(
    page_title="AI Customer Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# Load CSS
# ----------------------------------------------------

def load_css():

    with open("assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# ----------------------------------------------------

predictor = ChurnPredictor()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.image(
    "https://img.icons8.com/color/96/artificial-intelligence.png",
    width=80
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "",

    [

        "🏠 Dashboard",

        "📊 Prediction",

        "📈 Analytics",

        "ℹ About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("Model : Logistic Regression")

st.sidebar.metric(

    "Accuracy",

    "80.7 %"

)

st.sidebar.metric(

    "Recall",

    "56.7 %"

)

# ----------------------------------------------------
# DASHBOARD
# ----------------------------------------------------

if page == "🏠 Dashboard":

    st.title("🤖 AI Customer Retention Intelligence")

    st.caption(

        "Predict telecom customer churn using Machine Learning."

    )

    c1,c2,c3,c4 = st.columns(4)

    with c1:

        st.metric(

            "Customers",

            "7043"

        )

    with c2:

        st.metric(

            "Churn Rate",

            "26.5%"

        )

    with c3:

        st.metric(

            "Model Accuracy",

            "80.7%"

        )

    with c4:

        st.metric(

            "Features",

            "30"

        )

    st.markdown("---")

    left,right = st.columns([2,1])

    with left:

        st.subheader("Project Overview")

        st.write("""

This application predicts whether a telecom customer is likely to churn.

The model was trained using Logistic Regression on the IBM Telco Customer Churn Dataset.

The prediction pipeline performs

- Feature Engineering

- One Hot Encoding

- Feature Alignment

- Standard Scaling

- Probability Prediction

before producing the final result.

        """)

    with right:

        fig = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=80.7,

                title={"text":"Model Accuracy"},

                gauge={

                    "axis":{"range":[0,100]},

                    "bar":{"color":"royalblue"}

                }

            )

        )

        fig.update_layout(height=320)

        st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")

    st.subheader("Model Performance")

    metrics = pd.DataFrame(

        {

            "Metric":[

                "Accuracy",

                "Precision",

                "Recall",

                "F1 Score"

            ],

            "Value":[

                80.7,

                65.8,

                56.7,

                60.9

            ]

        }

    )

    fig = px.bar(

        metrics,

        x="Metric",

        y="Value",

        text="Value"

    )

    fig.update_traces(

        textposition="outside"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )