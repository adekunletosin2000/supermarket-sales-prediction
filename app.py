import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import shap
from model_utils import get_prediction_and_confidence, get_shap_values
from report_utils import generate_pdf
import os

st.set_page_config(layout="wide")

# ================= LOAD MODEL =================
model = joblib.load("supermarket_sales_xgb_full.pkl")

# ================= NAVIGATION =================
page = st.sidebar.radio("Navigation", [
    "🔮 Predictor",
    "📊 Analytics Dashboard"
])

# ================= INPUT DATA =================
def get_input():
    col1, col2 = st.columns(2)

    with col1:
        branch = st.selectbox("Branch", ["A", "B", "C"])
        city = st.selectbox("City", ["Yangon", "Mandalay", "Naypyitaw"])
        customer_type = st.selectbox("Customer type", ["Member", "Normal"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        product_line = st.selectbox("Product line", [
            "Health and beauty", "Electronic accessories",
            "Home and lifestyle", "Sports and travel",
            "Food and beverages", "Fashion accessories"
        ])
        payment = st.selectbox("Payment", ["Cash", "Credit card", "Ewallet"])

    with col2:
        unit_price = st.number_input("Unit price", 10.0, 100.0, 55.0)
        quantity = st.number_input("Quantity", 1, 10, 5)
        rating = st.slider("Rating", 1.0, 10.0, 7.0)
        hour = st.slider("Hour", 0, 23, 14)
        day_of_week = st.slider("DayOfWeek", 0, 6, 3)
        month = st.slider("Month", 1, 3, 2)

    input_dict = {
        'Branch': branch,
        'City': city,
        'Customer type': customer_type,
        'Gender': gender,
        'Product line': product_line,
        'Payment': payment,
        'Unit price': unit_price,
        'Quantity': quantity,
        'Rating': rating,
        'Month': month,
        'DayOfWeek': day_of_week,
        'Hour': hour
    }

    return pd.DataFrame([input_dict])

# ================= PREDICTOR PAGE =================
if page == "🔮 Predictor":

    st.title("🛒 Supermarket AI Predictor")

    input_data = get_input()

    if st.button("🚀 Predict"):

        prediction, confidence = get_prediction_and_confidence(model, input_data)

        # KPI Cards
        col1, col2 = st.columns(2)

        col1.metric("💰 Predicted Sales", f"${prediction:,.2f}")
        col2.metric("🎯 Model Confidence", f"{confidence}%")

        # Confidence Bar
        st.progress(confidence / 100)

        # SHAP
        st.subheader("🔍 Feature Impact")
        shap_values = get_shap_values(model, input_data)
        st.pyplot(shap.plots.waterfall(shap_values[0], show=False))

        # PDF Report
        filename = "prediction_report.pdf"
        generate_pdf(filename, prediction, confidence)

        with open(filename, "rb") as f:
            st.download_button(
                "📄 Download AI Report",
                f,
                file_name="Supermarket_AI_Report.pdf"
            )

# ================= ANALYTICS DASHBOARD =================
if page == "📊 Analytics Dashboard":

    st.title("📊 AI Analytics Dashboard")

    # Dummy aggregated data
    data = pd.DataFrame({
        "Branch": ["A", "B", "C"],
        "Sales": [42000, 38000, 46000]
    })

    chart = alt.Chart(data).mark_bar().encode(
        x="Branch",
        y="Sales",
        tooltip=["Branch", "Sales"]
    )

    st.altair_chart(chart, use_container_width=True)

    st.metric("📈 Total Revenue", "$126,000")
    st.metric("🏆 Best Branch", "Branch C")