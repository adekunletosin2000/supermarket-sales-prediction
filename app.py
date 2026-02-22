import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import altair as alt
import json
import shap
from pathlib import Path
import matplotlib.pyplot as plt
import os
import tempfile
from model_utils import get_prediction_and_confidence, get_shap_values
from report_utils import generate_pdf

# ================= CONFIG =================
st.set_page_config(layout="wide")
BASE_DIR = Path(__file__).parent

# ================= STYLING =================
def load_styles():
    st.markdown(
        """
        <style>
        .stButton>button {background-color:#4CAF50;color:white;}
        .stMetric>div {font-size:22px;}
        </style>
        """,
        unsafe_allow_html=True
    )
load_styles()

# ================= LOAD MODEL =================
model_path = BASE_DIR / "supermarket_sales_model.json"
model = xgb.XGBRegressor()
model.load_model(model_path)

feature_columns_path = BASE_DIR / "feature_columns.json"
with open(feature_columns_path, "r") as f:
    feature_columns = json.load(f)

# ================= NAVIGATION =================
page = st.sidebar.radio("Navigation", ["🔮 Predictor", "📊 Analytics Dashboard"])

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
        month = st.slider("Month", 1, 12, 2)

    # Step 1: raw input dictionary
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

    df = pd.DataFrame([input_dict])

    # Step 2: one-hot encode categorical columns
    categorical_cols = ["Branch", "City", "Customer type", "Gender", "Product line", "Payment"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Step 3: Add missing columns
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Step 4: Reorder columns to match training
    df_encoded = df_encoded[feature_columns]

    return df_encoded

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

        # ================= SHAP =================
        st.subheader("🔍 Feature Impact")
        shap_values = get_shap_values(model, input_data)

        # Single-row input: waterfall plot; multi-row: summary
        if len(input_data) == 1:
            shap_single = shap_values[0]
            fig = plt.figure(figsize=(10, 4))
            shap.plots.waterfall(shap_single, show=False)
            st.pyplot(fig)
            plt.close(fig)
        else:
            fig = plt.figure(figsize=(10, 4))
            shap.summary_plot(shap_values.values, input_data, show=False)
            st.pyplot(fig)
            plt.close(fig)

        # ================= PDF REPORT (TEMP FILE) =================
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_path = tmp_file.name
            generate_pdf(tmp_path, prediction, confidence)

            with open(tmp_path, "rb") as f:
                st.download_button(
                    "📄 Download AI Report",
                    f,
                    file_name="Supermarket_AI_Report.pdf"
                )

        os.remove(tmp_path)

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

    col1, col2 = st.columns(2)
    col1.metric("📈 Total Revenue", "$126,000")
    col2.metric("🏆 Best Branch", "Branch C")