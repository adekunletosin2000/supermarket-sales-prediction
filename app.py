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
import datetime
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
        .stButton>button {background-color:#4CAF50;color:white;border-radius:8px;}
        .stMetric>div {font-size:22px;}
        .block-container {padding-top: 2rem;}
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

# ================= LOAD DATASET =================
data_path = BASE_DIR / "supermarket_sales.csv"
df = pd.read_csv(data_path)

# --- Ensure Month and Hour exist (like in training) ---
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
else:
    df["Month"] = 1

if "Time" in df.columns:
    df["Hour"] = pd.to_datetime(df["Time"]).dt.hour
else:
    df["Hour"] = 12

# ================= SIDEBAR NAV =================
page = st.sidebar.radio("Navigation", ["🔮 Predictor", "📊 Analytics Dashboard"])

# ================= INPUT FUNCTION =================
def get_input():
    col1, col2 = st.columns(2)

    with col1:
        branch = st.selectbox("Branch", df["Branch"].unique())
        city = st.selectbox("City", df["City"].unique())
        customer_type = st.selectbox("Customer type", df["Customer type"].unique())
        gender = st.selectbox("Gender", df["Gender"].unique())
        product_line = st.selectbox("Product line", df["Product line"].unique())
        payment = st.selectbox("Payment", df["Payment"].unique())

    with col2:
        unit_price = st.number_input("Unit price", float(df["Unit price"].min()), float(df["Unit price"].max()), 50.0)
        quantity = st.number_input("Quantity", 1, 20, 5)
        rating = st.slider("Rating", 1.0, 10.0, 7.0)

        # Auto-fill current hour and month
        now = datetime.datetime.now()
        hour = st.slider("Hour", 0, 23, now.hour)
        day_of_week = st.slider("DayOfWeek", 0, 6, now.weekday())
        month = st.slider("Month", 1, 12, now.month)

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

    df_input = pd.DataFrame([input_dict])
    categorical_cols = ["Branch", "City", "Customer type", "Gender", "Product line", "Payment"]
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols)

    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[feature_columns]
    return df_encoded

# ==========================================================
# 🔮 PREDICTOR PAGE
# ==========================================================
if page == "🔮 Predictor":
    st.title("🛒 Supermarket AI Sales Predictor")
    input_data = get_input()
    if st.button("🚀 Predict Sales"):
        # Only get prediction, ignore confidence
        prediction, _ = get_prediction_and_confidence(model, input_data)
        # Show only predicted sales
        st.metric("💰 Predicted Sales", f"${prediction:,.2f}")
        # ---------- SHAP ----------
        st.subheader("🔍 Feature Contribution Analysis")
        shap_values = get_shap_values(model, input_data)
        fig = plt.figure(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        plt.close(fig)
        # ---------- PDF ----------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_path = tmp_file.name
            generate_pdf(tmp_path, prediction)  # pass only prediction
            with open(tmp_path, "rb") as f:
                st.download_button(
                    "📄 Download AI Report",
                    f,
                    file_name="Supermarket_AI_Report.pdf"
                )
        os.remove(tmp_path)

# ==========================================================
# 📊 ANALYTICS DASHBOARD
# ==========================================================
if page == "📊 Analytics Dashboard":

    st.title("📊 AI Business Intelligence Dashboard")

    # --- KPIs ---
    total_revenue = df["Sales"].sum()
    avg_transaction = df["Sales"].mean()
    best_branch = df.groupby("Branch")["Sales"].sum().idxmax()

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("💳 Avg Transaction", f"${avg_transaction:,.2f}")
    col3.metric("🏆 Best Branch", best_branch)

    st.divider()

    # --- Sales by Branch ---
    branch_sales = df.groupby("Branch")["Sales"].sum().reset_index()
    branch_chart = alt.Chart(branch_sales).mark_bar().encode(
        x="Branch",
        y=alt.Y("Sales", title="Sales ($)"),
        tooltip=[alt.Tooltip("Branch"), alt.Tooltip("Sales", format="$,.2f")]
    )
    st.subheader("Sales by Branch")
    st.altair_chart(branch_chart, use_container_width=True)

    # --- Sales by Product Line ---
    product_sales = df.groupby("Product line")["Sales"].sum().reset_index()
    product_chart = alt.Chart(product_sales).mark_bar().encode(
        x=alt.X("Product line", sort="-y"),
        y=alt.Y("Sales", title="Sales ($)"),
        tooltip=[alt.Tooltip("Product line"), alt.Tooltip("Sales", format="$,.2f")]
    )
    st.subheader("Sales by Product Line")
    st.altair_chart(product_chart, use_container_width=True)

    # --- Monthly Sales Trend ---
    monthly_sales = df.groupby("Month")["Sales"].sum().reset_index()
    monthly_chart = alt.Chart(monthly_sales).mark_line(point=True).encode(
        x=alt.X("Month", title="Month"),
        y=alt.Y("Sales", title="Sales ($)"),
        tooltip=[alt.Tooltip("Month"), alt.Tooltip("Sales", format="$,.2f")]
    )
    st.subheader("Monthly Sales Trend")
    st.altair_chart(monthly_chart, use_container_width=True)

    # --- Hourly Sales Trend ---
    hourly_sales = df.groupby("Hour")["Sales"].sum().reset_index()
    hourly_chart = alt.Chart(hourly_sales).mark_line(point=True).encode(
        x=alt.X("Hour", title="Hour of Day"),
        y=alt.Y("Sales", title="Sales ($)"),
        tooltip=[alt.Tooltip("Hour"), alt.Tooltip("Sales", format="$,.2f")]
    )
    st.subheader("Hourly Sales Trend")
    st.altair_chart(hourly_chart, use_container_width=True)