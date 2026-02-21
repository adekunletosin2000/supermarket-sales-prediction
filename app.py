#
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ───────────── Load model with absolute path ─────────────
MODEL_PATH = "supermarket_sales_xgb_full.pkl"   # ← change if your filename is different

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# ───────────── App Header ─────────────
st.title("🛒 Supermarket Sales Predictor")
st.markdown("""
This demo uses an XGBoost model trained on 1,000 transactions to estimate **Total Sales** (incl. 5% tax).
Try different customer profiles, times, and products!
""")

st.divider()

# ───────────── Inputs ─────────────
st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    branch = st.selectbox("Branch", ["A", "B", "C"])
    city = st.selectbox("City", ["Yangon", "Mandalay", "Naypyitaw"])
    customer_type = st.selectbox("Customer type", ["Member", "Normal"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    product_line = st.selectbox("Product line", [
        "Health and beauty", "Electronic accessories", "Home and lifestyle",
        "Sports and travel", "Food and beverages", "Fashion accessories"
    ])
    payment = st.selectbox("Payment", ["Cash", "Credit card", "Ewallet"])

with col2:
    unit_price = st.number_input("Unit price", min_value=10.0, max_value=100.0, value=55.0, step=0.1)
    quantity = st.number_input("Quantity", min_value=1, max_value=10, value=5, step=1)
    rating = st.slider("Rating", 1.0, 10.0, 7.0, step=0.1)
    hour = st.slider("Hour (0-23)", 0, 23, 14)
    day_of_week = st.slider("DayOfWeek (0=Mon, 6=Sun)", 0, 6, 3)
    month = st.slider("Month (1-3)", 1, 3, 2)

# ───────────── Prepare input exactly as model expects ─────────────
expected_columns = [
    'Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment',
    'Unit price', 'Quantity', 'Rating', 'Month', 'DayOfWeek', 'Hour'
]

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

input_data = pd.DataFrame([input_dict])[expected_columns]

# ───────────── Predict ─────────────
if st.button("🔮 Predict Total Sales", type="primary"):
    with st.spinner("Predicting..."):
        try:
            pred_log = model.predict(input_data)
            pred_total = np.expm1(pred_log)[0]

            # Show prediction + range side-by-side
            col_pred, col_range = st.columns([3, 2])
            with col_pred:
                st.success(f"**Predicted Total: ${pred_total:.2f}**")
            with col_range:
                st.info(f"**Rough range**: ${pred_total * 0.8:.2f} – ${pred_total * 1.2:.2f}\n"
                        f"(simple ±20% estimate)")

            st.balloons()

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("This usually means input format doesn't match what the model was trained on.")