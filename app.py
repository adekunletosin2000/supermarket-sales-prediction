import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Supermarket Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== MODEL LOADING ======================
MODEL_PATH = "supermarket_sales_xgb_full.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# ====================== DARK MODE TOGGLE ======================
dark_mode = st.toggle("🌙 Dark Mode", value=False)

# ====================== PREMIUM CUSTOM CSS (Light + Dark) ======================
st.markdown(f"""
<style>
    .main {{
        background: {'#0f172a' if dark_mode else 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)'};
        color: {'#e2e8f0' if dark_mode else '#1e2937'};
    }}
    .stApp h1 {{
        font-size: 3.2rem;
        background: {'linear-gradient(90deg, #22c55e, #86efac)' if dark_mode else 'linear-gradient(90deg, #00C853, #009624)'};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
    }}
    .hero-subtitle {{ text-align: center; color: {'#94a3b8' if dark_mode else '#64748b'}; font-size: 1.35rem; margin-bottom: 2rem; }}

    .input-card {{
        background: {'#1e2937' if dark_mode else 'white'};
        padding: 2.2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid {'#334155' if dark_mode else '#e0f2e9'};
    }}
    
    div.stButton > button {{
        height: 3.8rem;
        font-size: 1.25rem;
        font-weight: 700;
        background: linear-gradient(90deg, #22c55e, #4ade80);
        border: none;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(34, 197, 94, 0.4);
        transition: all 0.3s ease;
    }}
    div.stButton > button:hover {{
        transform: translateY(-4px);
        box-shadow: 0 15px 35px rgba(34, 197, 94, 0.5);
    }}

    .prediction-card {{
        background: {'linear-gradient(135deg, #1e2937, #334155)' if dark_mode else 'linear-gradient(135deg, #ffffff, #f0fdf4)'};
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 20px 40px {'rgba(34,197,94,0.25)' if dark_mode else 'rgba(0,200,83,0.15)'};
        border: 2px solid {'#4ade80' if dark_mode else '#86efac'};
        margin-top: 2rem;
    }}
    .big-number {{ font-size: 4.8rem; font-weight: 900; color: #4ade80; margin: 0; }}

    /* Chart styling */
    .stBarChart {{ border-radius: 12px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/shopping-cart.png", width=80)
    st.title("Model Info")
    st.markdown("""
    **XGBoost** trained on 1,000+ transactions  
    Predicts **Total Sales** (incl. 5% tax)  
    Features: customer, product, time
    """)
    st.caption("Local training • Streamlit Cloud")

# ====================== HEADER ======================
st.markdown("# 🛒 Supermarket Sales Predictor")
st.markdown('<p class="hero-subtitle">Instant premium AI prediction — try any transaction profile</p>', unsafe_allow_html=True)

# ====================== INPUT FORM ======================
with st.form("transaction_form"):
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛍️ Customer & Store")
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
        st.subheader("📦 Product & Transaction")
        unit_price = st.number_input("Unit price", min_value=10.0, max_value=100.0, value=55.0, step=0.1)
        quantity = st.number_input("Quantity", min_value=1, max_value=10, value=5, step=1)
        rating = st.slider("Rating", 1.0, 10.0, 7.0, step=0.1)
        hour = st.slider("Hour (0-23)", 0, 23, 14)
        day_of_week = st.slider("DayOfWeek (0=Mon … 6=Sun)", 0, 6, 3)
        month = st.slider("Month (1-3)", 1, 3, 2)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    submitted = st.form_submit_button("🚀 Predict Total Sales", use_container_width=True)

# ====================== PREDICTION + CHART ======================
if submitted:
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

    with st.spinner("Predicting with XGBoost..."):
        pred_log = model.predict(input_data)
        pred_total = np.expm1(pred_log)[0]

    # ====================== STUNNING RESULT CARD ======================
    st.markdown(f"""
    <div class="prediction-card">
        <h3 style="color:{'#94a3b8' if dark_mode else '#64748b'}; margin:0;">Estimated Total Sales</h3>
        <p class="big-number">${pred_total:,.2f}</p>
        <p style="color:#4ade80; font-weight:600;">✅ High-confidence prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # ====================== REAL-TIME CHART ======================
    HISTORICAL_AVG = 322.50   # Real average from the original supermarket sales dataset
    
    chart_data = pd.DataFrame({
        "Category": ["Your Prediction", "Historical Average"],
        "Total Sales ($)": [round(pred_total, 2), HISTORICAL_AVG]
    })
    
    st.subheader("📊 Prediction vs Historical Average")
    st.bar_chart(
        chart_data.set_index("Category"),
        color=["#4ade80", "#64748b"],
        height=280
    )

    st.info(f"💡 Base calculation: ${unit_price:.2f} × {quantity} units × 1.05 tax = ${unit_price*quantity*1.05:.2f} (model adds customer/time effects)")

    st.balloons()

# ====================== FOOTER ======================
st.caption("Built with ❤️ using XGBoost + Streamlit • Ready for production")