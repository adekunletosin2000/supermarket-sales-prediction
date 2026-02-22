<div align="center">

# 🛒 Supermarket Sales Predictor

**AI-powered tool to forecast transaction totals and uncover sales drivers**  
Helping supermarkets optimize inventory, staffing, and strategies by predicting sales per transaction (incl. 5% tax) based on customer, product, and time factors.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

</div>

## The Challenge We're Solving

Supermarkets deal with unpredictable sales—fluctuating by time of day, customer type, product category, and more. Without reliable forecasts for each transaction's total (including tax), it's tough to stock shelves right, schedule staff efficiently, or spot growth opportunities.  

This project tackles that head-on using a Kaggle dataset of ~1,000 real transactions. We analyze patterns, build a robust ML model, and deliver practical tools to predict sales and extract insights—directly supporting better inventory management, peak-hour staffing, and targeted promotions.

## How We Solved It

1. **Data Dive & Pattern Spotting**:  
   Parsed the dataset to pull out key drivers like peak hours (e.g., evenings spike sales), high-value product lines (e.g., Electronics lead), and customer behaviors (Members spend more). Added engineered features: Month, DayOfWeek, Hour for time-based trends.

2. **ML Model Building**:  
   Trained XGBoost regressor (outperformed Random Forest) on encoded features. Handles categoricals (Branch, City, etc.) and numerics (Quantity, Unit Price, Rating).  
   - With all features: Near-perfect accuracy (RMSE ~$6, R² 0.999)—but watch for leakage from Quantity/Unit Price.  
   - Realistic mode (no leakage): Highlights need for richer data, but still flags trends like time/product impacts.

3. **Evaluation & Selection**:  
   Measured with RMSE and R² on holdout set. XGBoost wins for speed and precision. Integrated SHAP for explainability—shows how each input boosts or drags the prediction.

4. **Actionable Insights**:  
   - **Inventory**: Stock more in high-margin lines like Food & Beverages during busy months (e.g., March peaks).  
   - **Staffing**: Ramp up during 7-8 PM rushes—hourly trends show 20-30% higher sales.  
   - **Strategy**: Target Member promotions; they drive bigger baskets. Avoid over-relying on ratings—they barely move the needle.

## Killer Features in the App

- **Predictor Tool**: Input customer/product details, get instant sales forecast + SHAP breakdown (visualizes feature impacts). Auto-fills current time for real-world use.  
- **Analytics Dashboard**: Interactive charts on sales by branch/product/month/hour—spot trends at a glance with Altair visuals.  
- **PDF Reports**: One-click download of predictions with key details—handy for team shares or audits.  
- **Extras Added**: Temp file handling for secure PDFs, custom styling for pro look, and wide layout for better UX.

## Live Demo

Jump in and test it:  
→ **[Supermarket Sales AI Predictor on Streamlit](https://your-username-supermarket-sales-prediction.streamlit.app)**  
(Hosted on Streamlit Cloud—always available, no setup needed.)

## Tech Setup

- **Core**: Pandas/Numpy for data, XGBoost for modeling, SHAP for insights.  
- **Viz**: Matplotlib/Altair for charts, ReportLab for PDFs.  
- **App**: Streamlit for the interactive frontend—deployed in minutes.  
- **Files**: Model saved as JSON, features listed for easy reuse.

## Quick Run Locally

```bash
git clone https://github.com/your-username/supermarket-sales-prediction.git
cd supermarket-sales-prediction
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py




Why This Matters for Business
This isn't just a model—it's a ready-to-use system that turns raw sales data into decisions. Predict highs/lows to cut waste, align staff with demand, and boost revenue through smarter strategies. Scalable to bigger datasets for even sharper forecasts.
License
MIT—use, tweak, or build on it freely. See LICENSE for details.

Built in Makurdi, 2026 | Let's connect on X or LinkedIn!