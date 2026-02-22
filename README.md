<div align="center">

# 🛒 Supermarket Sales Prediction & AI-Powered Insights

**End-to-end ML regression project**  
Predicting total transaction value (including 5% tax) using customer behavior, product details, time-based features and more.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

</div>

<br>

## 🎯 Project Purpose

This project demonstrates a complete **machine learning workflow** — from exploratory data analysis and feature engineering to model training, interpretability analysis (SHAP), and production-style interactive deployment.

It uses a real-world **supermarket sales dataset** (~1,000 transactions) to predict the **total sales amount** (including tax) per transaction — a common retail forecasting task.

Two modeling approaches are compared:

- **Realistic scenario** — using only behavioral, temporal and categorical features (no direct leakage like quantity × unit price)
- **Cheat-mode / baseline** — including quantity and unit price (shows near-perfect fit due to obvious leakage)

The project highlights an important lesson in retail ML: **without basket composition or historical customer data, per-transaction sales are very hard to predict accurately**.

<br>

## ✨ Live Interactive Demo

**Predict sales in real time • See SHAP explanations • Download PDF report**

→ **[Open Supermarket Sales AI Predictor](https://your-username-supermarket-sales-prediction.streamlit.app)**  
(Hosted on Streamlit Community Cloud – no login required)

https://github.com/user-attachments/assets/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  <!-- replace with short video/GIF if you record one -->

<br>

## 📊 Key Findings

| Scenario                        | RMSE (USD) | R²      | Interpretation                                      |
|----------------------------------|------------|---------|-----------------------------------------------------|
| XGBoost – full features (leakage) | **6.16**   | **0.9994** | Almost perfect — model learns the exact formula     |
| XGBoost – realistic features only | **~297**   | **-0.35** | Worse than predicting the mean → very weak signal   |
| Random Forest – full features     | 8.53       | 0.9889  | Strong but still leakage-dependent                  |

**Main business insight**  
Most of the predictive power comes from **quantity × unit price** (obvious mathematical dependency).  
Behavioral features (customer type, gender, time of day, rating, product line, etc.) provide **very limited signal** for predicting exact basket value in this dataset.

<br>

## 🛠️ Tech Stack & Tools

| Category               | Tools / Libraries                                 |
|------------------------|---------------------------------------------------|
| Data Processing        | pandas, numpy                                     |
| Machine Learning       | scikit-learn, **XGBoost**, SHAP                   |
| Visualization          | matplotlib, seaborn, **Altair** (interactive)     |
| Web Application        | **Streamlit** (v1.42+)                            |
| PDF Report Generation  | reportlab                                         |
| Environment / Deployment | venv, Git, Streamlit Community Cloud, GitHub      |

<br>

## 📁 Project Structure

```text
supermarket-sales-prediction/
├── app.py                    # Main Streamlit application
├── model_utils.py            # Model loading, prediction & SHAP logic
├── report_utils.py           # PDF report generation
├── train_model.py            # Model training & evaluation script
├── supermarket_sales.csv      # Original dataset
├── supermarket_sales_model.json   # Trained XGBoost model
├── feature_columns.json      # Ordered list of model input features
├── requirements.txt          # Dependencies
├── screenshots/              # Dashboard & EDA images
└── README.md



🚀 Quick Start (Local)
Bash# 1. Clone repository
git clone https://github.com/your-username/supermarket-sales-prediction.git
cd supermarket-sales-prediction

# 2. Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# or
.venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py



📸 Screenshots
Interactive Predictor + SHAP Waterfall
<img src="screenshots/predictor_shap.png" alt="Predictor + SHAP">
Analytics Dashboard
<img src="screenshots/dashboard.png" alt="Dashboard Overview">
PDF Report Example
<img src="screenshots/pdf_report_example.png" alt="PDF Report">
(Add 3–5 high-quality screenshots that show the most impressive parts of your app)


🔍 What I Learned / Focus Areas

Importance of feature leakage detection in retail forecasting
Using SHAP for model interpretability in production-facing apps
Building clean, professional Streamlit dashboards with Altair charts & PDF export
Writing modular ML code (separate model utils, report generation, app logic)
Deploying ML demos reliably on Streamlit Community Cloud



📄 License
MIT License — feel free to use any part of the code for learning, portfolios, or personal projects.



Made with ❤️ in Makurdi • 2026

```