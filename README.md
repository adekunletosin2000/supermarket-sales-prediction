<div align="center">

# 🛒 Supermarket Sales Prediction

**XGBoost-powered regression model + interactive Streamlit demo**  
Predicting total transaction value (incl. 5% tax) from customer profile, product, time, and more.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📊 Project Overview

Built an end-to-end machine learning pipeline to forecast **total sales per transaction** using a real-world supermarket dataset (1,000 records, Kaggle).

Key goals:
- Understand what drives basket size (customer loyalty, time of day, product category, etc.)
- Compare model performance with and without direct leakage features (quantity & unit price)
- Deliver a production-style **interactive prediction demo** via Streamlit

## 🎯 Key Results

| Model              | Features included              | RMSE (USD) | R²       | Notes                              |
|---------------------|--------------------------------|------------|----------|------------------------------------|
| XGBoost (best)      | All (incl. qty & unit price)   | **6.16**   | **0.9994** | Near-perfect – learns exact formula |
| Random Forest       | All                            | 8.53       | 0.9889   | Very strong                        |
| XGBoost (realistic) | Without qty & unit price       | 296.71     | -0.3532  | Worse than mean predictor → expected in retail |

**Business takeaway**:  
Quantity & unit price dominate (obvious leakage).  
Behavioral + temporal features alone give very limited signal — real retail forecasting needs richer basket-level data.

## 🛠️ Tech Stack

- **Data & Modeling**: pandas, scikit-learn, XGBoost  
- **Visualization**: matplotlib, seaborn  
- **Deployment**: Streamlit (interactive web app)  
- **Environment**: Google Colab + ngrok (dev), Streamlit Cloud (production-ready)

## 📈 Visual Highlights

### Distribution of Total Sales
![Distribution of Total Sales](screenshots/distribution_total_sales.png)

### Average Spend by Product Line
![Average by Product Line](screenshots/avg_by_product_line.png)

### Customer Type & Gender
![Customer Type & Gender](screenshots/customer_type_gender.png)

### Time Patterns (Hour of Day)
![Hour of Day](screenshots/hour_of_day.png)

*(More screenshots coming soon — feel free to add yours!)*

## 🚀 Live Demo

**Try the predictor yourself!**  
→ [Live Streamlit App](https://your-username-supermarket-sales-prediction.streamlit.app)  
(Deployed on Streamlit Community Cloud – always on)

## 🗂️ Project Structure
