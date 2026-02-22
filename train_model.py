import pandas as pd
import numpy as np
import json
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==============================
# 1️⃣ LOAD DATA
# ==============================
df = pd.read_csv("supermarket_sales.csv")

# Drop irrelevant columns if they exist
drop_cols = ["Invoice ID"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# ==============================
# 2️⃣ TARGET & FEATURES
# ==============================
target = "Sales"
y = df[target]
X = df.drop(columns=[target])

# ==============================
# 3️⃣ HANDLE DATE & TIME
# ==============================
if "Date" in X.columns:
    X["Date"] = pd.to_datetime(X["Date"])
    X["Day"] = X["Date"].dt.day
    X["Month"] = X["Date"].dt.month
    X = X.drop(columns=["Date"])

if "Time" in X.columns:
    X["Hour"] = pd.to_datetime(X["Time"]).dt.hour
    X = X.drop(columns=["Time"])

# ==============================
# 4️⃣ ENCODE CATEGORICALS
# ==============================
X = pd.get_dummies(X)

# Save feature names (VERY IMPORTANT)
feature_columns = X.columns.tolist()
with open("feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

# ==============================
# 5️⃣ TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6️⃣ TRAIN MODEL
# ==============================
model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 7️⃣ EVALUATION
# ==============================
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("Model Performance")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 4))

# ==============================
# 8️⃣ SAVE MODEL (JSON SAFE)
# ==============================
model.save_model("supermarket_sales_model.json")

print("✅ Model saved as supermarket_sales_model.json")
print("✅ Feature columns saved as feature_columns.json")