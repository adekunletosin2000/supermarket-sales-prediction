import pandas as pd
import json
import xgboost as xgb
from pathlib import Path

BASE_DIR = Path(__file__).parent

# --- Load trained model ---
model_path = BASE_DIR / "supermarket_sales_model.json"
feature_columns_path = BASE_DIR / "feature_columns.json"

with open(feature_columns_path, "r") as f:
    feature_columns = json.load(f)

model = xgb.XGBRegressor()
model.load_model(model_path)

# --- Load new data ---
new_data = pd.read_csv(BASE_DIR / "new_sales_data.csv")  # replace with your CSV

# --- Feature engineering ---
if "Time" in new_data.columns:
    new_data["Hour"] = pd.to_datetime(new_data["Time"]).dt.hour

# --- One-hot encode categorical columns ---
categorical_cols = ["Branch", "City", "Customer type", "Gender", "Product line", "Payment"]
df_encoded = pd.get_dummies(new_data, columns=categorical_cols)

# --- Add missing columns ---
for col in feature_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# --- Reorder columns ---
X_new = df_encoded[feature_columns]

# --- Make predictions ---
predictions = model.predict(X_new)
new_data["Predicted_Total"] = predictions

# --- Output ---
print(new_data)
new_data.to_csv(BASE_DIR / "predicted_sales.csv", index=False)
print("✅ Predictions saved to predicted_sales.csv")