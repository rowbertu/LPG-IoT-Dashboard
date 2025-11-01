import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# -------------------- FIREBASE CONNECTION --------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")  # Path to your Firebase key
    firebase_admin.initialize_app(cred)
db = firestore.client()

# -------------------- STREAMLIT SETUP --------------------
st.set_page_config(page_title="🔥 LPG Monitoring Dashboard", layout="wide")
st.title("🔥 LPG Smart Monitoring Dashboard")

st.markdown("""
This dashboard uses **Firestore + Machine Learning** to monitor and forecast LPG gas usage.  
Real-time data is fetched directly from your Firebase collection.
""")

# -------------------- FETCH DATA --------------------
st.markdown("### 📡 Fetching data from Firebase...")
docs = db.collection("gas_consumption_synthetic").order_by("timestamp").stream()
data = [doc.to_dict() for doc in docs]
df = pd.DataFrame(data)

if df.empty:
    st.warning("⚠️ No data found in Firestore yet.")
    st.stop()

# Fix timezone issues
df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce', utc=True)
df["timestamp"] = df["timestamp"].dt.tz_convert(None)
df = df.sort_values("timestamp").reset_index(drop=True)

# Compute gas percentage if missing
if "percentage" not in df.columns:
    df["percentage"] = (df["weight_kg"] / df["weight_kg"].iloc[0]) * 100

# -------------------- DAILY CONSUMPTION --------------------
df["date"] = df["timestamp"].dt.date
daily_usage = df.groupby("date")["gas_consumed"].sum().reset_index()
daily_usage["day_index"] = range(1, len(daily_usage) + 1)

# -------------------- TRAIN PREDICTION MODEL --------------------
X = daily_usage[["day_index"]]
y = daily_usage["gas_consumed"]

model = LinearRegression()
model.fit(X, y)

# -------------------- FORECAST FUTURE --------------------
days_to_predict = st.slider("Select number of days to forecast:", 3, 30, 7)
future_days = np.arange(len(daily_usage) + 1, len(daily_usage) + days_to_predict + 1).reshape(-1, 1)
future_consumption = model.predict(future_days)

# Estimated remaining weight (based on average usage rate)
avg_daily_use = daily_usage["gas_consumed"].mean()
latest_weight = df["weight_kg"].iloc[-1]
estimated_days_remaining = latest_weight / avg_daily_use if avg_daily_use > 0 else np.nan

# Create future depletion DataFrame
future_dates = [df["timestamp"].max() + timedelta(days=i) for i in range(1, days_to_predict + 1)]
predicted_weights = [max(latest_weight - avg_daily_use * i, 0) for i in range(1, days_to_predict + 1)]
predicted_percentage = [max((w / df["weight_kg"].iloc[0]) * 100, 0) for w in predicted_weights]
forecast_df = pd.DataFrame({
    "date": future_dates,
    "predicted_weight_kg": predicted_weights,
    "predicted_percentage": predicted_percentage
})

# -------------------- DISPLAY METRICS --------------------
col1, col2, col3 = st.columns(3)
col1.metric("⛽ Current Gas Weight (kg)", f"{latest_weight:.2f}")
col2.metric("📅 Estimated Days Remaining", f"{estimated_days_remaining:.1f} days")
col3.metric("🔥 Average Daily Consumption", f"{avg_daily_use:.2f} kg/day")

# -------------------- FEATURE 1: ESTIMATED DAYS REMAINING --------------------
st.subheader("1️⃣ Estimated Days Remaining")
st.write(f"Based on your average consumption of **{avg_daily_use:.2f} kg/day**, "
         f"you have approximately **{estimated_days_remaining:.1f} days** of LPG remaining.")

# -------------------- FEATURE 2: PREDICTED GAS PERCENTAGE --------------------
st.subheader("2️⃣ Predicted Gas Percentage Over Time")
st.dataframe(forecast_df[["date", "predicted_percentage"]].style.format({"predicted_percentage": "{:.2f}%"}))

plt.figure(figsize=(10, 5))
plt.plot(forecast_df["date"], forecast_df["predicted_percentage"], color="orange", marker="o")
plt.title("Predicted LPG Percentage Over Time")
plt.xlabel("Date")
plt.ylabel("Predicted Percentage (%)")
plt.grid(True)
st.pyplot(plt)

# -------------------- FEATURE 3: DEPLETION FORECAST --------------------
st.subheader("3️⃣ Depletion Forecast (Weight Over Time)")

plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["weight_kg"], label="Actual Weight", marker='o', color="blue")
plt.plot(forecast_df["date"], forecast_df["predicted_weight_kg"], label="Forecast", linestyle="--", color="red")
plt.title("Gas Depletion Forecast")
plt.xlabel("Date")
plt.ylabel("Gas Weight (kg)")
plt.legend()
plt.grid(True)
st.pyplot(plt)

# -------------------- FEATURE 4: DAILY USAGE PATTERN --------------------
st.subheader("4️⃣ Daily Usage Pattern")
plt.figure(figsize=(10, 4))
plt.bar(daily_usage["date"], daily_usage["gas_consumed"], color="green")
plt.title("Daily Gas Usage Pattern")
plt.xlabel("Date")
plt.ylabel("Gas Consumed (kg)")
plt.xticks(rotation=45)
plt.grid(axis="y")
st.pyplot(plt)

# -------------------- LAST UPDATED --------------------
st.caption(f"Last updated: {df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}")
