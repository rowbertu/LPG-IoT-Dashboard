import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- Initialize Firestore ---
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Fetch Firestore Data ---
docs = db.collection("LPG_Usage").stream()
data = [doc.to_dict() for doc in docs]
df = pd.DataFrame(data)

# --- Ensure proper sorting and formatting ---
df = df.sort_values("timestamp")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["day_index"] = range(1, len(df) + 1)

print("\n📊 Data loaded from Firestore:")
print(df[["timestamp", "weight_kg", "gas_consumed"]])

# --- Train Simple Linear Regression Model ---
# We’ll predict future weight_kg based on time (day_index)
X = df[["day_index"]]
y = df["weight_kg"]

model = LinearRegression()
model.fit(X, y)

# --- Predict gas level for the next day ---
next_day = [[df["day_index"].max() + 1]]
predicted_weight = model.predict(next_day)[0]

print(f"\n🔮 Predicted LPG weight for next day: {predicted_weight:.2f} kg")

# --- Visualization ---
plt.figure(figsize=(8,4))
plt.plot(df["timestamp"], df["weight_kg"], marker='o', label="Actual Weight")
plt.plot(df["timestamp"].tolist() + [df["timestamp"].max() + timedelta(days=1)],
         list(df["weight_kg"]) + [predicted_weight],
         linestyle='--', color='red', label="Predicted Next Day")
plt.title("LPG Weight Prediction Over Time")
plt.xlabel("Date")
plt.ylabel("Weight (kg)")
plt.legend()
plt.grid(True)
plt.show()
