# gas_ml_dashboard.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="LPG Gas Leak ML Dashboard", page_icon="🔥", layout="wide")

# --- HEADER ---
st.title("🔥 LPG Gas Leak Detection with Machine Learning")
st.write("This dashboard uses a **Random Forest** model to predict leak risk levels (Level 1–3) based on gas concentration and cylinder weight.")

# --- Step 1: Simulate Data ---
np.random.seed(42)
gas_ppm = np.random.uniform(100, 900, 200)
weight_kg = np.random.uniform(9.0, 13.0, 200)

labels = []
for g in gas_ppm:
    if g < 300:
        labels.append(1)
    elif 300 <= g < 600:
        labels.append(2)
    else:
        labels.append(3)

data = pd.DataFrame({
    "gas_ppm": gas_ppm,
    "weight_kg": weight_kg,
    "label": labels
})

# --- Step 2: Train Random Forest ---
X = data[["gas_ppm", "weight_kg"]]
y = data["label"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Step 3: User Inputs ---
st.subheader("🔧 Adjust Sensor Values")
gas_input = st.slider("Gas Concentration (ppm)", 100, 900, 450)
weight_input = st.slider("Cylinder Weight (kg)", 9.0, 13.0, 11.5, step=0.1)

# --- Step 4: Predict ---
sample = np.array([[gas_input, weight_input]])
predicted_level = model.predict(sample)[0]

# --- Step 5: Display Prediction ---
if predicted_level == 1:
    st.success(f"🟢 Level 1: SAFE\n\nGas: {gas_input} ppm | Weight: {weight_input:.2f} kg")
elif predicted_level == 2:
    st.warning(f"🟡 Level 2: WARNING\n\nGas: {gas_input} ppm | Weight: {weight_input:.2f} kg")
else:
    st.error(f"🔴 Level 3: CRITICAL LEAK DETECTED\n\nGas: {gas_input} ppm | Weight: {weight_input:.2f} kg")

# --- Step 6: Visualization (Optional) ---
show_plot = st.checkbox("Show Training Data Visualization")

if show_plot:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {1: "green", 2: "yellow", 3: "red"}
    for level, color in colors.items():
        subset = data[data["label"] == level]
        ax.scatter(subset["gas_ppm"], subset["weight_kg"], color=color, label=f"Level {level}", alpha=0.6, edgecolor="k")

    ax.scatter(gas_input, weight_input, color="blue", s=200, marker="*", label=f"Your Input → Level {predicted_level}")
    ax.set_xlabel("Gas Concentration (ppm)")
    ax.set_ylabel("Cylinder Weight (kg)")
    ax.set_title("Training Data & Your Input")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

st.caption("GaSolve: Developement of LPG Gas Leak Detection & Auto Shutoff Project 🧯")
