import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Load your dataset
data = pd.read_csv("prediction_log.csv")  # columns: day, gas_left
X = data[["elapsed_days"]]
y = data["predicted_days_remaining"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Test prediction for day 10
future_day = np.array([[10]])
pred = model.predict(future_day)
print(f"Predicted gas left on day 10: {pred[0]:.2f} kg")

# Evaluate accuracy
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"MAE: {mae:.3f}, R²: {r2:.3f}")

# Plot results
plt.scatter(X, y, label="Actual", color="blue")
plt.plot(X, y_pred, color="red", label="Predicted")
plt.xlabel("Day")
plt.ylabel("Gas Left (kg)")
plt.title("Gas Consumption Prediction")
plt.legend()
plt.show()
