import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
url = "https://raw.githubusercontent.com/aryan331277/Streamlit/main/IITH.csv"
data = pd.read_csv(url)

# Prepare features and target
X = data.drop(columns=["Land Surface Temperature"])
y = data["Land Surface Temperature"]

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
