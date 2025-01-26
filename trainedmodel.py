import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Generate example training data

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, Y_train)

# Save the model as a .pkl file
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
