import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

url = "https://raw.githubusercontent.com/aryan331277/Urban-Heat-Index/main/IITH.csv"
data = pd.read_csv(url)

X = data[['Latitude', 'Longitude', 'Population Density', 'Albedo', 
          'Green Cover Percentage', 'Relative Humidity', 'Wind Speed', 
          'Solar Radiation', 'Nighttime Surface Temperature', 'Building Height', 
          'Road Density', 'Proximity to Water Body', 'Urban Vegetation Index', 
          'Heat Stress Index', 'Carbon Emission Levels', 'Distance from Previous Point']]
y = data["Land Surface Temperature"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
