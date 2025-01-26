
import streamlit as st
import joblib
import pandas as pd
from PIL import Image  # For loading XAI image

# Load your model
model = joblib.load("model.pkl")  # Replace with your .pkl file path

# Load XAI image
xai_image = Image.open("feature importance of rf regressor.png")  # Replace with your XAI image path

# Streamlit app title
st.title("Urban Heat Island Effect Predictor 🌆")
st.write("Predict land surface temperature and get actionable suggestions to mitigate the urban heat island effect.")

# Sidebar for user input
st.sidebar.header("Input Features")

# Example input fields (customize based on your dataset)
latitude = st.sidebar.number_input("Latitude", value=19.0760)
longitude = st.sidebar.number_input("Longitude", value=72.8777)
green_cover = st.sidebar.number_input("Green Cover Percentage (%)", value=15)
building_height = st.sidebar.number_input("Building Height (m)", value=30)
population_density = st.sidebar.number_input("Population Density (people/km²)", value=20000)
albedo = st.sidebar.number_input("Albedo", value=0.3)

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    "Latitude": [latitude],
    "Longitude": [longitude],
    "Green Cover Percentage (%)": [green_cover],
    "Building Height (m)": [building_height],
    "Population Density (people/km²)": [population_density],
    "Albedo": [albedo]
})

# Predict temperature
if st.sidebar.button("Predict Temperature"):
    # Make prediction
    prediction = model.predict(input_data)
    st.write(f"### Predicted Land Surface Temperature: **{prediction[0]:.2f} °C**")

    # Show XAI image
    st.write("### Feature Importance")
    st.image(xai_image, caption="Factors affecting temperature prediction", use_column_width=True)

    # Generate suggestions based on rules
    st.write("### Recommendations")
    suggestions = []

    if green_cover < 25:
        suggestions.append("🌿 **Increase green cover** to at least 25% to reduce temperature.")
    if building_height > 20:
        suggestions.append("🏢 **Reduce building height** to improve airflow and reduce heat retention.")
    if albedo < 0.4:
        suggestions.append("🏗️ **Use reflective materials** for roofs and pavements to increase albedo and lower surface temperature.")
    if population_density > 15000:
        suggestions.append("👥 **Optimize urban planning** to reduce population density in heat-prone areas.")

    if suggestions:
        for suggestion in suggestions:
            st.write(f"- {suggestion}")
    else:
        st.write("✅ No major changes needed. The current urban features are well-balanced to mitigate the urban heat island effect.")

# Footer
st.write("---")
st.write("Built with ❤️ by [Your Name]")
