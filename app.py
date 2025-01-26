import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# --------------------------
# Configuration
# --------------------------
MODEL_PATH = "model.pkl"
XAI_IMAGE_PATH = "feature importance of rf regressor.png"
SURFACE_MATERIALS = ["Concrete", "Asphalt", "Grass", "Water", "Mixed"]  # Must match training categories

# --------------------------
# Load Resources
# --------------------------
try:
    model = joblib.load(MODEL_PATH)
    xai_image = Image.open(XAI_IMAGE_PATH)
    feature_names = model.feature_names_in_
except Exception as e:
    st.error(f"Error loading resources: {str(e)}")
    st.stop()

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Urban Analytics", layout="wide")
st.title("🏙️ Surface Material-Compatible Heat Predictor")

# --------------------------
# Input Handling
# --------------------------
with st.sidebar:
    st.header("Urban Parameters")
    
    # Collect all features EXCEPT Surface Material first
    inputs = {
        'Latitude': st.number_input("Latitude", 19.0, 19.2, 19.0760, 0.0001),
        'Longitude': st.number_input("Longitude", 72.8, 73.0, 72.8777, 0.0001),
        'Population Density': st.number_input("Population Density (people/km²)", 1000, 50000, 20000),
        'Albedo': st.slider("Albedo", 0.0, 1.0, 0.3, 0.05),
        'Green Cover Percentage': st.slider("Green Cover (%)", 0, 100, 25),
        'Relative Humidity': st.slider("Humidity (%)", 0, 100, 60),
        'Wind Speed': st.slider("Wind Speed (m/s)", 0.0, 15.0, 3.0, 0.1),
        'Building Height': st.slider("Building Height (m)", 5, 150, 30),
        'Road Density': st.slider("Road Density (km/km²)", 0.0, 20.0, 5.0, 0.1),
        'Proximity to Water Body': st.slider("Water Proximity (m)", 0, 5000, 1000),
        'Solar Radiation': st.slider("Solar Radiation (W/m²)", 0, 1000, 500),
        'Nighttime Surface Temperature': st.slider("Night Temp (°C)", 15.0, 40.0, 25.0, 0.1),
        'Distance from Previous Point': st.number_input("Distance from Previous Point (m)", 0, 5000, 100),
        'Heat Stress Index': st.slider("Heat Stress Index", 0.0, 10.0, 3.5, 0.1),
        'Urban Vegetation Index': st.slider("Vegetation Index", 0.0, 1.0, 0.5, 0.01),
        'Carbon Emission Levels': st.number_input("CO₂ Levels (ppm)", 300, 1000, 400)
    }
    
    # Handle Surface Material with one-hot encoding
    selected_material = st.selectbox("Surface Material", SURFACE_MATERIALS)
    for material in SURFACE_MATERIALS:
        inputs[f"Surface Material_{material}"] = 1 if material == selected_material else 0

# --------------------------
# Prediction System
# --------------------------
if st.sidebar.button("Predict Temperature"):
    try:
        # Create DataFrame with proper feature order
        input_df = pd.DataFrame([inputs], columns=feature_names)
        
        # Debug: Show feature alignment
        st.write("### Feature Verification")
        st.write("Expected features:", feature_names)
        st.write("Provided features:", input_df.columns.tolist())
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction Result")
            st.metric("Surface Temperature", f"{prediction:.1f}°C")
            st.image(xai_image, use_column_width=True)
            
        with col2:
            st.subheader("Material-Specific Suggestions")
            if selected_material in ["Concrete", "Asphalt"]:
                st.warning("🏗️ High heat retention material detected!")
                st.write("""
                - Apply reflective coatings
                - Increase shaded areas
                - Consider permeable pavement options
                """)
            elif selected_material == "Grass":
                st.success("🌿 Optimal natural surface detected")
            elif selected_material == "Water":
                st.info("💧 Water body helping with cooling")

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.write("Feature mismatch details:", str(e))

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.caption(f"Surface Material Options: {', '.join(SURFACE_MATERIALS)} | Model version: 3.0")
