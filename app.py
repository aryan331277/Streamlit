import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# --------------------------
# Configuration
# --------------------------
MODEL_PATH = "model.pkl"
XAI_IMAGE_PATH = "feature importance of rf regressor.png"

# --------------------------
# Load Resources
# --------------------------
try:
    model = joblib.load(MODEL_PATH)
    xai_image = Image.open(XAI_IMAGE_PATH)
    feature_names = model.feature_names_in_  # Get original feature order
except Exception as e:
    st.error(f"Error loading resources: {str(e)}")
    st.stop()

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Urban Analytics", layout="wide")
st.title("🌡️ Urban Heat Prediction Tool")

# --------------------------
# Input Fields (Aligned with Model Expectations)
# --------------------------
with st.sidebar:
    st.header("Urban Parameters")
    inputs = {}
    
    # Mandatory Features from Model
    inputs['Latitude'] = st.number_input("Latitude", 19.0, 19.2, 19.0760, 0.0001)
    inputs['Longitude'] = st.number_input("Longitude", 72.8, 73.0, 72.8777, 0.0001)
    inputs['Population Density'] = st.number_input("Population Density (people/km²)", 1000, 50000, 20000)
    inputs['Albedo'] = st.slider("Albedo", 0.0, 1.0, 0.3, 0.05)
    inputs['Green Cover Percentage'] = st.slider("Green Cover (%)", 0, 100, 25)
    inputs['Relative Humidity'] = st.slider("Humidity (%)", 0, 100, 60)
    inputs['Wind Speed'] = st.slider("Wind Speed (m/s)", 0.0, 15.0, 3.0, 0.1)
    inputs['Building Height'] = st.slider("Building Height (m)", 5, 150, 30)
    inputs['Road Density'] = st.slider("Road Density (km/km²)", 0.0, 20.0, 5.0, 0.1)
    inputs['Proximity to Water Body'] = st.slider("Water Proximity (m)", 0, 5000, 1000)
    inputs['Solar Radiation'] = st.slider("Solar Radiation (W/m²)", 0, 1000, 500)
    inputs['Nighttime Surface Temperature'] = st.slider("Night Temp (°C)", 15.0, 40.0, 25.0, 0.1)
    inputs['Distance from Previous Point'] = st.number_input("Distance from Previous Point (m)", 0, 5000, 100)
    inputs['Heat Stress Index'] = st.slider("Heat Stress Index", 0.0, 10.0, 3.5, 0.1)
    inputs['Urban Vegetation Index'] = st.slider("Vegetation Index", 0.0, 1.0, 0.5, 0.01)
    inputs['Carbon Emission Levels'] = st.number_input("CO₂ Levels (ppm)", 300, 1000, 400)
    
    # Surface Material with consistent encoding
    inputs['Surface Material'] = st.selectbox("Surface Material", 
                                            ["Concrete", "Asphalt", "Grass", "Water", "Mixed"])

# --------------------------
# Prediction System
# --------------------------
if st.sidebar.button("Predict Temperature"):
    try:
        # Create input DataFrame with EXACT feature order
        input_df = pd.DataFrame([inputs], columns=feature_names)
        
        # Encode surface material to match training
        material_map = {"Concrete":0, "Asphalt":1, "Grass":2, "Water":3, "Mixed":4}
        input_df['Surface Material'] = input_df['Surface Material'].map(material_map)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # --------------------------
        # Display Results
        # --------------------------
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Prediction Results")
            st.metric("Surface Temperature", f"{prediction:.1f}°C")
            st.image(xai_image, caption="Key Influencing Factors", use_column_width=True)
            
        with col2:
            st.subheader("Optimization Suggestions")
            
            # Generate actionable recommendations
            if prediction > 35:
                st.error("🚨 Critical heat levels detected!")
                st.write("""
                - Immediately increase green cover by 10-15%
                - Implement emergency cooling measures
                - Restrict high-emission activities
                """)
                
            if inputs['Heat Stress Index'] > 4.0:
                st.warning("🔥 High heat stress detected:")
                st.write("""
                - Install shade structures in public areas
                - Increase water station availability
                - Adjust outdoor work schedules
                """)
                
            if inputs['Carbon Emission Levels'] > 450:
                st.warning("🌍 High emissions detected:")
                st.write("""
                - Promote public transportation
                - Implement green energy incentives
                - Optimize waste management
                """)

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.caption(f"Model expecting {len(feature_names)} features | Version 2.2 | [GitHub Repo](#)")
