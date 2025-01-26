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
    # Get original feature names from the model
    feature_names = model.feature_names_in_
except Exception as e:
    st.error(f"Error loading resources: {str(e)}")
    st.stop()

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Urban Analytics", layout="wide")
st.title("🌇 Urban Heat Island Comprehensive Analysis")

# --------------------------
# Input Fields Matching Original Training Features
# --------------------------
with st.sidebar:
    st.header("Urban Parameters")
    
    # Create input fields in EXACT order of original training features
    inputs = {}
    
    # Geospatial Features
    inputs['Latitude'] = st.number_input("Latitude", 19.0, 19.2, 19.0760, 0.0001)
    inputs['Longitude'] = st.number_input("Longitude", 72.8, 73.0, 72.8777, 0.0001)
    
    # Environmental Features
    inputs['Land Cover Type'] = st.selectbox("Land Cover Type", 
                                           ["Urban", "Suburban", "Rural", "Water"])
    inputs['Population Density'] = st.number_input("Population Density (people/km²)", 1000, 50000, 20000)
    inputs['Albedo'] = st.slider("Albedo", 0.0, 1.0, 0.3, 0.05)
    inputs['Green Cover Percentage'] = st.slider("Green Cover (%)", 0, 100, 25)
    inputs['Relative Humidity'] = st.slider("Relative Humidity (%)", 0, 100, 60)
    inputs['Wind Speed'] = st.slider("Wind Speed (m/s)", 0.0, 15.0, 3.0, 0.1)
    
    # Infrastructure Features
    inputs['Surface Material'] = st.selectbox("Surface Material", 
                                            ["Concrete", "Asphalt", "Grass", "Water", "Mixed"])
    inputs['Building Height'] = st.slider("Building Height (m)", 5, 150, 30)
    inputs['Road Density'] = st.slider("Road Density (km/km²)", 0.0, 20.0, 5.0, 0.1)
    inputs['Proximity to Water Body'] = st.slider("Water Proximity (m)", 0, 5000, 1000)
    
    # Thermal Features
    inputs['Solar Radiation'] = st.slider("Solar Radiation (W/m²)", 0, 1000, 500)
    inputs['Nighttime Surface Temperature'] = st.slider("Night Temperature (°C)", 15.0, 40.0, 25.0, 0.1)
    
    # Required Features from Error Message
    inputs['Distance from Previous Point'] = st.number_input("Distance from Previous Point (m)", 0, 5000, 100)
    inputs['Heat Stress Index'] = st.slider("Heat Stress Index", 0.0, 10.0, 3.5, 0.1)
    inputs['Urban Vegetation Index'] = st.slider("Vegetation Index", 0.0, 1.0, 0.5, 0.01)
    inputs['Carbon Emission Levels'] = st.number_input("CO₂ Levels (ppm)", 300, 1000, 400)
    
    # Add any other features from model.feature_names_in_

# --------------------------
# Prediction System
# --------------------------
if st.sidebar.button("Run Analysis"):
    try:
        # Create DataFrame with EXACT feature order from training
        input_df = pd.DataFrame([inputs], columns=feature_names)
        
        # Encode categoricals to match training preprocessing
        categorical_mappings = {
            'Land Cover Type': {"Urban":0, "Suburban":1, "Rural":2, "Water":3},
            'Surface Material': {"Concrete":0, "Asphalt":1, "Grass":2, "Water":3, "Mixed":4}
        }
        
        for col, mapping in categorical_mappings.items():
            input_df[col] = input_df[col].map(mapping)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # --------------------------
        # Display Results
        # --------------------------
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("Core Prediction")
            st.metric("Predicted Surface Temperature", f"{prediction:.1f}°C")
            st.image(xai_image, caption="Model Feature Importance", use_column_width=True)
            
        with col2:
            st.subheader("Actionable Insights")
            
            # Generate recommendations based on thresholds
            recommendations = []
            if inputs['Green Cover Percentage'] < 25:
                recommendations.append("🌳 Increase green cover to ≥25%")
            if inputs['Heat Stress Index'] > 4.0:
                recommendations.append("🌡️ Implement heat stress reduction measures")
            if inputs['Distance from Previous Point'] > 500:
                recommendations.append("📍 Optimize urban connectivity")
                
            if recommendations:
                for rec in recommendations:
                    st.warning(rec)
            else:
                st.success("✅ Urban parameters within optimal ranges")

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("Urban Analytics Platform v2.1 | Model Feature Count: {}".format(len(feature_names)))
# Footer
# --------------------------
st.markdown("---")
st.markdown("Urban Analytics Platform v2.0 | [Documentation](#) | [GitHub Repo](https://github.com/your-repo)")
