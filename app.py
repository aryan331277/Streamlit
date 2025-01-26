import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# --------------------------
# Configuration
# --------------------------
MODEL_PATH = "model.pkl"
XAI_IMAGE_PATH = "feature importance of rf regressor.png"
HEAT_THRESHOLDS = {
    'critical_temp': 38.0,
    'green_cover_min': 25,
    'albedo_min': 0.4,
    'building_height_max': 35,
    'heat_stress_max': 4.0,
    'population_density_max': 15000
}

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
st.set_page_config(page_title="Urban Heat Analyst", layout="wide")
st.title("🌡️ Comprehensive Urban Heat Analysis")

# --------------------------
# Input Handling
# --------------------------
with st.sidebar:
    st.header("Urban Parameters")
    inputs = {}
    
    # Numerical Features
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
    
    # Categorical Feature
    inputs['Surface Material'] = st.selectbox("Surface Material", 
                                            ["Concrete", "Asphalt", "Grass", "Water", "Mixed"])

# --------------------------
# Prediction System
# --------------------------
if st.sidebar.button("Analyze Urban Heat"):
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([inputs], columns=feature_names)
        
        # Encode surface material
        material_map = {"Concrete":0, "Asphalt":1, "Grass":2, "Water":3, "Mixed":4}
        input_df['Surface Material'] = input_df['Surface Material'].map(material_map)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # --------------------------
        # Comprehensive Analysis
        # --------------------------
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Prediction Results")
            st.metric("Predicted Surface Temperature", f"{prediction:.1f}°C")
            st.image(xai_image, caption="Feature Impact Analysis", use_column_width=True)
            
        with col2:
            st.subheader("Heat Mitigation Strategy")
            
            # Initialize recommendations
            recommendations = []
            urgency_level = 0
            
            # Temperature-based urgency
            if prediction > HEAT_THRESHOLDS['critical_temp']:
                recommendations.append(("🚨 Emergency Cooling Needed", "critical"))
                urgency_level = 3
            elif prediction > 35.0:
                urgency_level = 2
            else:
                urgency_level = 1

            # Feature-based recommendations
            if inputs['Green Cover Percentage'] < HEAT_THRESHOLDS['green_cover_min']:
                deficit = HEAT_THRESHOLDS['green_cover_min'] - inputs['Green Cover Percentage']
                recommendations.append((
                    f"🌳 Increase green cover by {deficit}% (Current: {inputs['Green Cover Percentage']}%)",
                    "high"
                ))
                
            if inputs['Albedo'] < HEAT_THRESHOLDS['albedo_min']:
                recommendations.append((
                    f"🏗️ Use reflective materials to increase albedo above {HEAT_THRESHOLDS['albedo_min']}",
                    "medium"
                ))
                
            if inputs['Building Height'] > HEAT_THRESHOLDS['building_height_max']:
                recommendations.append((
                    f"🏢 Optimize building heights below {HEAT_THRESHOLDS['building_height_max']}m",
                    "medium"
                ))
                
            if inputs['Heat Stress Index'] > HEAT_THRESHOLDS['heat_stress_max']:
                recommendations.append((
                    f"🌡️ Implement heat stress reduction measures (Current HSI: {inputs['Heat Stress Index']})",
                    "high"
                ))
                
            if inputs['Population Density'] > HEAT_THRESHOLDS['population_density_max']:
                recommendations.append((
                    f"👥 Reduce population density through urban planning",
                    "medium"
                ))
                
            if inputs['Surface Material'] in ["Concrete", "Asphalt"]:
                recommendations.append((
                    "🛣️ Consider permeable pavement options for better heat dissipation",
                    "medium"
                ))

            # Display recommendations by priority
            if recommendations:
                st.write("### Priority Actions")
                priority_order = {"critical": 0, "high": 1, "medium": 2}
                for rec in sorted(recommendations, key=lambda x: priority_order[x[1]]):
                    if rec[1] == "critical":
                        st.error(rec[0])
                    elif rec[1] == "high":
                        st.warning(rec[0])
                    else:
                        st.info(rec[0])
            else:
                st.success("✅ Urban parameters within optimal ranges for heat management")

            # Thermal comfort analysis
            st.write("### Thermal Profile")
            col2.metric("Heat Stress Risk", 
                       "High" if inputs['Heat Stress Index'] > 4.0 else "Moderate" if inputs['Heat Stress Index'] > 2.5 else "Low",
                       help="Combined impact of temperature and humidity on human comfort")
            
            col2.metric("Cooling Potential", 
                       f"{inputs['Proximity to Water Body']}m from water body",
                       delta="-2°C per 100m" if inputs['Proximity to Water Body'] < 500 else "+1.5°C over 500m")

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.caption("Urban Heat Analysis Framework v4.0 | Thresholds based on WHO urban guidelines")
# Footer
# --------------------------
st.markdown("---")
st.caption(f"Surface Material Options: {', '.join(SURFACE_MATERIALS)} | Model version: 3.0")
