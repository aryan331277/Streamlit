import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# Required imports for model compatibility
from sklearn.ensemble import RandomForestRegressor
import sklearn

# --------------------------
# Configuration
# --------------------------
MODEL_PATH = "model.pkl"
XAI_IMAGE_PATH = "xai_feature_importance.png"

# --------------------------
# Load Resources
# --------------------------
try:
    model = joblib.load(MODEL_PATH)
    xai_image = Image.open(XAI_IMAGE_PATH)
except Exception as e:
    st.error(f"Error loading resources: {str(e)}")
    st.stop()

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Urban Analytics", layout="wide")
st.title("🌇 Urban Heat Island Comprehensive Analysis")
st.markdown("Complete urban environment assessment tool")

# --------------------------
# Sidebar Inputs
# --------------------------
with st.sidebar:
    st.header("Urban Parameters Configuration")
    
    # Geospatial Features
    st.subheader("Location Data")
    lat = st.number_input("Latitude", 19.0, 19.2, 19.0760, 0.0001)
    lon = st.number_input("Longitude", 72.8, 73.0, 72.8777, 0.0001)
    
    # Environmental Factors
    st.subheader("Environmental Features")
    green_cover = st.slider("Green Cover (%)", 0, 100, 25)
    albedo = st.slider("Albedo", 0.0, 1.0, 0.3, 0.05)
    humidity = st.slider("Relative Humidity (%)", 0, 100, 60)
    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 15.0, 3.0, 0.1)
    
    # Urban Infrastructure
    st.subheader("Urban Infrastructure")
    building_height = st.slider("Building Height (m)", 5, 150, 30)
    road_density = st.slider("Road Density (km/km²)", 0.0, 20.0, 5.0, 0.1)
    surface_material = st.selectbox("Surface Material", 
                                  ["Concrete", "Asphalt", "Grass", "Water", "Mixed"])
    
    # Advanced Metrics
    st.subheader("Advanced Metrics")
    population_density = st.number_input("Population Density (people/km²)", 1000, 50000, 20000)
    water_proximity = st.slider("Proximity to Water Body (m)", 0, 5000, 1000)
    vegetation_index = st.slider("Urban Vegetation Index", 0.0, 1.0, 0.5, 0.01)
    carbon_emissions = st.number_input("Carbon Emission Levels (CO₂ ppm)", 300, 1000, 400)
    
    # Temporal Features
    st.subheader("Temporal Data")
    solar_radiation = st.slider("Solar Radiation (W/m²)", 0, 1000, 500)
    night_temp = st.slider("Nighttime Surface Temperature (°C)", 15.0, 40.0, 25.0, 0.1)

# --------------------------
# Prediction System
# --------------------------
if st.sidebar.button("Run Analysis"):
    try:
        # Create input DataFrame with all features
        input_df = pd.DataFrame([[
            lat, lon, green_cover, albedo, humidity, wind_speed,
            building_height, road_density, surface_material,
            population_density, water_proximity, vegetation_index,
            carbon_emissions, solar_radiation, night_temp
        ]], columns=[
            'Latitude', 'Longitude', 'Green Cover Percentage', 'Albedo',
            'Relative Humidity', 'Wind Speed', 'Building Height',
            'Road Density', 'Surface Material', 'Population Density',
            'Proximity to Water Body', 'Urban Vegetation Index',
            'Carbon Emission Levels', 'Solar Radiation',
            'Nighttime Surface Temperature'
        ])
        
        # Encode categorical features
        surface_material_mapping = {"Concrete":0, "Asphalt":1, "Grass":2, "Water":3, "Mixed":4}
        input_df['Surface Material'] = input_df['Surface Material'].map(surface_material_mapping)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # --------------------------
        # Display Results
        # --------------------------
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("Core Metrics")
            st.metric("Predicted Surface Temperature", f"{prediction:.1f}°C")
            st.image(xai_image, caption="Feature Importance Analysis", use_column_width=True)
            
        with col2:
            st.subheader("Urban Health Assessment")
            
            # Environmental Recommendations
            if green_cover < 25:
                st.warning(f"🌳 Increase green cover (current: {green_cover}% → target: 25%+)")
            if water_proximity > 1000:
                st.warning(f"💧 Improve water proximity (current: {water_proximity}m → target: <1000m)")
            if albedo < 0.4:
                st.warning(f"🏗️ Use reflective materials (current albedo: {albedo} → target: 0.4+)")
                
            # Infrastructure Suggestions
            if building_height > 40:
                st.warning(f"🏢 Optimize building heights (current: {building_height}m → target: <40m)")
            if road_density > 8.0:
                st.warning(f"🛣️ Reduce road density (current: {road_density} km/km² → target: <8.0)")
                
            # Climate Impact
            if carbon_emissions > 450:
                st.warning(f"🌫️ Reduce carbon emissions (current: {carbon_emissions}ppm → target: <450)")
                
            # Thermal Comfort
            if night_temp > 28.0:
                st.warning(f"🌙 Address nighttime heat retention (current: {night_temp}°C → target: <28°C)")
                
            if prediction < 30.0:
                st.success("✅ Urban environment meets thermal comfort standards")

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("Urban Analytics Platform v2.0 | [Documentation](#) | [GitHub Repo](https://github.com/your-repo)")
