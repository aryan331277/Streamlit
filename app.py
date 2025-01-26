import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import traceback
import sklearn

# --------------------------
# Configuration
# --------------------------
MODEL_PATH = "model.pkl"
XAI_IMAGE_PATH = "xai_feature_importance.png"

HEAT_THRESHOLDS = {
    # Temperature Parameters
    'critical_temp': 38.0,
    'night_temp_max': 28.0,
    
    # Environmental Features
    'green_cover_min': 25,
    'albedo_min': 0.4,
    'vegetation_index_min': 0.6,
    'water_proximity_max': 500,  # meters
    
    # Urban Infrastructure
    'building_height_max': 35,
    'road_density_max': 8.0,  # km/km²
    'surface_material_risk': ["Concrete", "Asphalt"],
    
    # Population & Emissions
    'population_density_max': 15000,
    'carbon_emission_max': 450,  # ppm
    
    # Climate Factors
    'humidity_max': 75,  # %
    'wind_speed_min': 2.0,  # m/s
    'solar_radiation_max': 700,  # W/m²
    
    # Urban Connectivity
    'distance_previous_max': 1000  # meters
}

# --------------------------
# Load Resources
# --------------------------
try:
    model = joblib.load(MODEL_PATH)
    required_features = model.feature_names_in_
    xai_image = Image.open(XAI_IMAGE_PATH)
    st.session_state['sklearn_version'] = sklearn.__version__
except Exception as e:
    st.error(f"Initialization Error: {traceback.format_exc()}")
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
    
    # Geospatial Features
    inputs['Latitude'] = st.number_input("Latitude", 19.0, 19.2, 19.0760, 0.0001)
    inputs['Longitude'] = st.number_input("Longitude", 72.8, 73.0, 72.8777, 0.0001)
    
    # Environmental Features
    inputs['Green Cover Percentage'] = st.slider("Green Cover (%)", 0, 100, 25)
    inputs['Albedo'] = st.slider("Albedo", 0.0, 1.0, 0.3, 0.05)
    inputs['Urban Vegetation Index'] = st.slider("Vegetation Index", 0.0, 1.0, 0.5, 0.01)
    inputs['Proximity to Water Body'] = st.slider("Water Proximity (m)", 0, 5000, 1000)
    
    # Urban Infrastructure
    inputs['Building Height'] = st.slider("Building Height (m)", 5, 150, 30)
    inputs['Road Density'] = st.slider("Road Density (km/km²)", 0.0, 20.0, 5.0, 0.1)
    inputs['Surface Material'] = st.selectbox("Surface Material", 
                                            ["Concrete", "Asphalt", "Grass", "Water", "Mixed"])
    
    # Climate Factors
    inputs['Relative Humidity'] = st.slider("Humidity (%)", 0, 100, 60)
    inputs['Wind Speed'] = st.slider("Wind Speed (m/s)", 0.0, 15.0, 3.0, 0.1)
    inputs['Solar Radiation'] = st.slider("Solar Radiation (W/m²)", 0, 1000, 500)
    inputs['Nighttime Surface Temperature'] = st.slider("Night Temp (°C)", 15.0, 40.0, 25.0, 0.1)
    
    # Population & Emissions
    inputs['Population Density'] = st.number_input("Population Density (people/km²)", 1000, 50000, 20000)
    inputs['Carbon Emission Levels'] = st.number_input("CO₂ Levels (ppm)", 300, 1000, 400)
    
    # Urban Connectivity
    inputs['Distance from Previous Point'] = st.number_input("Distance from Previous Point (m)", 0, 5000, 100)

# --------------------------
# Prediction System
# --------------------------
if st.sidebar.button("Analyze Urban Heat"):
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([inputs], columns=required_features)
        
        # Encode categorical features
        material_map = {"Concrete":0, "Asphalt":1, "Grass":2, "Water":3, "Mixed":4}
        if 'Surface Material' in required_features:
            input_df['Surface Material'] = input_df['Surface Material'].map(material_map).astype(int)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # --------------------------
        # Comprehensive Analysis
        # --------------------------
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Core Prediction")
            st.metric("Surface Temperature", f"{prediction:.1f}°C")
            st.image(xai_image, caption="Feature Impact Analysis", use_column_width=True)
            
        with col2:
            st.subheader("Heat Mitigation Strategy")
            recommendations = []
            
            # Emergency check
            if prediction > HEAT_THRESHOLDS['critical_temp']:
                st.error("🚨 EMERGENCY: Activate heat emergency protocols")
                
            # Feature-based recommendations
            def add_rec(condition, message, priority):
                if condition: recommendations.append((message, priority))
            
            add_rec(inputs['Green Cover Percentage'] < HEAT_THRESHOLDS['green_cover_min'],
                   f"🌳 Increase green cover to ≥{HEAT_THRESHOLDS['green_cover_min']}% (Current: {inputs['Green Cover Percentage']}%)",
                   "high")
            
            add_rec(inputs['Albedo'] < HEAT_THRESHOLDS['albedo_min'],
                   f"🏗️ Improve albedo to ≥{HEAT_THRESHOLDS['albedo_min']} using reflective materials",
                   "medium")
            
            add_rec(inputs['Building Height'] > HEAT_THRESHOLDS['building_height_max'],
                   f"🏢 Limit building heights to ≤{HEAT_THRESHOLDS['building_height_max']}m",
                   "medium")
            
            add_rec(inputs['Road Density'] > HEAT_THRESHOLDS['road_density_max'],
                   f"🛣️ Reduce road density below {HEAT_THRESHOLDS['road_density_max']} km/km²",
                   "medium")
            
            add_rec(inputs['Proximity to Water Body'] > HEAT_THRESHOLDS['water_proximity_max'],
                   f"💧 Develop water bodies within {HEAT_THRESHOLDS['water_proximity_max']}m radius",
                   "high")
            
            add_rec(inputs['Urban Vegetation Index'] < HEAT_THRESHOLDS['vegetation_index_min'],
                   f"🌿 Improve vegetation index to ≥{HEAT_THRESHOLDS['vegetation_index_min']}",
                   "high")
            
            add_rec(inputs['Carbon Emission Levels'] > HEAT_THRESHOLDS['carbon_emission_max'],
                   f"🌍 Reduce CO₂ emissions below {HEAT_THRESHOLDS['carbon_emission_max']}ppm",
                   "high")
            
            add_rec(inputs['Relative Humidity'] > HEAT_THRESHOLDS['humidity_max'],
                   f"💨 Improve ventilation to reduce humidity below {HEAT_THRESHOLDS['humidity_max']}%",
                   "medium")
            
            add_rec(inputs['Wind Speed'] < HEAT_THRESHOLDS['wind_speed_min'],
                   f"🌬️ Enhance wind corridors for ≥{HEAT_THRESHOLDS['wind_speed_min']}m/s airflow",
                   "medium")
            
            add_rec(inputs['Solar Radiation'] > HEAT_THRESHOLDS['solar_radiation_max'],
                   f"☀️ Install shading structures for solar radiation ≤{HEAT_THRESHOLDS['solar_radiation_max']}W/m²",
                   "high")
            
            add_rec(inputs['Nighttime Surface Temperature'] > HEAT_THRESHOLDS['night_temp_max'],
                   f"🌙 Improve nighttime cooling below {HEAT_THRESHOLDS['night_temp_max']}°C",
                   "high")
            
            add_rec(inputs['Distance from Previous Point'] > HEAT_THRESHOLDS['distance_previous_max'],
                   f"📍 Optimize urban connectivity within {HEAT_THRESHOLDS['distance_previous_max']}m radius",
                   "medium")
            
            add_rec(inputs['Surface Material'] in HEAT_THRESHOLDS['surface_material_risk'],
                   f"🛣️ Convert {inputs['Surface Material']} surfaces to cooler alternatives",
                   "high")

            # Display recommendations
            if recommendations:
                priority_order = {"high": 0, "medium": 1}
                sorted_recs = sorted(recommendations, key=lambda x: priority_order[x[1]])
                
                st.write("### Priority Action Plan")
                for rec, priority in sorted_recs:
                    if priority == "high":
                        st.warning(f"🔥 {rec}")
                    else:
                        st.info(f"⚙️ {rec}")
            else:
                st.success("✅ All parameters within optimal heat management ranges")

    except Exception as e:
        st.error(f"Analysis failed: {traceback.format_exc()}")
        st.stop()

# --------------------------
# System Info
# --------------------------
st.markdown("---")
st.caption(f"""
**Threshold Guidelines**  
Green Cover: ≥{HEAT_THRESHOLDS['green_cover_min']}%  
Building Height: ≤{HEAT_THRESHOLDS['building_height_max']}m  
Night Temp: ≤{HEAT_THRESHOLDS['night_temp_max']}°C  
Water Proximity: ≤{HEAT_THRESHOLDS['water_proximity_max']}m  
Model Features: {len(required_features)}  
""")
