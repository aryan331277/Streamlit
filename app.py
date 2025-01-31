import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import traceback
import sklearn
import pydeck as pdk
import numpy as np

MODEL_PATH = "trainedmodel.pkl"
XAI_IMAGE_PATH = "feature importance of rf regressor.png"
HEAT_THRESHOLDS = {
    'critical_temp': 38.0,
    'green_cover_min': 25,
    'albedo_min': 0.4,
    'building_height_max': 35,
    'heat_stress_max': 4.0,
    'population_density_max': 15000,
    'road_density_max': 12.0,
    'water_proximity_optimal': 500,
    'humidity_optimal': (40, 70),
    'vegetation_index_min': 0.6,
    'co2_max': 450
}

CITIES = {
    'Mumbai': {
        'lat_range': (18.9, 19.3),
        'lon_range': (72.7, 73.1),
        'default_lat': 19.0760,
        'default_lon': 72.8777
    },
    'Hyderabad': {
        'lat_range': (17.3, 17.5),
        'lon_range': (78.3, 78.6),
        'default_lat': 17.3850,
        'default_lon': 78.4867
    },
    'Delhi': {
        'lat_range': (28.4, 28.9),
        'lon_range': (76.8, 77.4),
        'default_lat': 28.7041,
        'default_lon': 77.1025
    }
}

try:
    model = joblib.load(MODEL_PATH)
    required_features = model.feature_names_in_
    xai_image = Image.open(XAI_IMAGE_PATH)
    st.session_state['sklearn_version'] = sklearn.__version__

except FileNotFoundError as e:
    st.error(f"""
    **Missing Critical File**  
    {str(e)}  
    🔍 Required files:  
    1. `model.pkl` - Trained model  
    2. `feature_importance.png` - Feature importance image  
    """)
    st.stop()
except Exception as e:
    st.error(f"""
    **Initialization Error**  
    {traceback.format_exc()}  
    """)
    st.stop()

st.set_page_config(page_title="Urban Heat Analyst", layout="wide")
st.title("Comprehensive Urban Heat Analysis")

# Regional Heatmap Section
st.subheader("Regional Heatmap Analysis")

def generate_sample_data(coords, avg_temp, num_points=50, temp_std=2):
    np.random.seed(42)
    lats = np.random.normal(coords[0], 0.1, num_points)
    lons = np.random.normal(coords[1], 0.1, num_points)
    temps = avg_temp + np.random.randn(num_points) * temp_std
    return pd.DataFrame({'lat': lats, 'lon': lons, 'temperature': temps})

cities_data = {
    'Hyderabad': {'coords': (17.3850, 78.4867), 'avg_temp': 35},
    'Delhi': {'coords': (28.7041, 77.1025), 'avg_temp': 40},
    'Mumbai': {'coords': (19.0760, 72.8777), 'avg_temp': 32}
}

all_data = pd.DataFrame()
for city in cities_data.values():
    city_data = generate_sample_data(city['coords'], city['avg_temp'])
    all_data = pd.concat([all_data, city_data], ignore_index=True)

heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=all_data,
    get_position='[lon, lat]',
    get_weight='temperature',
    aggregation="MEAN",
    radius_pixels=50,
    opacity=0.8,
    threshold=0.05,
    color_range=[
        [0, 128, 0, 150],    # Green (cool)
        [255, 255, 0, 150],  # Yellow
        [255, 165, 0, 150],  # Orange
        [255, 0, 0, 150]     # Red (hot)
    ]
)

view_state = pdk.ViewState(
    latitude=23.0,
    longitude=77.0,
    zoom=4,
    pitch=0,
    bearing=0
)

st.pydeck_chart(pdk.Deck(
    layers=[heatmap_layer],
    initial_view_state=view_state,
    map_style='mapbox://styles/mapbox/dark-v10'
))

with st.sidebar:
    st.header("City Configuration")
    selected_city = st.selectbox("Select City", list(CITIES.keys()))
    
    st.header("Urban Parameters")
    inputs = {}
    
    try:
        city_config = CITIES[selected_city]
        
        inputs['Latitude'] = st.number_input(
            "Latitude", 
            min_value=city_config['lat_range'][0],
            max_value=city_config['lat_range'][1],
            value=city_config['default_lat'],
            step=0.0001
        )
        
        inputs['Longitude'] = st.number_input(
            "Longitude",
            min_value=city_config['lon_range'][0],
            max_value=city_config['lon_range'][1],
            value=city_config['default_lon'],
            step=0.0001
        )
        
        # Urban parameters inputs
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
        inputs['Surface Material'] = st.selectbox("Surface Material", ["Concrete", "Asphalt", "Grass", "Water", "Mixed"])
        
    except KeyError as e:
        st.error(f"Missing input field: {str(e)}")
        st.stop()

if st.sidebar.button("Analyze Urban Heat"):
    try:
        missing_features = [f for f in required_features if f not in inputs]
        if missing_features:
            st.error(f"Missing features: {', '.join(missing_features)}")
            st.stop()
            
        input_df = pd.DataFrame([inputs], columns=required_features)
        
        if 'Surface Material' in required_features:
            material_map = {"Concrete":0, "Asphalt":1, "Grass":2, "Water":3, "Mixed":4}
            input_df['Surface Material'] = input_df['Surface Material'].map(material_map)
            input_df['Surface Material'] = input_df['Surface Material'].astype(int)

        with st.expander("Debug Information", expanded=False):
            st.write("### Model Expectations")
            st.write(f"scikit-learn version: {st.session_state.sklearn_version}")
            st.write(f"Required features ({len(required_features)}):", list(required_features))
            st.write("### Provided Inputs")
            st.write(input_df.T)

        prediction = model.predict(input_df)[0]

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Core Prediction")
            st.metric("Surface Temperature", f"{prediction:.1f}°C")
            st.image(xai_image, caption="Feature Impact Analysis", use_column_width=True)
            
        with col2:
            st.subheader("Urban Heat Mitigation Strategy")
            recommendations = []
            
            # Emergency protocol
if prediction > HEAT_THRESHOLDS['critical_temp']:
    st.error(f"🚨 {selected_city}: Extreme Heat Alert!")
    recommendations.extend([
        "🏢 Open cooling centers 24/7 with free hydration stations",
        "🚒 Deploy emergency misting stations in high-density areas",
        "🚑 Mobilize medical teams for heatstroke prevention and response"
    ])

# Parameter-Specific Recommendations
param_actions = {
    'Green Cover Percentage': (
        inputs['Green Cover Percentage'] < HEAT_THRESHOLDS['green_cover_min'],
        f"🌳 Increase urban greenery by at least {HEAT_THRESHOLDS['green_cover_min'] - inputs['Green Cover Percentage']}% to improve heat absorption. (Current: {inputs['Green Cover Percentage']}%)"
    ),
    'Albedo': (
        inputs['Albedo'] < HEAT_THRESHOLDS['albedo_min'],
        f"🌞 Enhance reflectivity by {HEAT_THRESHOLDS['albedo_min'] - inputs['Albedo']:.2f} using cool roofing and high-albedo pavements. (Current: {inputs['Albedo']})"
    ),
    'Building Height': (
        inputs['Building Height'] > HEAT_THRESHOLDS['building_height_max'],
        f"🏗️ Implement zoning laws to maintain max building height of {HEAT_THRESHOLDS['building_height_max']}m for better airflow. (Current: {inputs['Building Height']}m)"
    ),
    'Road Density': (
        inputs['Road Density'] > HEAT_THRESHOLDS['road_density_max'],
        f"🛣️ Reduce paved surfaces by {inputs['Road Density'] - HEAT_THRESHOLDS['road_density_max']:.1f} km/km²; prioritize permeable materials and shaded roads. (Current: {inputs['Road Density']} km/km²)"
    ),
    'Proximity to Water Body': (
        inputs['Proximity to Water Body'] > HEAT_THRESHOLDS['water_proximity_optimal'],
        f"💧 Establish water features within {HEAT_THRESHOLDS['water_proximity_optimal']}m for localized cooling. (Current: {inputs['Proximity to Water Body']}m)"
    ),
    'Population Density': (
        inputs['Population Density'] > HEAT_THRESHOLDS['population_density_max'],
        f"👥 Promote mixed-use development and urban decongestion strategies to redistribute {(inputs['Population Density'] - HEAT_THRESHOLDS['population_density_max'])/1000:.1f}k people/km²."
    ),
    'Heat Stress Index': (
        inputs['Heat Stress Index'] > HEAT_THRESHOLDS['heat_stress_max'],
        f"🔥 Enforce workplace heat safety protocols, increase shaded rest areas, and schedule work during cooler hours. (Current index: {inputs['Heat Stress Index']})"
    )
}

for param, (condition, message) in param_actions.items():
    if condition:
        recommendations.append(message)

# Humidity Optimization
hum = inputs['Relative Humidity']
if hum < HEAT_THRESHOLDS['humidity_optimal'][0]:
    recommendations.append(f"🏜️ Increase humidity via urban water bodies and misting systems. (Current: {hum}%)")
elif hum > HEAT_THRESHOLDS['humidity_optimal'][1]:
    recommendations.append(f"🌫️ Improve stormwater management to prevent excessive humidity buildup. (Current: {hum}%)")

# Vegetation & Carbon Emission Control
if inputs['Urban Vegetation Index'] < HEAT_THRESHOLDS['vegetation_index_min']:
    recommendations.append(f"🌿 Boost urban forestation and green walls. (Current index: {inputs['Urban Vegetation Index']:.2f})")

if inputs['Carbon Emission Levels'] > HEAT_THRESHOLDS['co2_max']:
    recommendations.append(f"🏭 Implement stricter air quality controls to cut emissions by {inputs['Carbon Emission Levels'] - HEAT_THRESHOLDS['co2_max']}ppm. (Current: {inputs['Carbon Emission Levels']}ppm)")

# City-Specific Strategies
city_strategies = {
    'Delhi': [
        "🌫️ Deploy advanced air purification towers and green buffer zones",
        "🌇 Retrofit public buildings with cool roofs and vertical gardens",
        "🚲 Expand shaded pedestrian and cycling networks to reduce vehicle dependency"
    ],
    'Mumbai': [
        "🌊 Strengthen mangrove conservation to enhance coastal cooling",
        "🌉 Improve cross-ventilation in sea bridge infrastructure",
        "🏙️ Design urban wind corridors for improved air circulation"
    ],
    'Hyderabad': [
        "🏞️ Develop extensive urban wetlands for natural cooling",
        "🌆 Implement AI-driven lake monitoring for sustainable water management",
        "🏗️ Enforce green building codes with mandatory rooftop gardens and solar reflectors"
    ]
}
recommendations.extend(city_strategies.get(selected_city, []))

# Display Recommendations
if recommendations:
    st.write("### 📌 Priority Action Plan")
    for i, rec in enumerate(recommendations, 1):
        st.warning(f"{i}. {rec}")
else:
    st.success("✅ All parameters are within optimal urban heat resilience ranges!")

except Exception as e:
    st.error(f"""
    **Analysis Failed**  
    {traceback.format_exc()}
    """)
    st.stop()
