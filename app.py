import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import traceback
import sklearn
import requests  

API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-1.3B"  
API_KEY = "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"  
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

def generate_suggestions(prompt):
    """Generate suggestions using Hugging Face's Inference API."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 200,  
            "temperature": 0.7,  
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"Error: {response.status_code}, {response.text}"

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
    2. `xai_feature_importance.png` - Feature importance image  
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

with st.sidebar:
    st.header("Urban Parameters")
    inputs = {}
    
    try:
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
            st.subheader("Heat Mitigation Strategy")
            
            prompt = f"""
            Based on the following urban heat data, provide actionable suggestions to mitigate heat issues:
            - Surface Temperature: {prediction:.1f}°C
            - Green Cover: {inputs['Green Cover Percentage']}%
            - Albedo: {inputs['Albedo']}
            - Population Density: {inputs['Population Density']} people/km²
            - Building Height: {inputs['Building Height']}m
            - Heat Stress Index: {inputs['Heat Stress Index']}
            - Surface Material: {inputs['Surface Material']}
            """
            
            suggestions = generate_suggestions(prompt)
            
            st.write("### Recommendations")
            st.write(suggestions)

    except Exception as e:
        st.error(f"""
        **Analysis Failed**  
        {traceback.format_exc()}
        """)
        st.stop()
