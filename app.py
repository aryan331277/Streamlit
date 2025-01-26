# app.py
import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# Required for scikit-learn model compatibility
from sklearn.ensemble import RandomForestRegressor
import sklearn

# --------------------------
# Configuration
# --------------------------
MODEL_PATH = "model.pkl"
XAI_IMAGE_PATH = "feature importance of rf regressor.png"

# --------------------------
# Load Resources with Error Handling
# --------------------------
try:
    model = joblib.load(MODEL_PATH)
    xai_image = Image.open(XAI_IMAGE_PATH)
except FileNotFoundError as e:
    st.error(f"Critical file missing: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"Error loading resources: {str(e)}")
    st.stop()

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Urban Heat Predictor", layout="wide")
st.title("🌍 Urban Heat Island Prediction")
st.markdown("Predict temperature and get mitigation strategies")

# --------------------------
# Sidebar Inputs
# --------------------------
with st.sidebar:
    st.header("Urban Parameters")
    
    lat = st.slider("Latitude", 19.0, 19.2, 19.0760, 0.0001, format="%.4f")
    lon = st.slider("Longitude", 72.8, 73.0, 72.8777, 0.0001, format="%.4f")
    green_cover = st.slider("Green Cover (%)", 0, 100, 25)
    building_height = st.slider("Building Height (m)", 5, 150, 30)
    population_density = st.slider("Population Density (people/km²)", 1000, 50000, 20000)
    albedo = st.slider("Albedo (reflectivity)", 0.0, 1.0, 0.3, 0.05)

# --------------------------
# Prediction & Results
# --------------------------
if st.sidebar.button("Calculate Temperature"):
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([[lat, lon, green_cover, building_height, population_density, albedo]],
                               columns=["Latitude", "Longitude", "Green_Cover", "Building_Height", 
                                        "Population_Density", "Albedo"])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Predicted Temperature: {prediction:.1f}°C")
            st.image(xai_image, caption="Feature Importance Analysis", use_column_width=True)
            
        with col2:
            st.subheader("Mitigation Recommendations")
            
            recommendations = []
            if green_cover < 25:
                recommendations.append("🌳 **Increase green cover** to at least 25% (current: {green_cover}%)")
            if building_height > 25:
                recommendations.append("🏗️ **Reduce building height** below 25m (current: {building_height}m)")
            if albedo < 0.4:
                recommendations.append("🏠 **Use reflective materials** to increase albedo above 0.4 (current: {albedo})")
            if population_density > 15000:
                recommendations.append("🏙️ **Optimize urban layout** to reduce density (current: {population_density} people/km²)")
                
            if not recommendations:
                st.success("✅ Urban design meets heat mitigation standards")
            else:
                for rec in recommendations:
                    st.warning(rec.format(
                        green_cover=green_cover,
                        building_height=building_height,
                        albedo=albedo,
                        population_density=population_density
                    ))
                    
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("Built with ♻️ | [GitHub Repo](https://github.com/your-repo) | v1.0")
