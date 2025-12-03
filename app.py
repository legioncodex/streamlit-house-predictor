import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. LOAD THE MODEL (The Pipeline)
@st.cache_resource
def load_model():
    return joblib.load('house_price_model.joblib')

model = load_model()

# 2. APP TITLE & INPUTS
st.title("🏡 California House Price Predictor")
st.markdown("Use the input fields on the left to configure the neighborhood.")

st.sidebar.header("Input Features")

def user_input_features():
    # --- Input Fields replaced Sliders ---
    MedInc = st.sidebar.number_input('Median Income ($10k)', min_value=0.5, max_value=15.0, value=3.5, step=0.1)
    HouseAge = st.sidebar.number_input('Median House Age', min_value=1.0, max_value=52.0, value=20.0, step=1.0)
    AveRooms = st.sidebar.number_input('Avg. Rooms per Household', min_value=1.0, max_value=10.0, value=5.0, step=0.1)
    AveBedrms = st.sidebar.number_input('Avg. Bedrooms', min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    Population = st.sidebar.number_input('Population', min_value=100.0, max_value=5000.0, value=1000.0, step=10.0)
    AveOccup = st.sidebar.number_input('Avg. Occupancy', min_value=1.0, max_value=10.0, value=3.0, step=0.1)
    Latitude = st.sidebar.number_input('Latitude', min_value=32.0, max_value=42.0, value=34.0, step=0.01)
    Longitude = st.sidebar.number_input('Longitude', min_value=-125.0, max_value=-114.0, value=-118.0, step=0.01)

    data = {
        'MedInc': MedInc, 'HouseAge': HouseAge, 'AveRooms': AveRooms, 
        'AveBedrms': AveBedrms, 'Population': Population, 'AveOccup': AveOccup, 
        'Latitude': Latitude, 'Longitude': Longitude
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 3. PREDICTION
if st.button('Predict Price'):
    # The loaded pipeline automatically scales the input_df before predicting
    prediction = model.predict(input_df)
    
    st.subheader("Predicted Median House Value:")
    st.metric(label="USD", value=f"${prediction[0]:,.2f}")

st.divider()

# 4. EXPLANATION (Feature Importance)
if st.checkbox('Show Model Explanation (Feature Importance)'):
    st.subheader("What drives the price?")
    
    # Extract coefficients from the 'model' step of the pipeline
    coefficients = model.named_steps['model'].coef_
    
    # Feature names must be in the same order as used during training
    feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                     'Population', 'AveOccup', 'Latitude', 'Longitude']
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coefficients
    }).sort_values(by='Importance', ascending=False)
    
    # Use Streamlit's built-in plotting for simplicity
    st.bar_chart(importance_df.set_index('Feature')['Importance'])
    
    st.info("""
    **Interpretation:** Positive bars (high importance) push the price UP. 
    Negative bars (low importance) pull the price DOWN. 
    (E.g., High Median Income should always be a high positive bar.)
    """)
