import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. LOAD THE MODEL (The Pipeline)
@st.cache_resource
def load_model():
    # The saved file contains the StandardScaler AND the Ridge model
    return joblib.load('house_price_model.joblib')

model = load_model()

# 2. APP TITLE & INPUTS
st.title("🏡 California House Price Predictor")
st.markdown("Use the sliders on the left to configure the neighborhood.")

st.sidebar.header("Input Features")

def user_input_features():
    # Feature inputs must match the data used in train_model.py
    MedInc = st.sidebar.slider('Median Income ($10k)', 0.5, 15.0, 3.5)
    HouseAge = st.sidebar.slider('Median House Age', 1.0, 52.0, 20.0)
    AveRooms = st.sidebar.slider('Avg. Rooms per Household', 1.0, 10.0, 5.0)
    AveBedrms = st.sidebar.slider('Avg. Bedrooms', 0.5, 5.0, 1.0)
    Population = st.sidebar.slider('Population', 100.0, 5000.0, 1000.0)
    AveOccup = st.sidebar.slider('Avg. Occupancy', 1.0, 10.0, 3.0)
    Latitude = st.sidebar.slider('Latitude', 32.0, 42.0, 34.0)
    Longitude = st.sidebar.number_input('Longitude', value=-118.0)

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
