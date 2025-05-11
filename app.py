# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

# Session state to store history
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("🏠 RentWise - India Rental Estimator")
st.markdown("Enter apartment details below to estimate monthly rent.")

# User Inputs
area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
floor = st.number_input("Floor", min_value=0, max_value=100, value=1)

city = st.selectbox("City", label_encoders['city'].classes_)
neighborhood = st.selectbox("Neighborhood", label_encoders['neighborhood'].classes_)
furnishing = st.selectbox("Furnishing Status", label_encoders['furnishing'].classes_)
area_type = st.selectbox("Area Type", label_encoders['area_type'].classes_)
tenant_preferred = st.selectbox("Tenant Preferred", label_encoders['tenant_preferred'].classes_)

# Predict Button
if st.button("Predict Rent"):
    city_encoded = label_encoders['city'].transform([city])[0]
    neighborhood_encoded = label_encoders['neighborhood'].transform([neighborhood])[0]
    furnishing_encoded = label_encoders['furnishing'].transform([furnishing])[0]
    area_type_encoded = label_encoders['area_type'].transform([area_type])[0]
    tenant_encoded = label_encoders['tenant_preferred'].transform([tenant_preferred])[0]

    features = np.array([[area, bedrooms, bathrooms, floor,
                          city_encoded, neighborhood_encoded,
                          furnishing_encoded, area_type_encoded, tenant_encoded]])

    predicted_rent = model.predict(features)[0]
    st.success(f"Estimated Monthly Rent: ₹{round(predicted_rent, 2)}")

    # Add to history
    st.session_state.history.append({
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floor": floor,
        "city": city,
        "neighborhood": neighborhood,
        "furnishing": furnishing,
        "area_type": area_type,
        "tenant_preferred": tenant_preferred,
        "predicted_rent": round(predicted_rent, 2)
    })

# Show history
if st.checkbox("Show Prediction History"):
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history))
    else:
        st.info("No predictions made yet.")
