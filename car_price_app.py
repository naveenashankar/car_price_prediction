import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")
st.title("🚗 Car Price Prediction App")
st.write("Predict the selling price of a used car based on its specifications.")

car_age = st.number_input("Car Age (years)", min_value=0, max_value=20, value=5)
km_driven = st.number_input("KM Driven", min_value=0, max_value=300000, value=50000)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=40.0, value=18.0)
engine = st.number_input("Engine Capacity (CC)", min_value=800, max_value=5000, value=1200)
owner_type = st.selectbox("Owner Type", ['First', 'Second', 'Third', 'Fourth'])

label = LabelEncoder()
encoded = [label.fit_transform([fuel_type])[0],
           label.fit_transform([transmission])[0],
           label.fit_transform([owner_type])[0]]

input_data = pd.DataFrame([[car_age, km_driven, mileage, engine] + encoded],
                          columns=['Car_Age', 'KM_Driven', 'Mileage', 'Engine',
                                   'Fuel_Type', 'Transmission', 'Owner_Type'])

@st.cache_resource
def train_dummy_model():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 7), columns=input_data.columns)
    y = np.random.rand(100) * 20
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    return model

model = train_dummy_model()

if st.button("Predict Selling Price"):
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Selling Price: ₹ {prediction[0]:.2f} lakhs")
