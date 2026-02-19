import streamlit as st
import pandas as pd
import joblib

model = joblib.load("vehicle_fuel_efficency_model (3).pkl")

st.title("🚗 Vehicle Fuel Efficiency Prediction App")
st.write("Predict MPG (Miles Per Gallon) of a vehicle")

cylinders = st.number_input("cylinders", min_value=1, step=1)
displacement = st.number_input("displacement", min_value=0.0)
horsepower = st.number_input("horsepower", min_value=0)
weight = st.number_input("weight", min_value=0)
acceleration = st.number_input("acceleration", min_value=0.0)
model_year = st.number_input("model Year", min_value=1900, step=1)
origin = st.number_input("origin (1=USA, 2=Europe, 3=Japan)", min_value=1, max_value=3, step=1)

if st.button("Predict MPG"):

    input_data = pd.DataFrame([[
        cylinders,
        displacement,
        horsepower,
        weight,
        acceleration,
        model_year,
        origin
    ]], columns=model.feature_names_in_)  # 🔥 automatically match training columns

    prediction = model.predict(input_data)

    st.success(f"Predicted MPG: {prediction[0]:.2f}")
