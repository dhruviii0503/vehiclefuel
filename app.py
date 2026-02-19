import streamlit as st
import pandas as pd
import joblib

model = joblib.load("vehicle_fuel_efficency_model (3).pkl")


st.title("🚗 Vehicle Fuel Efficiency Prediction App")
st.write("Predict MPG (Miles Per Gallon) of a vehicle")


cylinders = st.number_input("Cylinders", min_value=1, step=1)
displacement = st.number_input("Displacement", min_value=0.0)
horsepower = st.number_input("Horsepower", min_value=0)
weight = st.number_input("Weight", min_value=0)
acceleration = st.number_input("Acceleration", min_value=0.0)
model_year = st.number_input("Model Year", min_value=1900, step=1)
origin = st.number_input("Origin (1=USA, 2=Europe, 3=Japan)", min_value=1, max_value=3, step=1)

input_data = pd.DataFrame({
        "cylinders": [cylinders],
        "displacement": [displacement],
        "horsepower": [horsepower],
        "weight": [weight],
        "acceleration": [acceleration],
        "model year": [model_year],
        "origin": [origin]
    })
input_data = input_data[model.feature_names_in_]


if st.button("Predict MPG"):
    prediction = model.predict(input_data)
    st.success(f"Predicted MPG: {prediction[0]:.2f}")



