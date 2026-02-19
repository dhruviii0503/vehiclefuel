import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("vehicle_fuel_efficency_model (3).pkl")

st.title("🚗 Vehicle Fuel Efficiency Prediction App")
st.write("Predict MPG (Miles Per Gallon)")

# Inputs
cylinders = st.number_input("Cylinders", min_value=1, step=1)
displacement = st.number_input("Displacement", min_value=0.0)
horsepower = st.number_input("Horsepower", min_value=0)
weight = st.number_input("Weight", min_value=0)
acceleration = st.number_input("Acceleration", min_value=0.0)
model_year = st.number_input("Model Year", min_value=1900, step=1)
origin = st.selectbox("Origin", [1, 2, 3])

if st.button("Predict MPG"):

    # Create empty dataframe with correct columns
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=model.feature_names_in_
    )

    # Fill numeric columns if they exist
    for col in input_data.columns:
        if col == "cylinders":
            input_data[col] = cylinders
        elif col == "displacement":
            input_data[col] = displacement
        elif col == "horsepower":
            input_data[col] = horsepower
        elif col == "weight":
            input_data[col] = weight
        elif col == "acceleration":
            input_data[col] = acceleration
        elif col in ["model year", "model_year"]:
            input_data[col] = model_year
        elif col == "origin":
            input_data[col] = origin
        elif col.startswith("origin_"):
            if col == f"origin_{origin}":
                input_data[col] = 1

    prediction = model.predict(input_data)

    st.success(f"Predicted MPG: {prediction[0]:.2f}")
