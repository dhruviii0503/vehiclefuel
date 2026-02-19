import streamlit as st
import pandas as pd
import joblib


try:
    model = joblib.load("vehicle_fuel_efficency_model (3).pkl")
    encoder = joblib.load("label_encoder (7).pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


st.title("Vehicle Fuel Efficiency Prediction App")


engine_size = st.number_input("Engine Size", min_value=0.0, step=0.1)
cylinders = st.number_input("Cylinders", min_value=1, step=1)

transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
vehicle_class = st.selectbox("Vehicle Class", ["SUV", "Sedan", "Hatchback", "Truck"])


input_df = pd.DataFrame({
    "engine_size": [engine_size],
    "cylinders": [cylinders],
    "transmission": [transmission],
    "fuel_type": [fuel_type],
    "vehicle_class": [vehicle_class]
})


try:
    categorical_cols = ["transmission", "fuel_type", "vehicle_class"]

    for col in categorical_cols:
        input_df[col] = encoder[col].transform(input_df[col])

except Exception as e:
    st.error(f"Encoding error: {e}")
    st.stop()


if st.button("Predict Fuel Efficiency"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Fuel Efficiency: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")



