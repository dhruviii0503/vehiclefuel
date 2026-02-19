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


cylinders = st.number_input("Cylinders", min_value=1, step=1)
displacement = st.number_input("Displacement", min_value=0.0)
horsepower = st.number_input("Horsepower", min_value=0)
weight = st.number_input("Weight", min_value=0)
acceleration = st.number_input("Acceleration", min_value=0.0)
model_year = st.number_input("Model Year", min_value=1900, step=1)
origin = st.number_input("Origin (1=USA, 2=Europe, 3=Japan)", min_value=1, max_value=3, step=1)
car name = st.number_input("Car Name (Encoded Number)", min_value=0)


input_df = pd.DataFrame({
    "cylinders": [cylinders],
        "displacement": [displacement],
        "horsepower": [horsepower],
        "weight": [weight],
        "acceleration": [acceleration],
        "model year": [model year],
        "origin": [origin],
        "car name": [car name]
})


try:
    categorical_cols = ["car name", "model year", "acceleration"]

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



