import streamlit as st
import pandas as pd
import joblib

# ------------------ Load Model & Encoders ------------------ #
try:
    model = joblib.load("vehicle_fuel_efficency_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ------------------ App Title ------------------ #
st.title("🚗 Vehicle Fuel Efficiency Prediction App")

# ------------------ User Inputs ------------------ #
engine_size = st.number_input("Engine Size", min_value=0.0, step=0.1)
cylinders = st.number_input("Cylinders", min_value=1, step=1)

transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
vehicle_class = st.selectbox("Vehicle Class", ["SUV", "Sedan", "Hatchback", "Truck"])

# ------------------ Create DataFrame ------------------ #
input_df = pd.DataFrame({
    "engine_size": [engine_size],
    "cylinders": [cylinders],
    "transmission": [transmission],
    "fuel_type": [fuel_type],
    "vehicle_class": [vehicle_class]
})

# ------------------ Encode Categorical Columns ------------------ #
try:
    categorical_cols = ["transmission", "fuel_type", "vehicle_class"]
    
    for col in categorical_cols:
        input_df[col] = encoder[col].transform(input_df[col])

except Exception as e:
    st.error(f"Encoding error: {e}")
    st.stop()

# ------------------ Prediction ------------------ #
if st.button("Predict Fuel Efficiency"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Fuel Efficiency: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
