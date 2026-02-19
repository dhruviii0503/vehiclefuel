import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Trained Model
# ----------------------------
model = joblib.load("vehicle_fuel_efficency_model (3).pkl")

# ----------------------------
# App Title
# ----------------------------
st.title("🚗 Vehicle Fuel Efficiency Prediction App")
st.write("Predict MPG (Miles Per Gallon) of a vehicle")

# ----------------------------
# User Inputs
# ----------------------------
cylinders = st.number_input("Cylinders", min_value=1, step=1)
displacement = st.number_input("Displacement", min_value=0.0)
horsepower = st.number_input("Horsepower", min_value=0)
weight = st.number_input("Weight", min_value=0)
acceleration = st.number_input("Acceleration", min_value=0.0)
model_year = st.number_input("Model Year", min_value=1900, step=1)
origin = st.selectbox("Origin", [1, 2, 3])  # 1=USA, 2=Europe, 3=Japan

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict MPG"):

    # Create base dataframe (numeric features)
    input_dict = {
        "cylinders": cylinders,
        "displacement": displacement,
        "horsepower": horsepower,
        "weight": weight,
        "acceleration": acceleration,
        "model year": model_year,
    }

    # Add one-hot encoded origin columns (if model expects them)
    for col in model.feature_names_in_:
        if col.startswith("origin_"):
            input_dict[col] = 1 if col == f"origin_{origin}" else 0

    # If model expects plain 'origin'
    if "origin" in model.feature_names_in_:
        input_dict["origin"] = origin

    # Convert to DataFrame with correct column order
    input_data = pd.DataFrame([input_dict])
    input_data = input_data[model.feature_names_in_]

    # Predict
    prediction = model.predict(input_data)

    st.success(f"Predicted MPG: {prediction[0]:.2f}")
