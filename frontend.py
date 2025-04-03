import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = r"C:\Users\AKSHAY\Resume Projects\ML - Regression Projects\House_Price_Prediction.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load dataset to extract unique locations
csv_path = r"C:\Users\AKSHAY\Resume Projects\ML - Regression Projects\Housing Price Prediction using Machine Learning\House_Data.csv"
df = pd.read_csv(csv_path)
unique_locations = df["location"].dropna().unique().tolist()

# Streamlit UI
st.title("🏡 Housing Price Prediction App")
st.write("Enter the house features below to predict its price.")

# Example feature inputs
sqft = st.number_input("Total Square Footage", min_value=300, max_value=10000, step=10)
bedrooms = st.selectbox("Number of Bedrooms", list(range(1, 11)))
bathrooms = st.selectbox("Number of Bathrooms", list(range(1, 11)))
balconies = st.selectbox("Number of Balconies", list(range(0, 6)))
location = st.selectbox("Location", unique_locations)

# Handle missing or unseen locations
location_encoded = unique_locations.index(location) if location in unique_locations else -1

# Predict Button
if st.button("Predict Price"):
    features = np.array([[sqft, bedrooms, bathrooms, balconies, location_encoded]])
    prediction = model.predict(features)
    st.success(f"🏠 Estimated House Price: ₹{prediction[0]:,.2f} Lakhs")
