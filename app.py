import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Restaurant Rating Predictor", layout="centered")

model = joblib.load("model.pkl")

st.title("🍽️ Restaurant Rating Prediction System")
st.markdown("Enter restaurant details to predict its customer rating using a Machine Learning model.")

location = st.number_input("Location Code", min_value=0, max_value=500)
rest_type = st.number_input("Restaurant Type Code", min_value=0, max_value=500)
cuisine = st.number_input("Cuisine Code", min_value=0, max_value=500)
cost = st.number_input("Approx Cost for Two", min_value=100, max_value=10000, step=50)
online = st.selectbox("Online Order Available?", [0, 1])

if st.button("Predict Rating"):
    features = np.array([location, rest_type, cuisine, cost, online]).reshape(1, -1)
    prediction = model.predict(features)[0]
    st.success(f"⭐ Predicted Rating: {round(prediction, 2)} / 5")
