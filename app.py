import streamlit as st
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Restaurant Rating Predictor", layout="centered")

st.title("🍽️ Restaurant Rating Prediction System")
st.write("Cloud-deployed ML app that trains a model and predicts restaurant ratings")

# -----------------------
# CREATE SAMPLE DATASET
# -----------------------
@st.cache_resource
def train_model():
    # Synthetic dataset (acts like restaurant data)
    np.random.seed(42)

    data = pd.DataFrame({
        "online_order": np.random.randint(0, 2, 200),
        "table_booking": np.random.randint(0, 2, 200),
        "votes": np.random.randint(10, 500, 200),
        "cost": np.random.randint(100, 2000, 200),
    })

    # Fake but realistic rating logic
    data["rating"] = (
        3
        + 0.5 * data["online_order"]
        + 0.7 * data["table_booking"]
        + 0.002 * data["votes"]
        - 0.0003 * data["cost"]
        + np.random.normal(0, 0.2, 200)
    ).clip(1, 5)

    X = data[["online_order", "table_booking", "votes", "cost"]]
    y = data["rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = r2_score(y_test, model.predict(X_test))

    return model, round(score, 3)


model, r2 = train_model()

# -----------------------
# USER INPUT
# -----------------------
st.subheader("Enter Restaurant Details")

online_order = st.selectbox("Online Ordering Available?", ["Yes", "No"])
table_booking = st.selectbox("Table Booking Available?", ["Yes", "No"])
votes = st.number_input("Number of Votes", min_value=0, step=1)
cost = st.number_input("Approx Cost for Two (₹)", min_value=50, step=50)

online_val = 1 if online_order == "Yes" else 0
table_val = 1 if table_booking == "Yes" else 0

# -----------------------
# PREDICTION
# -----------------------
if st.button("Predict Rating"):
    features = np.array([[online_val, table_val, votes, cost]])
    prediction = model.predict(features)[0]

    st.success(f"⭐ Predicted Rating: **{round(prediction, 2)} / 5.0**")
    st.caption(f"Model Accuracy (R² Score): {r2}")
