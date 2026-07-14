import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.title("🥇 Gold Price Prediction")

# Load model and scaler
model = load_model("gold_price_lstm.keras")
scaler = joblib.load("gold_scaler.pkl")

# Load dataset
df = pd.read_csv("goldprices.csv")

st.write(df.tail())

# Last 30 Close prices
last_30 = df["Close"].tail(30).values.reshape(-1,1)

# Scale
scaled = scaler.transform(last_30)

# Reshape for LSTM
X = scaled.reshape(1,30,1)

if st.button("Predict Tomorrow Price"):

    prediction = model.predict(X)

    prediction = scaler.inverse_transform(prediction)

    st.success(f"Predicted Gold Price: {prediction[0][0]:.2f}")

st.line_chart(df["Close"])

if st.button("Predict Next 10 Days"):

    last_30 = df["Close"].tail(30).values.reshape(-1,1)

    scaled_data = scaler.transform(last_30)

    temp_input = scaled_data.flatten().tolist()

    future = []

    for i in range(10):

        x_input = np.array(temp_input[-30:])
        x_input = x_input.reshape(1,30,1)

        pred = model.predict(x_input, verbose=0)

        future.append(pred[0][0])

        temp_input.append(pred[0][0])

    future = scaler.inverse_transform(
        np.array(future).reshape(-1,1)
    )

    st.subheader("10-Day Gold Price Prediction")

    result = []

    for i in range(10):
        result.append({
            "Day": f"Day {i+1}",
            "Predicted Price": round(float(future[i][0]),2)
        })

    prediction_df = pd.DataFrame(result)

    st.dataframe(prediction_df)

    st.line_chart(prediction_df.set_index("Day"))
