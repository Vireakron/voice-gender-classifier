import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

st.title("🎤 Voice Gender Classifier")

uploaded_file = st.file_uploader("Upload a CSV file with voice features (excluding label)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 🔥 Drop non-numeric columns like 'label' if present
    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    # Standardize input
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    # Load model
    model = load_model("models/voice_gender_model.keras")

    # Predict
    pred = model.predict(X)

    # Show results
    for i, p in enumerate(pred):
        gender = "Male" if p[0] > 0.5 else "Female"
        st.write(f"Sample {i+1}: **{gender}** ({p[0]:.2f})")
