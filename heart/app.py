import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('heart_disease_model.h5')

# Function to normalize input data
def normalize_input(data):
    # Assuming the scaler was fitted on the same feature set during training
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# User input form
st.title("Heart Disease Prediction")

age = st.number_input("Age", min_value=0, max_value=120)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", (0, 1))
cp = st.selectbox("Chest Pain Type (0-3)", (0, 1, 2, 3))
trestbps = st.number_input("Resting Blood Pressure", min_value=0)
chol = st.number_input("Cholesterol", min_value=0)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = False, 1 = True)", (0, 1))
restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", (0, 1, 2))
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0)
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", (0, 1))
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest")
slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", (0, 1, 2))
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-3)", min_value=0)
thal = st.selectbox("Thalassemia (1 = Normal; 2 = Fixed Defect; 3 = Reversible Defect)", (1, 2, 3))

# One-hot encode categorical variables
cp_encoded = pd.get_dummies([cp], prefix='cp', drop_first=False)
restecg_encoded = pd.get_dummies([restecg], prefix='restecg', drop_first=False)
slope_encoded = pd.get_dummies([slope], prefix='slope', drop_first=False)
thal_encoded = pd.get_dummies([thal], prefix='thal', drop_first=False)

# Create a DataFrame from user inputs and encoded features
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'ca': [ca]
})

# Concatenate one-hot encoded features to the input DataFrame
input_data = pd.concat([input_data, cp_encoded, restecg_encoded, slope_encoded, thal_encoded], axis=1)

# Normalize the input data
normalized_data = normalize_input(input_data)

# Check the shape of the normalized data
st.write(f"Normalized data shape: {normalized_data.shape}")

# Make prediction
if st.button('Predict'):
    prediction = model.predict(normalized_data)
    result = "Heart Disease" if prediction[0][0] > 0.5 else "No Heart Disease"
    st.success(f'The model predicts: {result}')
