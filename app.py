import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the model
model = tf.keras.models.load_model("model.h5")

# Load encoders and scaler
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Title
st.title("📊 Customer Churn Prediction App")

# User Inputs
geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("👤 Gender", label_encoder_gender.classes_)
age = st.slider("🎂 Age", 18, 92)
credit_score = st.number_input("💳 Credit Score", min_value=300, max_value=900, value=650)
balance = st.number_input("💰 Balance")
estimated_salary = st.number_input("💼 Estimated Salary")
tenure = st.slider("📅 Tenure (years)", 0, 10)
num_of_products = st.slider("📦 Number of Products", 1, 4)
has_cr_card = st.selectbox("💳 Has Credit Card?", [0, 1])
is_active_member = st.selectbox("✅ Is Active Member?", [0, 1])

# Encode inputs
gender_encoded = label_encoder_gender.transform([gender])[0]
geography_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

# Create input DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Combine inputs
final_input = np.concatenate([geography_encoded, input_data.values], axis=1)

# Scale
scaled_input = scaler.transform(final_input)

# Predict
if st.button("🔮 Predict Churn"):
    result = model.predict(scaled_input)
    prediction = (result > 0.5).astype(int)[0][0]

    if prediction == 1:
        st.error("❌ This customer is likely to churn!")
    else:
        st.success("✅ This customer will likely stay.")

