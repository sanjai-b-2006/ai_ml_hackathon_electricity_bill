import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load Saved Models and Data ---
assets = {}
try:
    model_files = {
            "Linear Regression ": "linear_regression.pkl",
             "Linear Regression without 100 percent precision": "linear_regression_without.pkl",
        }
    assets['models'] = {}
    for name, file in model_files.items():
        with open(file, 'rb') as f:
            assets['models'][name] = pickle.load(f)
    with open('linear_regression.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    with open('app_data.pkl', 'rb') as f:
        app_data = pickle.load(f)
    CITIES = app_data['cities']
    COMPANIES = app_data['companies']

except FileNotFoundError:
    st.error("Model files not found. Please run `train_and_save_models.py` first.")
    st.stop()


st.set_page_config(page_title="Electricity Bill Forecaster", layout="wide")
st.title("💡 Electricity Bill Forecaster")
st.markdown("Predict monthly electricity bills based on appliance usage and location.")
st.sidebar.header("⚙️ Model Controls")
selected_model_name = st.sidebar.selectbox(
    "Choose a regression model",
    list(assets['models'].keys())
)
model = assets['models'][selected_model_name]
st.header("Enter Details for Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    fan = st.number_input("Number of Fans", min_value=0, max_value=50, value=10, step=1)
    refrigerator = st.number_input("Number of Refrigerators", min_value=0, max_value=10, value=2, step=1)
    ac = st.number_input("Number of Air Conditioners", min_value=0, max_value=10, value=3, step=1)
    
with col2:
    tv = st.number_input("Number of Televisions", min_value=0, max_value=20, value=1, step=1)
    monitor = st.number_input("Number of Monitors", min_value=0, max_value=20, value=1, step=1)
    month = st.slider("Month of the Year", 1, 12, 6)

with col3:
    monthly_hours = st.number_input("Monthly Usage Hours", min_value=0, value=450)
    tariff_rate = st.number_input("Tariff Rate (per unit)", min_value=0.0, value=8.5, format="%.2f")
    city = st.selectbox("City", options=CITIES)
    company = st.selectbox("Company", options=COMPANIES)


# --- Prediction Logic ---
if st.button("Predict Electricity Bill", type="primary"):
    # 1. Create a DataFrame from user inputs with the correct column structure
    input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    
    # 2. Populate numerical features
    input_data['Fan'] = fan
    input_data['Refrigerator'] = refrigerator
    input_data['AirConditioner'] = ac
    input_data['Television'] = tv
    input_data['Monitor'] = monitor
    input_data['Month'] = month
    input_data['MonthlyHours'] = monthly_hours
    input_data['TariffRate'] = tariff_rate
    
    # 3. Handle one-hot encoded categorical features
    city_column = f"City_{city}"
    if city_column in model_columns:
        input_data[city_column] = 1
        
    company_column = f"Company_{company}"
    if company_column in model_columns:
        input_data[company_column] = 1
    for comp in model_columns:
        if comp != company_column:
            if comp!= city_column:
                input_data[comp] = 0
    prediction = model.predict(input_data)[0]

    # Display the result
    st.success(f"Predicted Monthly Electricity Bill:  **₹ {prediction:,.2f}**")