import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Electricity Bill Forecaster", layout="wide")

# --- Load all assets using Streamlit's cache ---
@st.cache_resource
def load_assets():
    assets = {}
    try:
        assets['model_full'] = pickle.load(open('linear_regression.pkl', 'rb'))
        assets['model_less'] = pickle.load(open('linear_regression_without.pkl', 'rb'))
        assets['scaler_full'] = pickle.load(open('scaler.pkl', 'rb'))
        assets['scaler_less'] = pickle.load(open('scaler_without.pkl', 'rb'))
        assets['columns_full'] = pickle.load(open('model_columns.pkl', 'rb'))
        assets['columns_less'] = pickle.load(open('model_columns_without.pkl', 'rb'))
        assets['app_data'] = pickle.load(open('app_data.pkl', 'rb'))
    except FileNotFoundError as e:
        st.error(f"Error loading asset file: {e}. Please run `train_models.py` first.")
        return None
    return assets

assets = load_assets()
if not assets:
    st.stop()

# --- App Layout ---
st.title("💡 Electricity Bill Model Comparator")

# --- Sidebar for Model Selection ---
st.sidebar.header("⚙️ Model Selection")
model_options = ["Linear Regression", "Linear Regression without 100 percent precision(Without Monthly Usage Hours and Tariff Rate)"]
selected_model_name = st.sidebar.selectbox("Choose a model", model_options)

# --- Display Performance Metrics in Sidebar ---
st.sidebar.markdown("---")
st.sidebar.header("📊 Model Performance")

# Determine which model and data to use for evaluation
if selected_model_name == "Linear Regression":
    model_eval = assets['model_full']
    X_test_scaled_eval = assets['scaler_full']
else: # "without 100 percent precision"
    model_eval = assets['model_less']
    X_test_scaled_eval = assets['scaler_less']


# --- Main Section for User Input ---
st.header(f"Enter Details for: {selected_model_name}")

# Conditionally show/hide input fields
is_full_model = (selected_model_name == "Linear Regression")
if not is_full_model:
    st.info("This model was trained without 'Monthly Usage Hours' and 'Tariff Rate' for lower precision.")

col1, col2, col3 = st.columns(3)
with col1:
    fan = st.number_input("Number of Fans", 0, 50, 10)
    refrigerator = st.number_input("Number of Refrigerators", 0, 50, 2)
    ac = st.number_input("Number of Air Conditioners", 0, 50, 3)
with col2:
    tv = st.number_input("Number of Televisions", 0, 50, 1)
    monitor = st.number_input("Number of Monitors", 0, 50, 1)
    month = st.slider("Month of the Year", 1, 12, 6)
with col3:
    monthly_hours = st.number_input("Monthly Usage Hours", 0, 1000, 450, disabled=not is_full_model)
    tariff_rate = st.number_input("Tariff Rate (per unit)", 0.0, 20.0, 8.5, format="%.2f", disabled=not is_full_model)
    city = st.selectbox("City", options=assets['app_data']['cities'])
    company = st.selectbox("Company", options=assets['app_data']['companies'])

# --- Prediction Logic ---
if st.button("Predict Electricity Bill", type="primary"):
    # Select the correct tools based on user choice
    if is_full_model:
        model_columns = assets['columns_full']
        scaler = assets['scaler_full']
        model = assets['model_full']
    else:
        model_columns = assets['columns_less']
        scaler = assets['scaler_less']
        model = assets['model_less']

    # Create the input DataFrame
    input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

    # Populate the DataFrame with user inputs
    input_data['Fan'] = fan
    input_data['Refrigerator'] = refrigerator
    input_data['AirConditioner'] = ac
    input_data['Television'] = tv
    input_data['Monitor'] = monitor
    input_data['Month'] = month
    
    if is_full_model:
        input_data['MonthlyHours'] = monthly_hours
        input_data['TariffRate'] = tariff_rate
        input_data['Usage_Tariff_Interaction'] = tariff_rate* monthly_hours

    city_col = f"City_{city}"
    if city_col in model_columns:
        input_data[city_col] = 1
    company_col = f"Company_{company}"
    if company_col in model_columns:
        input_data[company_col] = 1

    # Scale and predict
    input_data = input_data[[col for col in scaler.feature_names_in_]]
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]


    st.success(f"Predicted Bill ({selected_model_name}):  **₹ {prediction:,.2f}**")