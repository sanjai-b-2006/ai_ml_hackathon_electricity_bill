# ai_ml_hackathon_electricity_bill

⚡ Electricity Bill Prediction Model – Summary
I developed a supervised regression model to predict Electricity Bill using:

✅ Features:

- Appliance counts (Fan, Refrigerator, Air Conditioner, Television, Monitor)

- Monthly usage hours

- Tariff rate

- City and Company

✅ Approach:

Trained two Linear Regression models:

- One with all features (including Monthly Usage Hours and Tariff Rate) for higher precision.

- Another without Monthly Usage Hours and Tariff Rate to test lower precision scenarios.

- In misc i have given models on ridge , elastic net , regression decision tree regression and gradient boost.

Evaluated models using RMSE and R² on the dataset for accuracy comparison.

✅ Deployment:

Built an interactive Streamlit app where users can:

Select the regression model to compare.

Input appliance counts, usage hours, tariff rate, city, and company.

Instantly view predicted electricity bills.
<img width="1918" height="1002" alt="Screenshot 2025-07-11 141859" src="https://github.com/user-attachments/assets/bd0b4c47-6289-4a98-8c7d-2046e07b6c11" />
<img width="1919" height="995" alt="Screenshot 2025-07-11 141837" src="https://github.com/user-attachments/assets/3e8e0829-bee9-4c32-aa3a-701b919ef124" />
As shown in the images:

The app displays predicted bill values based on user inputs and the selected model.

Helps users understand how different features impact electricity bills while visually comparing model outputs.
