import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Load all the pre-trained assets
@st.cache_resource
def load_models():
    """Loads all saved models and the scaler."""
    try:
        log_reg = joblib.load("logistic_churn_model.pkl")
        dt_model = joblib.load("decision_tree_churn_model.pkl")
        rf_model = joblib.load("random_forest_churn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return log_reg, dt_model, rf_model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'logistic_churn_model.pkl', 'decision_tree_churn_model.pkl', 'random_forest_churn_model.pkl', and 'scaler.pkl' are in the same directory.")
        return None, None, None, None

log_reg, dt_model, rf_model, scaler = load_models()

# --- Preprocessing Logic (recreated from the Jupyter Notebooks) ---
# Define the exact list of columns from the training data
# This list MUST match the columns from your X_train DataFrame in 03_modeling.ipynb
TRAINING_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    # Corrected engineered features (must be last to match saved scaler)
    'TotalServices', 'AvgCharges', 'TotalRevenue'
]

def preprocess_input(input_data):
    """
    Preprocesses raw user input into a format the model can understand.
    This function must replicate the steps from your 02_Preprocessing_and_featureEngg.ipynb file.
    """
    new_df = pd.DataFrame([input_data])
    
    # Manually create one-hot encoded columns and engineered features
    new_df['gender'] = new_df['gender'].map({'Male': 1, 'Female': 0})
    new_df['SeniorCitizen'] = new_df['SeniorCitizen'].map({'Yes': 1, 'No': 0})
    new_df['Partner'] = new_df['Partner'].map({'Yes': 1, 'No': 0})
    new_df['Dependents'] = new_df['Dependents'].map({'Yes': 1, 'No': 0})
    new_df['PhoneService'] = new_df['PhoneService'].map({'Yes': 1, 'No': 0})
    new_df['PaperlessBilling'] = new_df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    
    # One-hot encode the rest of the features.
    # Note: drop_first=True is important to match your preprocessing notebook.
    new_df = pd.get_dummies(new_df, columns=['MultipleLines', 'InternetService', 'OnlineSecurity',
                                             'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                             'StreamingTV', 'StreamingMovies', 'Contract',
                                             'PaymentMethod'], drop_first=True)

    # Reindex to ensure column order matches the training data
    # This is the most critical step for deployment
    final_df = new_df.reindex(columns=TRAINING_COLUMNS, fill_value=0)
    
    # Scale numerical features using the saved scaler
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'AvgCharges', 'TotalRevenue']
    final_df[numeric_cols] = scaler.transform(final_df[numeric_cols])
    
    return final_df

# --- Streamlit UI and Logic ---
st.title("Customer Churn Prediction Web App")
st.markdown("Enter customer details to get a churn prediction using our best-performing machine learning model.")

with st.form("churn_form"):
    st.header("Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        
    with col2:
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        
    with col3:
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

    st.header("Billing and Contract Information")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        
    with col5:
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        tenure = st.slider("Tenure (Months)", 0, 72, 1)
        
    with col6:
        monthly_charges = st.number_input("Monthly Charges", value=70.0, step=1.0)
        total_charges = st.number_input("Total Charges", value=2000.0, step=1.0)

    # Prediction button
    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Get user input in a format ready for preprocessing
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Preprocess the data
    processed_df = preprocess_input(input_data)
    
    # Make a prediction using the best-performing model (Random Forest)
    prediction = best_rf_model.predict(processed_df)[0]
    probability = best_rf_model.predict_proba(processed_df)[0][1]

    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error(f"The model predicts this customer is likely to **CHURN**.")
        st.write(f"The probability of churn is **{probability:.2%}**.")
    else:
        st.success(f"The model predicts this customer will **NOT CHURN**.")
        st.write(f"The probability of churn is **{probability:.2%}**.")
    
    st.markdown("""
        ---
        **What does this mean?**
        - **CHURN:** The customer is predicted to leave the company soon.
        - **NOT CHURN:** The customer is predicted to stay.
        
        **Important Factors for this prediction (from our analysis):**
        - **Tenure:** Shorter tenure is a strong indicator of churn.
        - **Monthly Charges:** Higher charges often increase the risk of churn.
        - **Contract Type:** Month-to-month contracts are highly correlated with churn.
        - **Internet Service:** "Fiber optic" users tend to have a higher churn risk.
    """)
