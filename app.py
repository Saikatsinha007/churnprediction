
import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define feature inputs for manual prediction
def get_user_input():
    st.sidebar.header("User Input Features")
    age = st.sidebar.slider("Age", 18, 100, 35)
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 24)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 1200.0)
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    payment_method = st.sidebar.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber Optic", "None"])
    
    # Convert categorical features to numerical values
    contract_mapping = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
    payment_mapping = {"Electronic Check": 0, "Mailed Check": 1, "Bank Transfer": 2, "Credit Card": 3}
    internet_mapping = {"DSL": 0, "Fiber Optic": 1, "None": 2}
    
    data = {
        "Age": age,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract_mapping[contract],
        "PaymentMethod": payment_mapping[payment_method],
        "InternetService": internet_mapping[internet_service],
    }
    return pd.DataFrame([data])

# Streamlit UI
def main():
    st.title("Customer Churn Prediction App")
    st.write("This app predicts whether a customer will churn based on input features.")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())
        
        # Ensure the correct columns are used for prediction
        if set(df.columns) >= set(["Age", "Tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod", "InternetService"]):
            # Map categorical variables
            contract_mapping = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
            payment_mapping = {"Electronic Check": 0, "Mailed Check": 1, "Bank Transfer": 2, "Credit Card": 3}
            internet_mapping = {"DSL": 0, "Fiber Optic": 1, "None": 2}
            df["Contract"] = df["Contract"].map(contract_mapping)
            df["PaymentMethod"] = df["PaymentMethod"].map(payment_mapping)
            df["InternetService"] = df["InternetService"].map(internet_mapping)
            
            predictions = model.predict(df)
            df["Churn Prediction"] = predictions
            
            st.write("Prediction Results:")
            st.dataframe(df)
        else:
            st.error("CSV does not contain the required columns!")
    
    # Manual Input Prediction
    st.write("Or, enter details manually:")
    user_input = get_user_input()
    
    if st.button("Predict Churn"):
        prediction = model.predict(user_input)[0]
        result = "Churn" if prediction == 1 else "Not Churn"
        st.write(f"## Prediction: {result}")

if __name__ == "__main__":
    main()
