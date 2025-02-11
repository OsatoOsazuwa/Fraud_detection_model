import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# Loading the label encoders, scaler, autoencoder, and classifier
label_encoders = joblib.load("label_encoders_fixed.pkl")
scaler = joblib.load("scaler_fixed.pkl")
encoder = tf.keras.models.load_model("encoder.keras")
classifier = tf.keras.models.load_model("fraud_classification_09_0.819.keras")  # Loading trained model

# Reading the feature names from the file "features.txt"
with open("features.txt", "r") as file:
    feature_names = [line.strip() for line in file.readlines()]  # Read each line and strip newline characters

# Get median values from the scaler (use mean if median is unavailable)
median_values = dict(zip(feature_names, scaler.mean_))  

# Streamlit UI
st.title("🛡️ Fraud Detection System")
st.write("🔍 Enter transaction details manually or upload a CSV file.")

 # Add a slider for threshold adjustment
threshold = st.slider("Set Fraud Probability Threshold", 0.1, 0.9, 0.5, step=0.05)

# Sidebar information
st.sidebar.title("🗂️ Navigation")
if st.sidebar.button("📌 About Model"):
    st.sidebar.write("🤖 This fraud detection system utilizes deep learning techniques to analyze transaction data and predict fraudulent activities.")

# Help Section
with st.expander("📝 Feature Descriptions"):
    st.write("**ProductCD**: Type of payment method used in the transaction.")
    st.write("**card4**: 💳 Network associated with the payment card.")
    st.write("**card6**: Type of payment card (Credit or Debit).")
    st.write("**P_emaildomain**: 📧 Email domain associated with the transaction.")
    st.write("**M1-M9**: Binary indicators for various transaction characteristics.")
    st.write("**M4**: Matching status of transaction details (No Match, Partial Match, Full Match).")

# Predefined categories with explanations
product_cd_options = {"W": "Wallet Payment", "H": "Home Payment", "C": "Card Payment", "S": "Store Payment", "R": "Recurring Payment"}
card4_options = {"discover": "Discover Network", "mastercard": "Mastercard", "visa": "Visa", "american express": "American Express"}
card6_options = {"credit": "Credit Card", "debit": "Debit Card"}
p_emaildomain_options = [
    "gmail.com", "outlook.com", "yahoo.com", "mail.com", "anonymous.com",
    "hotmail.com", "verizon.net", "aol.com", "me.com", "comcast.net",
    "optonline.net", "cox.net", "charter.net", "rocketmail.com", "prodigy.net.mx",
    "embarqmail.com", "icloud.com", "live.com.mx", "gmail", "live.com",
    "att.net", "juno.com", "ymail.com", "sbcglobal.net", "bellsouth.net",
    "msn.com", "q.com", "yahoo.com.mx", "centurylink.net", "servicios-ta.com"
]
m_options = {"T": "True (Transaction characteristic present)", "F": "False (Transaction characteristic absent)"}
m4_options = {"M0": "No Matching", "M1": "Partial Matching", "M2": "Full Matching"}

# User selection: Manual Input or CSV Upload
option = st.radio("Choose Input Method", ("Manual Input", "📂 Upload CSV"))

if option == "Manual Input":
    # Input fields for each feature
    transaction_amt = st.number_input("Transaction Amount", min_value=0.0)
    product_cd = st.selectbox("ProductCD", options=list(product_cd_options.keys()))
    card4 = st.selectbox("Card Network (card4)", options=list(card4_options.keys()))
    card6 = st.selectbox("Card Type (card6)", options=list(card6_options.keys()))
    p_emaildomain = st.selectbox("P_emaildomain", options=p_emaildomain_options)
    
    # M1-M6 and M4 are binary selections
    m1 = st.selectbox("M1", options=list(m_options.keys()))
    m2 = st.selectbox("M2", options=list(m_options.keys()))
    m3 = st.selectbox("M3", options=list(m_options.keys()))
    m4 = st.selectbox("M4", options=list(m4_options.keys()))
    m6 = st.selectbox("M6", options=list(m_options.keys()))
   
    
    # Prepare the input data for prediction
    input_data = {
        "TransactionAmt": [transaction_amt],
        "ProductCD": [product_cd],
        "card4": [card4],
        "card6": [card6],
        "P_emaildomain": [p_emaildomain],
        "M1": [m1],
        "M2": [m2],
        "M3": [m3],
        "M4": [m4],
        "M6": [m6],
    }

    # Convert the input data into a DataFrame
    df_input = pd.DataFrame(input_data)

    # Apply label encoding and missing values handling
    for col in ["ProductCD", "card4", "card6", "P_emaildomain", "M1", "M2", "M3", "M4", "M6"]:
        if col in df_input.columns:
            df_input[col] = label_encoders[col].transform(df_input[col])

    for feature in feature_names:
        if feature not in df_input.columns:
            df_input[feature] = median_values.get(feature, 0)

    # Scale the input and make predictions
    input_scaled = scaler.transform(df_input[feature_names]) 
    input_encoded = encoder.predict(input_scaled)  # Reduces to 32 features
    fraud_probability = classifier.predict(input_encoded)[0][0]



    if fraud_probability > threshold:
        st.error(f"🚨 Fraudulent Transaction Detected! 🚨\n\n**Fraud Probability: {fraud_probability:.2%}**")
    else:
        st.success(f"✅ Legitimate Transaction\n\n**Fraud Probability: {fraud_probability:.2%}**")

else:
    # CSV Upload Section
    st.write("Download the template CSV file and upload your transaction data.")
    template_data = pd.DataFrame(columns=["TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain", 
                                          "M1", "M2", "M3", "M4", "M6"])
    st.download_button("Download Template", template_data.to_csv(index=False), "template.csv")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        for col in ["ProductCD", "card4", "card6", "P_emaildomain", "M1", "M2", "M3", "M4", "M6"]:
            if col in df.columns:
                df[col] = df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)
        
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = median_values.get(feature, 0)
        
        input_scaled = scaler.transform(df[feature_names])
        input_encoded = encoder.predict(input_scaled)
        predictions = classifier.predict(input_encoded)

        # Add results to dataframe
        df["Fraud Probability"] = predictions
        df["Prediction"] = df["Fraud Probability"].apply(
            lambda x: "Fraudulent" if x > threshold else "Legitimate")

        # Display results
        st.write("📊 Prediction Results")
        st.dataframe(df[["Fraud Probability", "Prediction"]])
        st.download_button("📥 Download Results", df.to_csv(index=False), "fraud_predictions.csv")

# Footer information
st.markdown("---")
st.markdown("👨‍💻 Developed by **Osato Osazuwa**")
st.markdown("📧 Contact: [osato.osazuwa@gmail.com](mailto:osato.osazuwa@gmail.com)")
