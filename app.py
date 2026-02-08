import streamlit as st
import pickle
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
# ১. মডেল লোড করা
try:
    with open('churn_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("মডেল ফাইলটি পাওয়া যায়নি! আগে 'churn_rf_model.pkl' সেভ করো।")

# ২. অ্যাপের ইন্টারফেস
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("Telecom Churn Predictor 📊")

# ৩. ইনপুট বক্স
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 18, 100, 30)
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.number_input("Number of Dependents", 0, 10, 0)
    referrals = st.number_input("Number of Referrals", 0, 20, 0)

with col2:
    tenure = st.number_input("Tenure in Months", 0, 100, 12)
    offer = st.selectbox("Offer", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    long_dist = st.number_input("Avg Long Distance Charges", 0.0, 100.0, 25.0)
    multiple = st.selectbox("Multiple Lines", ["Yes", "No"])

# ৪. প্রেডিকশন বাটন
if st.button("Analyze Customer"):
    # এনকোডিং ম্যাপিং (মডেল ট্রেইনিং এর Label Encoding অনুযায়ী)
    gen_val = 1 if gender == "Male" else 0
    mar_val = 1 if married == "Yes" else 0
    phone_val = 1 if phone == "Yes" else 0
    mult_val = 1 if multiple == "Yes" else 0
    
    # Offer ম্যাপিং
    offer_dict = {"None": 0, "Offer A": 1, "Offer B": 2, "Offer C": 3, "Offer D": 4, "Offer E": 5}
    offer_val = offer_dict[offer]
    
    # ইনপুট অ্যারে (সিরিয়াল অনুযায়ী ১০টি ফিচার)
    input_data = np.array([[gen_val, age, mar_val, dependents, referrals, tenure, offer_val, phone_val, long_dist, mult_val]])
    
    # প্রেডিকশন
    prediction = model.predict(input_data)
    
    st.divider()
    if prediction[0] == 1:
        st.error("### ⚠️ সতর্কবার্তা: কাস্টমারটি চলে যাওয়ার (Churn) ঝুঁকিতে আছে!")
    else:
        st.success("### ✅ অভিনন্দন: কাস্টমারটি আমাদের সাথেই থাকছে।")
