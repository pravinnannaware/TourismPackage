import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="wide")

model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

st.title("🌴 Wellness Tourism Package Prediction")
st.write("Predict whether a customer is likely to purchase the Wellness Tourism Package.")

st.sidebar.header("Customer Profile")

age = st.sidebar.number_input("Age", 18, 100, 35)
typeof_contact = st.sidebar.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.sidebar.number_input("Duration of Pitch (mins)", 1, 120, 15)
occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
num_persons = st.sidebar.number_input("Number of Persons Visiting", 1, 10, 2)
num_followups = st.sidebar.number_input("Number of Follow-ups", 1, 10, 3)
product_pitched = st.sidebar.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
preferred_star = st.sidebar.selectbox("Preferred Property Star Rating", [3.0, 4.0, 5.0])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
num_trips = st.sidebar.number_input("Average Annual Trips", 1, 20, 3)
passport = st.sidebar.selectbox("Holds Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
pitch_satisfaction = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.sidebar.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
num_children = st.sidebar.number_input("Children (<5 yrs) Visiting", 0, 5, 0)
designation = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.sidebar.number_input("Gross Monthly Income", 5000, 100000, 22000)

input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

st.subheader("Customer Details Summary")
st.dataframe(input_df)

if st.button("Predict Purchase Likelihood", type="primary"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    st.markdown("---")
    if prediction == 1:
        st.success(f"🎯 **High Likelihood to Purchase!** (Estimated Probability: {prob:.2%})")
    else:
        st.warning(f"⚠️ **Low Likelihood to Purchase.** (Estimated Probability: {prob:.2%})")
