import streamlit as st
import pandas as pd
import joblib

# Load the model committed to the repo
model = joblib.load('tourism_project/deployment/best_model.pkl')

st.title("Wellness Tourism Package Prediction")
st.write("Enter customer details to predict if they will purchase the package.")

# Create input fields for ALL features used in training
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=20000)

with col2:
    typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    passport = st.selectbox("Passport (0: No, 1: Yes)", [0, 1])
    own_car = st.selectbox("Own Car (0: No, 1: Yes)", [0, 1])
    num_trips = st.number_input("Number of Trips", min_value=1, max_value=2
