import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wellness Tourism Predictor")
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')

@st.cache_resource
def load_model():
    return joblib.load(model_path)

st.title("Wellness Tourism Package Prediction")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model. Please wait for GitHub Actions to finish training. Error: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, max_value=120.0, value=15.0)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_followups = st.number_input("Number of Follow-ups", min_value=1.0, max_value=10.0, value=3.0)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

with col2:
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_trips = st.number_input("Average Annual Trips", min_value=1.0, max_value=25.0, value=3.0)
    passport = st.selectbox("Holds Valid Passport?", options=[0, 1])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    own_car = st.selectbox("Owns a Car?", options=[0, 1])
    num_children_visiting = st.number_input("Children Visiting", min_value=0.0, max_value=10.0, value=0.0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=500000.0, value=25000.0)

if st.button("Predict Purchase Likelihood"):
    input_df = pd.DataFrame({
        'Age': [float(age)], 'TypeofContact': [typeof_contact], 'CityTier': [int(city_tier)],
        'DurationOfPitch': [float(duration_of_pitch)], 'Occupation': [occupation], 'Gender': [gender],
        'NumberOfPersonVisiting': [int(num_person_visiting)], 'NumberOfFollowups': [float(num_followups)],
        'ProductPitched': [product_pitched], 'PreferredPropertyStar': [float(preferred_property_star)],
        'MaritalStatus': [marital_status], 'NumberOfTrips': [float(num_trips)], 'Passport': [int(passport)],
        'PitchSatisfactionScore': [int(pitch_satisfaction_score)], 'OwnCar': [int(own_car)],
        'NumberOfChildrenVisiting': [float(num_children_visiting)], 'Designation': [designation],
        'MonthlyIncome': [float(monthly_income)]
    })

    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("Prediction: Customer WILL purchase the package!")
        st.balloons()
    else:
        st.warning("Prediction: Customer will NOT purchase the package.")
