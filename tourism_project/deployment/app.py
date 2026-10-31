import streamlit as st
import pandas as pd
import joblib
import os

# Robust path resolution to locate the model in Streamlit Community Cloud
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')
model = joblib.load(model_path)

# Build the Streamlit interface
st.set_page_config(page_title="Wellness Tourism Predictor", layout="centered")
st.title("Wellness Tourism Package Prediction")
st.write("Enter customer details to predict the likelihood of package purchase.")

# Create two columns for better UI layout
col1, col2 = st.columns(2)

# Column 1 inputs
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=25000)

# Column 2 inputs
with col2:
    typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    passport = st.selectbox("Passport (0: No, 1: Yes)", [0, 1])
    own_car = st.selectbox("Own Car (0: No, 1: Yes)", [0, 1])
    num_trips = st.number_input("Number of Trips", min_value=1, max_value=20, value=2)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    num_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=3)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=5, max_value=120, value=15)

# Prediction execution block
if st.button("Predict Package Purchase"):
    # Map the UI inputs to a dictionary matching the exact column names of Xtrain
    input_data = {
        'Age': [age],
        'TypeofContact': [typeof_contact],
        'CityTier': [city_tier],
        'DurationOfPitch': [duration_of_pitch],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [num_person_visiting],
        'NumberOfFollowups': [num_followups],
        'ProductPitched': [product_pitched],
        'PreferredPropertyStar': [preferred_property_star],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [num_trips],
        'Passport': [passport],
        'PitchSatisfactionScore': [pitch_satisfaction_score],
        'OwnCar': [own_car],
        'NumberOfChildrenVisiting': [num_children_visiting],
        'Designation': [designation],
        'MonthlyIncome': [monthly_income]
    }
    
    try:
        # Convert dictionary to DataFrame for the model
        df = pd.DataFrame(input_data)
        
        # Execute prediction
        prediction = model.predict(df)
        
        # Display results
        if prediction[0] == 1:
            st.success("High Probability: The customer is likely to purchase the Wellness Tourism Package!")
            st.balloons()
        else:
            st.warning("Low Probability: The customer is unlikely to purchase the package.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Check if all input features match the training dataset schema perfectly.")
