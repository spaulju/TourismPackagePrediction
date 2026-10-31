
import os
import joblib
import pandas as pd
import streamlit as st

# Configure the Streamlit application
st.set_page_config(
    page_title="Wellness Tourism | AI Predictor",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom styling for a premium travel-themed interface
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f0f9ff 0%, #f8fafc 50%, #ecfdf5 100%);
        color: #172b4d;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(115deg, #0f766e, #0891b2, #2563eb);
        border-radius: 24px;
        padding: 42px 36px;
        color: white;
        text-align: center;
        margin-bottom: 28px;
        box-shadow: 0 12px 32px rgba(8,145,178,0.18);
    }

    .hero h1 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .hero p {
        color: #e0f2fe;
        font-size: 1.08rem;
        margin: 0;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        color: #0f766e;
        margin: 22px 0 12px;
        padding: 12px 16px;
        background: rgba(13,148,136,0.08);
        border-left: 5px solid #0d9488;
        border-radius: 8px;
    }

    .info-banner {
        padding: 16px 20px;
        background: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        margin-bottom: 24px;
        line-height: 1.6;
    }

    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.93);
        padding: 26px;
        border: 1px solid #dbeafe;
        border-radius: 20px;
        box-shadow: 0 6px 28px rgba(15,23,42,0.05);
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 9px;
    }

    div[data-testid="stSelectbox"] > div {
        border-radius: 9px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #0f766e, #0284c7);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        min-height: 52px;
        width: 100%;
        font-size: 1.05rem;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(90deg, #115e59, #0369a1);
        color: white;
        border: none;
    }

    .result-success {
        background: linear-gradient(120deg, #d1fae5, #ecfdf5);
        color: #065f46;
        padding: 28px;
        border-radius: 18px;
        border: 1px solid #6ee7b7;
        text-align: center;
        margin-top: 20px;
    }

    .result-warning {
        background: linear-gradient(120deg, #fef3c7, #fff7ed);
        color: #92400e;
        padding: 28px;
        border-radius: 18px;
        border: 1px solid #fcd34d;
        text-align: center;
        margin-top: 20px;
    }

    .result-success h2, .result-warning h2 {
        color: inherit !important;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 28px 10px;
        margin-top: 32px;
        border-top: 1px solid #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# Display the application header
st.markdown("""
<div class="hero">
    <h1>🌴 Wellness Tourism AI Predictor</h1>
    <p>Discover customer travel preferences and predict tourism package purchases using Machine Learning.</p>
    <p style="margin-top:12px;">✈️ Smart Travel Insights &nbsp;|&nbsp; 🧠 AI-Powered Predictions &nbsp;|&nbsp; 🌿 Wellness Tourism</p>
</div>
""", unsafe_allow_html=True)

# Display introductory instructions
st.markdown("""
<div class="info-banner">
    <strong>📋 Customer Prediction Form</strong><br>
    Complete the customer profile below and click
    <strong>Predict Purchase Likelihood</strong> to generate an AI-powered prediction.
    All numerical inputs use whole numbers only.
</div>
""", unsafe_allow_html=True)

# Define the trained model location
model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")

# Cache the trained model for efficient application reruns
@st.cache_resource
def load_model():
    return joblib.load(model_path)

# Load the trained model and handle loading errors
try:
    model = load_model()
except Exception as e:
    st.error(
        "Failed to load model. Please wait for GitHub Actions "
        f"to finish training. Error: {e}"
    )
    st.stop()

# Organize all customer inputs inside a single form
with st.form("tourism_prediction_form"):

    # Section 1: Customer demographic details
    st.markdown(
        '<div class="section-title">👤 Customer Profile</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        age = st.number_input(
            "🎂 Customer Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

        gender = st.selectbox(
            "👤 Gender",
            ["Male", "Female"]
        )

    with col2:
        marital_status = st.selectbox(
            "💍 Marital Status",
            ["Single", "Married", "Divorced"]
        )

        occupation = st.selectbox(
            "💼 Occupation",
            ["Salaried", "Small Business", "Large Business", "Free Lancer"]
        )

    with col3:
        designation = st.selectbox(
            "🏢 Designation",
            ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )

        monthly_income = st.number_input(
            "💰 Monthly Income",
            min_value=1000,
            max_value=500000,
            value=25000,
            step=1000
        )

    # Section 2: Customer travel preferences
    st.markdown(
        '<div class="section-title">✈️ Travel Preferences</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        city_tier = st.selectbox(
            "🏙️ City Tier",
            [1, 2, 3]
        )

        num_person_visiting = st.number_input(
            "👨‍👩‍👧 Number of Persons Visiting",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

        num_children_visiting = st.number_input(
            "🧒 Children Visiting",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

    with col2:
        preferred_property_star = st.selectbox(
            "⭐ Preferred Property Rating",
            [3, 4, 5]
        )

        num_trips = st.number_input(
            "🧳 Average Annual Trips",
            min_value=1,
            max_value=25,
            value=3,
            step=1
        )

        passport = st.selectbox(
            "🛂 Holds Valid Passport?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col3:
        own_car = st.selectbox(
            "🚗 Owns a Car?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        product_pitched = st.selectbox(
            "🏝️ Product Pitched",
            ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
        )

    # Section 3: Sales interaction details
    st.markdown(
        '<div class="section-title">📊 Sales & Interaction Details</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        typeof_contact = st.selectbox(
            "📞 Type of Contact",
            ["Self Enquiry", "Company Invited"]
        )

        duration_of_pitch = st.number_input(
            "⏱️ Duration of Pitch (minutes)",
            min_value=1,
            max_value=120,
            value=15,
            step=1
        )

    with col2:
        num_followups = st.number_input(
            "🔁 Number of Follow-ups",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )

    with col3:
        pitch_satisfaction_score = st.slider(
            "🌟 Pitch Satisfaction Score",
            min_value=1,
            max_value=5,
            value=3,
            step=1
        )

    # Display the prediction submission button
    st.markdown("---")

    submitted = st.form_submit_button(
        "🔮 Predict Purchase Likelihood",
        use_container_width=True
    )

# Perform model inference after form submission
if submitted:

    # Construct a single-row DataFrame with training-compatible feature names
    # Keep original numeric type conversions for model compatibility
    input_df = pd.DataFrame({
        'Age': [float(age)],
        'TypeofContact': [typeof_contact],
        'CityTier': [int(city_tier)],
        'DurationOfPitch': [float(duration_of_pitch)],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [int(num_person_visiting)],
        'NumberOfFollowups': [float(num_followups)],
        'ProductPitched': [product_pitched],
        'PreferredPropertyStar': [float(preferred_property_star)],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [float(num_trips)],
        'Passport': [int(passport)],
        'PitchSatisfactionScore': [int(pitch_satisfaction_score)],
        'OwnCar': [int(own_car)],
        'NumberOfChildrenVisiting': [float(num_children_visiting)],
        'Designation': [designation],
        'MonthlyIncome': [float(monthly_income)]
    })

    # Generate predictions using the trained classification pipeline
    try:
        prediction = model.predict(input_df)[0]

        # Display the prediction results in a dedicated dashboard section
        st.markdown(
            '<div class="section-title">🎯 AI Prediction Result</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:
            # Display the positive purchase prediction
            st.markdown("""
            <div class="result-success">
                <h2>🎉 High Purchase Potential</h2>
                <h3>Customer WILL Purchase the Package</h3>
                <p>The AI model predicts a positive purchase outcome
                based on the customer profile and travel preferences.</p>
            </div>
            """, unsafe_allow_html=True)

            st.balloons()

        else:
            # Display the negative purchase prediction
            st.markdown("""
            <div class="result-warning">
                <h2>📋 Low Purchase Potential</h2>
                <h3>Customer Will NOT Purchase the Package</h3>
                <p>The AI model predicts a negative purchase outcome
                based on the information provided.</p>
            </div>
            """, unsafe_allow_html=True)

        # Display a summary of the submitted customer profile
        st.markdown("### 📌 Customer Prediction Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.metric("Customer Age", f"{age} Years")

        with summary_col2:
            st.metric("Package Type", product_pitched)

        with summary_col3:
            st.metric(
                "Predicted Outcome",
                "Will Purchase" if prediction == 1 else "Will Not Purchase"
            )

        # Allow the user to inspect the prediction input data
        with st.expander("🔍 View Submitted Customer Data"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        # Handle unexpected errors during prediction
        st.error(f"Prediction failed: {e}")

# Display the application footer
st.markdown("""
<div class="footer">
    🌴 Wellness Tourism AI Predictor<br>
    Powered by Machine Learning • Random Forest • Streamlit<br>
    <strong>Intelligent Tourism Decision Support</strong>
</div>
""", unsafe_allow_html=True)
