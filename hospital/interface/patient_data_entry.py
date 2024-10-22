import streamlit as st
import requests

# Set up the page configuration
st.set_page_config(page_title="Patient Data Entry Form", page_icon=":hospital:", layout="centered")

st.write("""
Please fill in the required information for the patient below.
""")

st.title("Customer Prediction Form")

# Create a form for user input
with st.form(key="user_form"):
    age = st.number_input("Age", min_value=0, max_value=120)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    TYPE_OF_ADMISSION = st.selectbox("Type of Admission", options=["EMERGENCY", "OPD"])
    SMOKING = st.selectbox("Smoking", options=["Yes", "No"])
    DM = st.selectbox("Does the patient have Diabetes (DM)?", options=["Yes", "No"])
    HTN = st.selectbox("Does the patient have Hypertension (HTN)?", options=["Yes", "No"])
    SEVERE_ANAEMIA = st.selectbox("Does the patient have Severe Anemia?", options=["Yes", "No"])

    st.subheader(":octagonal_sign: Critical Medical Conditions")

    HEART_FAILURE = st.selectbox("Does the patient have Heart Failure?", options=["Yes", "No"])
    AKI = st.selectbox("Does the patient have Acute Kidney Injury (AKI)?", options=["Yes", "No"])
    CVA_INFRACT = st.selectbox("Does the patient have a Cerebral Infarct (CVA)?", options=["Yes", "No"])
    CARDIOGENIC_SHOCK = st.selectbox("Does the patient have Cardiogenic Shock?", options=["Yes", "No"])
    PULMONARY_EMBOLISM = st.selectbox("Does the patient have Pulmonary Embolism?", options=["Yes", "No"])

    submit_button = st.form_submit_button("Submit")

# Collect user data when the form is submitted
if submit_button:
    user_data = {
        "age": age,
        "gender": gender,
        "type_of_admission": TYPE_OF_ADMISSION,
        "smoking": SMOKING,
        "diabetes": DM,
        "hypertension": HTN,
        "severe_anaemia": SEVERE_ANAEMIA,
        "heart_failure": HEART_FAILURE,
        "aki": AKI,
        "cva_infarct": CVA_INFRACT,
        "cardiogenic_shock": CARDIOGENIC_SHOCK,
        "pulmonary_embolism": PULMONARY_EMBOLISM,
    }

    # Send the user data to the FastAPI backend
    api_url = 'http://127.0.0.1:8000/testing'
    response = requests.post(api_url, json=user_data)

    # Process the response
    if response.status_code == 200:
        prediction = response.json()
        st.header(f"Prediction Result: {prediction}")
    else:
        st.error("Error in fetching prediction. Please check the API connection.")
