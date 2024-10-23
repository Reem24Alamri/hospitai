import streamlit as st
import requests

# Set up the page configuration
st.set_page_config(page_title="Patient Data Entry Form", page_icon=":hospital:", layout="centered")

st.write("""
Please fill in the required information for the patient below.
""")

st.title("Patient Prediction Form")

# Create a form for user input
with st.form(key="user_form"):
    # Collecting inputs for the new set of important features
    age = st.number_input("Age", min_value=0, max_value=120)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    type_of_admission = st.selectbox("Type of Admission", options=["EMERGENCY", "OPD"])

    urea = st.number_input("Urea (mg/dL)", min_value=0.0, max_value=200.0)
    tlc = st.number_input("Total Leukocyte Count (TLC)", min_value=0.0, max_value=100.0)
    stable_angina = st.selectbox("Stable Angina", options=["Yes", "No"])
    acs = st.selectbox("Acute Coronary Syndrome (ACS)", options=["Yes", "No"])

    creatinine = st.number_input("Creatinine ", min_value=0.0, max_value=20.0)
    atypical_chest_pain = st.selectbox("Atypical Chest Pain", options=["Yes", "No"])
    anaemia = st.selectbox("Anaemia", options=["Yes", "No"])
    stemi = st.selectbox("ST-Elevation Myocardial Infarction (STEMI)", options=["Yes", "No"])

    glucose = st.number_input("Glucose", min_value=0.0, max_value=600.0)
    hb = st.number_input("Hemoglobin (HB)", min_value=0.0, max_value=20.0)

    submit_button = st.form_submit_button("Submit")

# Collect user data when the form is submitted
if submit_button:
    # Prepare the API URL with the correct capitalization
    api_url = f'http://127.0.0.1:8000/predict?AGE={age}&Gender={gender}&TYPE_OF_ADMISSION={type_of_admission}&UREA={urea}&TLC={tlc}&STABLE_ANGINA={stable_angina}&ACS={acs}&CREATININE={creatinine}&ATYPICAL_CHEST_PAIN={atypical_chest_pain}&ANAEMIA={anaemia}&STEMI={stemi}&GLUCOSE={glucose}&HB={hb}'

    # Send the request to the FastAPI backend
    response = requests.get(api_url)

# Process the response
if response.status_code == 200:
    prediction = response.json()

    # Strip brackets if present and convert to float
    prediction_value = prediction['Prediction'].strip('[]')

    # Round to the nearest whole number (no decimal places)
    rounded_prediction = round(float(prediction_value))  # Round to nearest whole number

    st.header(f"The Stay Duration is {rounded_prediction} days")  # Display the rounded prediction without decimals
else:
    st.error(f"Error in fetching prediction. Status code: {response.status_code}, Response: {response.text}")
