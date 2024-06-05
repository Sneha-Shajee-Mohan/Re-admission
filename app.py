import os
import streamlit as st
from dotenv import load_dotenv
from utils.b2 import B2
import pandas as pd
from joblib import load
import numpy as np



# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
REMOTE_DATA = 'training_data.csv'


# ------------------------------------------------------
#                        CONFIG
# ------------------------------------------------------
load_dotenv()

# load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_KEYID'],
        secret_key=os.environ['B2_APPKEY'])

# ------------------------------------------------------
#                        CACHING
# ------------------------------------------------------
@st.cache_data
def get_data():
#     # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df = b2.get_df(REMOTE_DATA)
    
    return df



# ------------------------------------------------------
#                         APP
# ------------------------------------------------------
# ------------------------------
# PART 0 : Overview
# ------------------------------
st.title('Patients Re-admission Prediction')

df_base = get_data()

# Load the trained model and label encoder
model = load('decision_tree_model.joblib')
label_encoder = load('label_encoder.joblib')


# Create input fields
st.subheader("Enter you details")
admission_type_id = st.selectbox("Select your admission type", sorted(df_base['admission_type_id'].unique()))
discharge_disposition_id = st.selectbox("Select your admission type", sorted(df_base['discharge_disposition_id'].unique()))
time_in_hospital = st.number_input("Enter your time in Hospital", min_value=1,max_value=14)
medical_specialty = st.selectbox("Select your medical speciality", sorted(df_base['medical_specialty'].unique()))
num_procedures = st.number_input("Number of Procedures", min_value=0,max_value=6)
num_medications = st.number_input("Number of Medications", min_value=1,max_value=81)
number_outpatient = st.number_input("Number of Outpatient", min_value=0,max_value=42)
number_emergency = st.number_input("Number of Emergency", min_value=0,max_value=76)
number_inpatient = st.number_input("Number of Inpatient", min_value=0,max_value=21)
diag_1 = st.selectbox("Select your primary diagnosis", sorted(df_base['diag_1'].unique()))
number_diagnoses = st.number_input("Number of Diagnoses", min_value=1,max_value=16)
change = st.selectbox("Change", [0, 1])

# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    'admission_type_id': [admission_type_id],
    'discharge_disposition_id': [discharge_disposition_id],
    'time_in_hospital': [time_in_hospital],
    'medical_specialty': [medical_specialty],
    'num_procedures': [num_procedures],
    'num_medications': [num_medications],
    'number_outpatient': [number_outpatient],
    'number_emergency': [number_emergency],
    'number_inpatient': [number_inpatient],
    'diag_1': [diag_1],
    'number_diagnoses': [number_diagnoses],
    'change': [change]
})

# Make predictions
if st.button("Predict"):
    # Load label encoder and transform categorical variables
    input_data['admission_type_id'] = label_encoder.fit_transform(input_data['admission_type_id'])
    input_data['discharge_disposition_id'] = label_encoder.fit_transform(input_data['discharge_disposition_id'])
    input_data['medical_specialty'] = label_encoder.fit_transform(input_data['medical_specialty'])
    input_data['diag_1'] = label_encoder.fit_transform(input_data['diag_1'])

    prediction = model.predict(input_data)
    # st.write(prediction)
 
    if prediction[0] == 0:
        st.write("There is no chance of re-admission")
    elif prediction[0] == 1:
        st.write("You are likely to get re-admitted within 30 days.")
