import os
import streamlit as st
from dotenv import load_dotenv
from utils.b2 import B2
import pandas as pd
from joblib import load
import numpy as np
from joblib import dump


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
# collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df = b2.get_df(REMOTE_DATA)
    
    return df

# ------------------------------------------------------
#                         APP
# ------------------------------------------------------

# Display image at the top
st.image('diab.jpg', use_column_width=True)  
st.title('Patients Re-admission Prediction')

df_base = get_data()

# Load the trained model and label encoder
model = load('RandomForestClassifier.joblib')

label_encoder = load('label_encoder.joblib')
# Incase of compressing
# dump(model, 'Random_forest_compress.joblib',compress=3)

# Preset values
presets = {
    "Preset 1": {
        'admission_type_id': 1,
        'discharge_disposition_id': 1,
        'time_in_hospital': 5,
        'medical_specialty': 'Cardiology',
        'num_procedures': 1,
        'num_medications': 10,
        'diag_1': 'ICD250',
        'high_number_diagnoses': 0,
        # 'change': 1,
        'total_visits': 3
    },
    "Preset 2": {
        'admission_type_id': 1,
        'discharge_disposition_id': 1,
        'time_in_hospital': 3,
        'medical_specialty': 'Unknown',
        'num_procedures': 1,
        'num_medications': 13,
        'diag_1': 'ICD428',
        'high_number_diagnoses': 0,
        # 'change': 0,
        'total_visits': 7
    }
}

# Create a sidebar for preset selection
st.sidebar.title("Select a Preset")
selected_preset = st.sidebar.selectbox("Choose a preset", ["None"] + list(presets.keys()))

# If a preset is selected, load its values
if selected_preset != "None":
    preset_values = presets[selected_preset]
    st.sidebar.write(f"Loaded {selected_preset} values")

# Create input fields
st.subheader("Enter your details")
admission_type_id = st.selectbox("Select your admission type", sorted(df_base['admission_type_id'].unique()), index=preset_values.get('admission_type_id', 0) if selected_preset != "None" else 0)
discharge_disposition_id = st.selectbox("Select your discharge disposition", sorted(df_base['discharge_disposition_id'].unique()), index=preset_values.get('discharge_disposition_id', 0) if selected_preset != "None" else 0)
time_in_hospital = st.number_input("Enter your time in Hospital", min_value=1, max_value=14, value=preset_values.get('time_in_hospital', 1) if selected_preset != "None" else 1)
medical_specialty = st.selectbox("Select your medical specialty", df_base['medical_specialty'].unique(), index=df_base['medical_specialty'].unique().tolist().index(preset_values.get('medical_specialty', '')) if selected_preset != "None" else 0)
num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=6, value=preset_values.get('num_procedures', 0) if selected_preset != "None" else 0)
num_medications = st.number_input("Number of Medications", min_value=1, max_value=81, value=preset_values.get('num_medications', 1) if selected_preset != "None" else 1)
diag_1 = st.selectbox("Select your primary diagnosis", df_base['diag_1'].unique(), index=df_base['diag_1'].unique().tolist().index(preset_values.get('diag_1', '')) if selected_preset != "None" else 0)
# number_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=16, value=preset_values.get('number_diagnoses', 1) if selected_preset != "None" else 1)
# change = st.selectbox("Change", [0, 1], index=preset_values.get('change', 0) if selected_preset != "None" else 0)
total_visits = st.number_input("Number of visits to hospital", min_value=0, max_value=80, value=preset_values.get('total_visits', 0) if selected_preset != "None" else 0)
high_number_diagnoses = st.selectbox('Number of Medical Diagnosis',[0, 1], index=(preset_values.get('high_number_diagnoses', 0)) if selected_preset != "None" else 0)
st.caption('Select "1" if it\'s more than Eight or else "0"', unsafe_allow_html=False, help=None)


# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    'admission_type_id': [admission_type_id],
    'discharge_disposition_id': [discharge_disposition_id],
    'time_in_hospital': [time_in_hospital],
    'medical_specialty': [medical_specialty],
    'num_procedures': [num_procedures],
    'num_medications': [num_medications],
    'diag_1': [diag_1],
    'total_visits': [total_visits],
    'high_number_diagnoses': [high_number_diagnoses]
    # 'change': [change],
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
        st.write(":green[There is no chance of re-admission]")
    elif prediction[0] == 1:
        st.write(":red[You are likely to get re-admitted within 30 days.]")


