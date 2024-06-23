import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from joblib import dump
import os
import streamlit as st
from dotenv import load_dotenv
from utils.b2 import B2
from imblearn.over_sampling import ADASYN, BorderlineSMOTE


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


# Load your data 
df = get_data()

# ------------------------------------------------------
#                        FEATURE ENGINEERING
# ------------------------------------------------------
# df['high_num_procedures'] = df['num_procedures'].apply(lambda x: 1 if x > 3 else 0)  # Example threshold
df['total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
df['high_number_diagnoses'] = df['number_diagnoses'].apply(lambda x: 1 if x > 8 else 0)
df = df.drop(['number_diagnoses','number_inpatient','number_outpatient','number_emergency','change'], axis=1)


# Preprocess categorical features
categorical_features = ['admission_type_id','medical_specialty','discharge_disposition_id','diag_1']
le = LabelEncoder()
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])


# -----------------------------------------------------------------
#      Custom train_test_split based on patient_nbr(Unique rows)
# -----------------------------------------------------------------


def custom_train_test_split(df, patient_col='patient_nbr', test_size=0.3):
    patients = df[patient_col].unique()
    train_patients, test_patients = train_test_split(patients, test_size=test_size, random_state=42)
    train_data = df[df[patient_col].isin(train_patients)]
    test_data = df[df[patient_col].isin(test_patients)]
    return train_data, test_data

train_data, test_data = custom_train_test_split(df)

# Split features and target
X_train = train_data.drop(columns=['readmitted', 'patient_nbr'])
y_train = train_data['readmitted']
X_test = test_data.drop(columns=['readmitted', 'patient_nbr'])
y_test = test_data['readmitted']

# Handle class imbalance using class weights
class_weights = dict(pd.Series(y_train).value_counts(normalize=True).apply(lambda x: 1/x).to_dict())

# Initialize the RandomForestClassifier with the best parameters(selected after going through gridsearchCV with param grids)

clf = RandomForestClassifier(
   random_state=42,
   max_depth=10,
   min_samples_leaf=4,
   min_samples_split=10,
   n_estimators=200,
   class_weight=class_weights
)
clf.fit(X_train, y_train)
# print(X_train.columns)

# CV-SCORE 
# cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
# print("Cross-validation ROC-AUC scores:", cv_scores)
# print("Mean cross-validation ROC-AUC score:", np.mean(cv_scores))

# Save the model
# dump(clf, 'RandomForestClassifier.joblib')
# dump(le, 'label_encoder.joblib')
