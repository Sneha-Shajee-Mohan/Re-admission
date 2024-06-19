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
#     # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df = b2.get_df(REMOTE_DATA)
    
    return df


# Load your data 
df = get_data()


# ------------------------------------------------------
#                        FEATURE ENGINEERING
# ------------------------------------------------------
df['total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
df = df.drop(['number_inpatient','number_outpatient','number_emergency'], axis=1)

# Preprocess categorical features
categorical_features = ['admission_type_id','discharge_disposition_id','medical_specialty','diag_1']
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


# Apply BorderlineSMOTE to the training set
smote = BorderlineSMOTE(sampling_strategy='minority', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Shuffle the resampled training set
X_train_resampled, y_train_resampled = X_train_resampled.sample(frac=1, random_state=42).reset_index(drop=True), y_train_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Initialize the RandomForestClassifier with the best parameters

clf = RandomForestClassifier(
    bootstrap=False,
    class_weight='balanced',
    max_depth=10,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100
)

# clf = GradientBoostingClassifier(
#     # bootstrap=False,
#     # class_weight='balanced',
#     random_state=42,
#     max_depth=10,
#     min_samples_leaf=1,
#     min_samples_split=2,
#     n_estimators=200
# )


clf.fit(X_train_resampled, y_train_resampled)
# print(X_train_resampled.columns)

# Cross-validation
# cv_scores = cross_val_score(clf, X_train_resampled, y_train_resampled, cv=5)
# print("Cross-validation scores:", cv_scores)
# print("Mean cross-validation score:", np.mean(cv_scores))


# Save the model
dump(clf, 'RandomForestClassifier.joblib')
dump(le, 'label_encoder.joblib')
