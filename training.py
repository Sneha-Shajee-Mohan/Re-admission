import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import accuracy_score
from joblib import dump
import os
import streamlit as st
from dotenv import load_dotenv
from utils.b2 import B2
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




# Load your data (replace 'your_data.csv' with the path to your actual data)
df = get_data()

# Preprocess categorical features
categorical_features = ['admission_type_id','discharge_disposition_id','medical_specialty','diag_1']
le = LabelEncoder()
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])


# Separate features and target variable
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Convert target variable to numerical (YES=1, NO=0)
y = y.map({'NO': 0, 'YES': 1})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the testing set
# y_pred = clf.predict(X_test)

# Evaluate model performance (you can add metrics like accuracy, precision, recall, etc.)

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# confusion matrix
# Assuming you have predicted labels (y_pred) and true labels (y_true)
# cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
# print(cm)

# Calculate precision
# precision = precision_score(y_test, y_pred,average='weighted')
# print("Precision:", precision)

# Calculate recall
# recall = recall_score(y_test, y_pred,average='weighted')
# print("Recall:", recall)

# Save the model
dump(clf, 'decision_tree_model.joblib')
dump(le, 'label_encoder.joblib')
