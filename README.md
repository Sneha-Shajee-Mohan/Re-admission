# Re-admission

**Problem Statement**

Developed a machine learning model to predict the likelihood of hospital readmission of diabetic patients based on their medical history, diagnosis data, medication, demographics, and treatment data. The goal is to provide healthcare providers with a tool to identify patients at high risk of readmission, enabling targeted interventions to reduce readmission rates and improve pa6ent outcomes.

**Data Preference**

https://www.kaggle.com/datasets/saurabhtayal/diabetic-patients-readmission-prediction?select=diabetic_data.csv 

**Tools Used**
Python: Python is the primary programming language used for developing the application. It is known for its simplicity, readability, and extensive libraries for data manipulation, web development, and API integration.

Streamlit: Streamlit is a popular Python library used for building interactive web applications with minimal code. It provides easy-to-use widgets and components for creating user interfaces directly from Python scripts.

Pandas: Pandas is a powerful data manipulation library in Python used for data analysis and manipulation. It is used to handle and pre-process the patient data stored in the DataFrame.

Git: Git is a version control system used for tracking changes in the codebase, collaborating with other developers, and managing project history. It helps in maintaining code quality, facilitating collaboration, and ensuring project integrity.

Backblaze: The Backblaze application, often referred to as Backblaze Personal Backup or Backblaze Business Backup, is a software application developed by Backblaze that enables users to securely backup their files and data to the cloud.

**Algorithm Description**

1. Data Collection and Preprocessing

2. Key Variables

    • Target Variable: Readmission status

    • Features: admission_type_id, discharge_disposition_id, time_in_hospital, medical_specialty,   num_procedures, num_medications, number_inpatient, number_outpatient, number_emergency, diag_1, number_diagnoses, change

3. Feature Selection and Engineering
   Based on p-values, pair-plots, Accuracy, precision

4. Model Training and Evaluation
   Trained the model using machine learning models such as logistic regression, decision trees, and random forests.
   Evaluated the models using appropriate metrics such as accuracy, precision, recall.

5. Deployment
   Developed a user-friendly interface using Streamlit.

**Next Steps**

1)Modify User interface

2)Feature selection

3)Improve Accuracy

**Ethical Concerns**

 
 
