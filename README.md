# Re-admission

**Problem Statement**

Developed a machine learning model to predict the likelihood of hospital readmission of diabetic patients based on their medical history, diagnosis data, medication, demographics, and treatment data. The goal is to provide healthcare providers with a tool to identify patients at high risk of readmission, enabling targeted interventions to reduce readmission rates and improve patient outcomes.

**Data Preference**

https://www.kaggle.com/datasets/saurabhtayal/diabetic-patients-readmission-prediction?select=diabetic_data.csv 

**Tools Used**

Python: 
Python is the primary programming language used for developing the application. It is known for its simplicity, readability, and extensive libraries for data manipulation, web development, and API integration.

Scikit-Learn:
The primary machine learning library used for building, training, and evaluating models. It includes tools for data preprocessing, model selection, and performance metrics.

Streamlit: 
Streamlit is a popular Python library used for building interactive web applications with minimal code. It provides easy-to-use widgets and components for creating user interfaces directly from Python scripts.

Jupyter Notebook:
An interactive environment for running the code, exploring data, and visualizing results. It allows for iterative development and easy debugging.

Joblib:
Used for saving and loading machine learning models efficiently. It helps in persisting models to disk for future use.

Pandas: 
Pandas is a powerful data manipulation library in Python used for data analysis and manipulation. It is used to handle and pre-process the patient data stored in the DataFrame.

Git: 
Git is a version control system used for tracking changes in the codebase, collaborating with other developers, and managing project history. It helps in maintaining code quality, facilitating collaboration, and ensuring project integrity.

Backblaze: 
The Backblaze application, often referred to as Backblaze Personal Backup or Backblaze Business Backup, is a software application developed by Backblaze that enables users to securely backup their files and data to the cloud.

**Algorithm Description**

1. Data Collection and Preprocessing
   The initial dataset comprises over 100,000 records with 50 features. It underwent meticulous preprocessing, including datatype conversion, removal of NaN values, replacement of unknown values, mapping of integers to corresponding strings, and ultimately the selection of a baseline dataset for further training.

2. Key Variables

    • Target Variable: Readmission status

    • Features: admission_type_id, discharge_disposition_id, time_in_hospital, medical_specialty,  num_procedures, num_medications, total visits, diagnosis, high number of diagnoses.

3. Feature Engineering
   Feature engineering is a critical step in preparing the dataset for model training. It involves creating new features, transforming existing ones, and removing redundant or irrelevant features to improve the model's performance. 

   A new feature "total_visits" was created by summing up the number of outpatient, emergency, and inpatient visits. This feature provides a comprehensive view of a patient's total interactions with healthcare services.

   A binary feature "high_number_diagnoses" was created to identify patients with a high number of diagnoses. Patients with more than 8 diagnoses were flagged as 1, otherwise 0. This helps in distinguishing patients with complex medical histories.

   After creating the new features, the original features that contributed to the new features were dropped to avoid redundancy. This helps in reducing the dimensionality of the dataset and simplifying the model.


4. Handling Class Imbalance
   Calculate the class weights based on the inverse of the class frequencies in the training set to address class imbalance.
   Synthetic sampling methods like ADASYN (Adaptive Synthetic Sampling) and SMOTE (Synthetic Minority Over-sampling Technique) are often used to address class imbalance by creating synthetic examples of the minority class. While these methods can be very useful, they also have some potential drawbacks and limitations that may affect the performance and reliability of the resulting models.

5. Model Training and Evaluation
   A RandomForestClassifier with the computed class weights and the best parameters obtained from tuning was chosen as the final model
   The model was chosen based on the model's performance on precision, recall, and F1-score.

6. Addressing Class Imbalance and Emphasizing Recall
   
   Class imbalance is a common challenge in many real-world datasets, where the number of instances in one class significantly outnumbers those in another. This imbalance can lead to models that are biased towards the majority class, often resulting in high accuracy but poor performance in identifying the minority class. my project faces such an imbalance, with a critical need to correctly identify instances of the minority class. Hence, i prioritized improving recall over accuracy. Recall measures the model's ability to correctly identify all relevant instances of the minority class, which is crucial in scenarios like medical diagnoses or fraud detection, where missing a positive case can have severe consequences. By focusing on recall, we ensure that our model is more sensitive to detecting the minority class, thereby reducing false negatives and improving overall performance in critical applications. This approach helps create a more balanced and effective model, even in the presence of significant class imbalance.

7. Deployment
   Developed a user-friendly interface using Streamlit.


**Ethical Concerns**

 A significant ethical issue is bias and fairness in this model. Class imbalance can lead to biased predictions that unfairly disadvantage minority groups. Therefore, it is essential to continuously monitor and mitigate bias throughout the model development process, ensuring that the model's predictions do not perpetuate or exacerbate existing inequalities. Additionally, transparency in model decisions is vital; stakeholders should understand how the model arrives at its conclusions, particularly in high-stakes scenarios like healthcare in this case. Clear communication about the model's limitations and the uncertainty in its predictions is necessary to prevent over-reliance on the model and ensure that human judgment remains an integral part of the decision-making process. 
 
