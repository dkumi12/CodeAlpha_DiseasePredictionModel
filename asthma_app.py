import streamlit as st  
import joblib  
import numpy as np  
import pandas as pd  
   
 # Load model  
@st.cache_resource  
def load_model():  
     model = joblib.load('asthma_severity_model.joblib')  
     return model  
   
model = load_model()  
   
st.title("Asthma Severity Prediction")  
   
st.markdown("### Input Your Data")  
st.write("Fill in the symptom details and demographic features to get a prediction.")  
   
 # Define input fields based on the numeric and categorical features seen previously.  
 # Numeric features (as per our training):   
 # ['Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'None_Sympton', 'Pains', 'Nasal-Congestion', 'Runny-Nose',   
 # 'None_Experiencing', 'Total_Symptoms', 'Respiratory_Score', 'Respiratory_Symptoms', 'Discomfort_Symptoms', 'Age_Risk_Factor',   
 # 'Symptom_Severity_Index', 'Symptom_Diversity', 'Breathing_Difficulty_Young', 'Breathing_Difficulty_Elderly']  
 #  
 # Categorical features:  
 # ['Primary_Age_Group', 'Gender']  
   
st.markdown("#### Numeric Features")  
def num_input(name, default=0.0):  
     return st.number_input(name, value=default)  
   
tiredness = num_input("Tiredness", 0)  
dry_cough = num_input("Dry-Cough", 0)  
difficulty_in_breathing = num_input("Difficulty-in-Breathing", 0)  
sore_throat = num_input("Sore-Throat", 0)  
none_sympton = num_input("None_Sympton", 0)  
pains = num_input("Pains", 0)  
nasal_congestion = num_input("Nasal-Congestion", 0)  
runny_nose = num_input("Runny-Nose", 0)  
none_experiencing = num_input("None_Experiencing", 0)  
total_symptoms = num_input("Total_Symptoms", 0)  
respiratory_score = num_input("Respiratory_Score", 0)  
respiratory_symptoms = num_input("Respiratory_Symptoms", 0)  
discomfort_symptoms = num_input("Discomfort_Symptoms", 0)  
age_risk_factor = num_input("Age_Risk_Factor", 0)  
symptom_severity_index = num_input("Symptom_Severity_Index", 0)  
symptom_diversity = num_input("Symptom_Diversity", 0)  
breathing_difficulty_young = num_input("Breathing_Difficulty_Young", 0)  
breathing_difficulty_elderly = num_input("Breathing_Difficulty_Elderly", 0)  
   
st.markdown("#### Categorical Features")  
primary_age_group = st.selectbox("Primary Age Group", options=["Child", "Adult", "Elderly"])  
gender = st.selectbox("Gender", options=["Male", "Female"])  
   
 # Assemble user inputs into a DataFrame for prediction.  
 # The columns order must be exactly as expected by the model.  
data = {  
     "Tiredness": [tiredness],  
     "Dry-Cough": [dry_cough],  
     "Difficulty-in-Breathing": [difficulty_in_breathing],  
     "Sore-Throat": [sore_throat],  
     "None_Sympton": [none_sympton],  
     "Pains": [pains],  
     "Nasal-Congestion": [nasal_congestion],  
     "Runny-Nose": [runny_nose],  
     "None_Experiencing": [none_experiencing],  
     "Total_Symptoms": [total_symptoms],  
     "Respiratory_Score": [respiratory_score],  
     "Respiratory_Symptoms": [respiratory_symptoms],  
     "Discomfort_Symptoms": [discomfort_symptoms],  
     "Age_Risk_Factor": [age_risk_factor],  
     "Symptom_Severity_Index": [symptom_severity_index],  
     "Symptom_Diversity": [symptom_diversity],  
     "Breathing_Difficulty_Young": [breathing_difficulty_young],  
     "Breathing_Difficulty_Elderly": [breathing_difficulty_elderly],  
     "Primary_Age_Group": [primary_age_group],  
     "Gender": [gender]  
 } 
   
input_df = pd.DataFrame(data)  
   
if st.button("Predict Severity"):  
    prediction = model.predict(input_df)  
    st.success("Predicted Asthma Severity: " + prediction[0])