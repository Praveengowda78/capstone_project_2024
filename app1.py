import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Mock data from model.py
symptoms= {
        "Influenza": ["Fever", "Cough", "Sore throat", "Muscle aches", "Fatigue"],
        "Dengue": ["High fever", "Severe headache", "Rash", "Muscle pain"],
        "Malaria": ["Chills", "Fever", "Fatigue", "Sweats"],
        "Typhoid Fever": ["Abdominal pain", "Loss of appetite", "High fever"],
        "Pneumonia": ["Chest pain", "Shortness of breath", "Fever", "Cough"],
        "Appendicitis": ["Sudden abdominal pain (lower right)", "Nausea", "Vomiting", "Low-grade fever"],
        "Gastroenteritis": ["Diarrhea", "Abdominal cramps", "Vomiting", "Low-grade fever"],
        "Migraine": ["Severe headache", "Sensitivity to light/sound", "Nausea"],
        "Urinary Tract Infection (UTI)": ["Painful urination", "Frequent urination", "Lower abdominal pain", "Cloudy urine"],
        "Bronchitis": ["Cough with mucus", "Wheezing", "Chest tightness"],
        "COVID-19": ["Fever", "Dry cough", "Loss of taste/smell", "Fatigue", "Difficulty breathing"],
        "Sinusitis": ["Facial pain", "Nasal congestion", "Headache", "Post-nasal drip"],
        "Chikungunya": ["High fever", "Joint pain", "Rash", "Headache"],
        "Chickenpox": ["Fever", "Rash (blisters)", "Itching", "Fatigue"],
        "Zika Virus": ["Fever", "Rash", "Conjunctivitis", "Joint pain"],
        "Leptospirosis": ["Fever", "Headache", "Muscle pain", "Abdominal pain"],
        "Hepatitis A": ["Fever", "Nausea", "Vomiting", "Jaundice", "Abdominal pain"],
        "Hepatitis B": ["Fatigue", "Jaundice", "Abdominal pain", "Nausea"],
        "Cholera": ["Severe diarrhea", "Vomiting", "Dehydration"],
        "Tetanus": ["Muscle stiffness", "Lockjaw", "Difficulty swallowing"],
        "Rabies": ["Fever", "Headache", "Hydrophobia", "Agitation"],
        "Mumps": ["Fever", "Swelling of salivary glands", "Headache"],
        "Measles": ["Fever", "Cough", "Rash", "Conjunctivitis"],
        "Diphtheria": ["Sore throat", "Fever", "Weakness", "Swollen glands"],
        "Scarlet Fever": ["Sore throat", "Red rash", "Fever", "Strawberry tongue"],
        "Typhus": ["Fever", "Rash", "Headache", "Muscle pain"],
        "Rotavirus": ["Severe diarrhea", "Vomiting", "Fever", "Abdominal cramps"],
        "Norovirus": ["Diarrhea", "Vomiting", "Stomach cramps"],
        "Enteric Fever": ["Abdominal pain", "High fever", "Diarrhea", "Weakness"],
        "H1N1 (Swine Flu)": ["Fever", "Cough", "Sore throat", "Body aches"]
    }

diseases= [
        "Influenza",
        "Dengue",
        "Malaria", 
        "Typhoid Fever", 
        "Pneumonia",
        "Appendicitis",
        "Gastroenteritis",
        "Migraine", 
        "Urinary Tract Infection (UTI)",
        "Bronchitis",
        "COVID-19", 
        "Sinusitis", 
        "Chikungunya",
        "Chickenpox",
        "Zika Virus",
        "Leptospirosis",
        "Hepatitis A",
        "Hepatitis B", 
        "Cholera",
        "Tetanus", 
        "Rabies",
        "Mumps", 
        "Measles",
        "Diphtheria",
        "Scarlet Fever",
        "Typhus",
        "Rotavirus", 
        "Norovirus",
        "Enteric Fever",
        "H1N1 (Swine Flu)"
]

Medicines= {
        "Influenza": ["Paracetamol 500mg", "Ibuprofen 400mg", "Oseltamivir 75mg"],
        "Dengue": ["Acetaminophen 500mg", "ORS (Oral Rehydration Solution)"],
        "Malaria": ["Artemisinin-based Combination Therapy (ACT)"],
        "Typhoid Fever": ["Ciprofloxacin 500mg", "Azithromycin 250mg"],
        "Pneumonia": ["Amoxicillin 500mg", "Doxycycline 100mg"],
        "Appendicitis": ["Painkillers (e.g., Ibuprofen 200mg)", "Antibiotics (e.g., Ceftriaxone 1g)"],
        "Gastroenteritis": ["ORS (Oral Rehydration Solution)", "Loperamide 2mg", "Zinc 20mg"],
        "Migraine": ["Sumatriptan 50mg", "Ibuprofen 400mg"],
        "Urinary Tract Infection (UTI)": ["Nitrofurantoin 100mg", "Ciprofloxacin 500mg"],
        "Bronchitis": ["Bronchodilators (e.g., Salbutamol 2.5mg)", "Cough Suppressants (e.g., Dextromethorphan 10mg)"],
        "COVID-19": ["Paracetamol 500mg", "Favipiravir 200mg"],
        "Sinusitis": ["Decongestants (e.g., Pseudoephedrine 60mg)", "Antihistamines (e.g., Loratadine 10mg)"],
        "Chikungunya": ["Paracetamol 500mg", "Ibuprofen 200mg"],
        "Chickenpox": ["Calamine Lotion", "Antihistamines (e.g., Diphenhydramine 25mg)"],
        "Zika Virus": ["Paracetamol 500mg", "Fluids"],
        "Leptospirosis": ["Doxycycline 100mg", "Penicillin 500mg"],
        "Hepatitis A": ["Supportive care", "Hydration"],
        "Hepatitis B": ["Tenofovir 300mg", "Entecavir 0.5mg"],
        "Cholera": ["Rehydration (e.g., ORS)", "Antibiotics (e.g., Doxycycline 300mg)"],
        "Tetanus": ["Tetanus vaccination", "Painkillers (e.g., Ibuprofen 400mg)"],
        "Rabies": ["Post-exposure prophylaxis (PEP) with Rabies Vaccine"],
        "Mumps": ["Supportive care", "Pain relief (e.g., Paracetamol 500mg)"],
        "Measles": ["Vitamin A 100,000 IU (for children)", "Supportive care"],
        "Diphtheria": ["Penicillin 250mg", "Diphtheria Toxoid Vaccine"],
        "Scarlet Fever": ["Penicillin 500mg", "Amoxicillin 500mg"],
        "Typhus": ["Doxycycline 100mg", "Chloramphenicol 500mg"],
        "Rotavirus": ["ORS (Oral Rehydration Solution)", "Zinc 20mg"],
        "Norovirus": ["Fluids", "Antiemetics (e.g., Ondansetron 4mg)"],
        "Enteric Fever": ["Ciprofloxacin 500mg", "Azithromycin 250mg"],
        "H1N1 (Swine Flu)": ["Oseltamivir 75mg", "Paracetamol 500mg"]
    }

description={
        "Influenza": ["The disease Influenza causes various symptoms and can be diagnosed through specific tests."],
        
        "Dengue": ["Dengue causes a high fever with severe headache and rash."],
        
        "Malaria": ["Malaria involves chills, fever, and sweats."],
        
        "Typhoid Fever":  ["Typhoid is a bacterial infection that requires antibiotics."],
        
        "Pneumonia":["Pneumonia causes breathing difficulties and chest pain."],
        
        "Appendicitis": ["Appendicitis needs immediate surgery to avoid complications."],
        
        "Gastroenteritis": ["Gastroenteritis involves inflammation of the stomach and intestines."],
        
        "Migraine":  ["Migraines are severe headaches often accompanied by nausea."],
        
        "Urinary Tract Infection (UTI)":  ["UTI is characterized by painful urination and frequent trips to the bathroom."],
        
        "Bronchitis": ["Bronchitis causes persistent coughing and mucus production."],
        
        "COVID-19":  ["COVID-19 is a respiratory disease with diverse symptoms."],
        
        "Sinusitis": ["Sinusitis affects nasal passages and causes severe headaches."],
        
        "Chikungunya":  ["Chikungunya causes joint pain and fever."],
        
        "Chickenpox":["Chickenpox is a contagious rash illness."],
        
        "Zika Virus":  ["Zika Virus can cause birth defects in pregnant women."],
        
        "Leptospirosis": ["Leptospirosis is a bacterial infection that spreads through animal urine."],
        
        "Hepatitis A":  ["Hepatitis A is a liver infection caused by contaminated food or water."],
        
        "Hepatitis B":  ["Hepatitis B is a liver infection spread through body fluids."],
        
        "Cholera":  ["Cholera is caused by contaminated water and causes severe dehydration."],
        
        "Tetanus": ["Tetanus affects the nervous system and causes muscle stiffness."],
        
        "Rabies":  ["Rabies is a viral infection often fatal once symptoms appear."],
        
        "Mumps":  ["Mumps is a viral infection causing swollen salivary glands."],
        
        "Measles":  ["Measles causes a characteristic rash and fever."],
        
        "Diphtheria": ["Diphtheria affects the throat and nose."],
        
        "Scarlet Fever": ["Scarlet Fever involves a bright red rash."],
        
        "Typhus":  ["Typhus is transmitted by lice and fleas."],
        
        "Rotavirus":  ["Rotavirus causes severe diarrhea in children."],
        
        "Norovirus":  ["Norovirus is a contagious stomach flu."],
        
        "Enteric Fever":  ["Enteric Fever is a severe bacterial infection."],
        
        "H1N1 (Swine Flu)": ["H1N1 (Swine Flu) is a subtype of influenza."],
      }  
    


    
diagnostic_Test= {
        "Influenza": ["Rapid Antigen Test", "RT-PCR"],
        "Dengue": ["NS1 Antigen", "Dengue IgM & IgG Antibody"],
        "Malaria": ["Blood Smear", "Rapid Diagnostic Test"],
        "Typhoid Fever": ["Blood Culture", "Widal Test"],
        "Pneumonia": ["Chest X-ray", "Blood Tests"],
        "Appendicitis": ["Abdominal Ultrasound", "CT Scan"],
        "Gastroenteritis": ["Stool Culture", "Blood Tests"],
        "Migraine": ["Clinical Diagnosis", "CT/MRI (if necessary)"],
        "Urinary Tract Infection (UTI)": ["Urine Culture", "Urine Dipstick Test"],
        "Bronchitis": ["Chest X-ray", "Sputum Culture"],
        "COVID-19": ["RT-PCR", "Rapid Antigen Test"],
        "Sinusitis": ["CT Scan", "Nasal Endoscopy"],
        "Chikungunya": ["Serology Test", "RT-PCR"],
        "Chickenpox": ["Clinical Diagnosis", "Blood Test (for IgM)"],
        "Zika Virus": ["RT-PCR", "Serology Test"],
        "Leptospirosis": ["Blood Tests", "Serology Test"],
        "Hepatitis A": ["Liver Function Test", "Hepatitis A IgM Antibody"],
        "Hepatitis B": ["HBV Surface Antigen", "HBV DNA PCR"],
        "Cholera": ["Stool Culture", "Rapid Diagnostic Test"],
        "Tetanus": ["Clinical Diagnosis", "Tetanus Toxin Test"],
        "Rabies": ["Saliva Test", "Serum Antibodies"],
        "Mumps": ["Serum IgM Antibody Test", "RT-PCR"],
        "Measles": ["Serum IgM Test", "RT-PCR"],
        "Diphtheria": ["Throat Culture", "Cultural Test for C. diphtheriae"],
        "Scarlet Fever": ["Throat Culture", "Rapid Antigen Test"],
        "Typhus": ["Blood Test (Serology)", "PCR"],
        "Rotavirus": ["Stool Antigen Test", "RT-PCR"],
        "Norovirus": ["Stool Sample Test", "PCR"],
        "Enteric Fever": ["Blood Culture", "Widal Test"],
        "H1N1 (Swine Flu)": ["RT-PCR", "Rapid Antigen Test"]
    }
Precautions= {
        "Influenza": ["Avoid close contact", "Cover mouth when coughing", "Get vaccinated annually"],
        "Dengue": ["Use mosquito repellent", "Wear long sleeves and pants", "Sleep under mosquito nets"],
        "Malaria": ["Use insect repellents", "Sleep under a mosquito net", "Take anti-malarial drugs as prescribed"],
        "Typhoid Fever": ["Drink clean water", "Wash hands frequently", "Avoid eating raw food"],
        "Pneumonia": ["Cover mouth when coughing", "Get vaccinated", "Avoid smoking"],
        "Appendicitis": ["Seek medical help immediately", "Avoid self-medication"],
        "Gastroenteritis": ["Wash hands regularly", "Avoid contaminated food or water"],
        "Migraine": ["Avoid triggers like bright lights and loud sounds", "Manage stress"],
        "Urinary Tract Infection (UTI)": ["Drink plenty of water", "Avoid holding urine for too long", "Wipe from front to back"],
        "Bronchitis": ["Avoid smoke and pollution", "Rest and hydrate"],
        "COVID-19": ["Wear masks", "Social distancing", "Frequent handwashing"],
        "Sinusitis": ["Avoid allergens", "Use a humidifier", "Drink plenty of fluids"],
        "Chikungunya": ["Avoid mosquito bites", "Stay indoors during peak mosquito activity"],
        "Chickenpox": ["Avoid scratching blisters", "Stay home until all blisters scab over"],
        "Zika Virus": ["Avoid mosquito bites", "Wear long sleeves", "Use insect repellent"],
        "Leptospirosis": ["Avoid exposure to contaminated water", "Wear protective gear"],
        "Hepatitis A": ["Get vaccinated", "Avoid unclean food and water"],
        "Hepatitis B": ["Get vaccinated", "Avoid sharing needles"],
        "Cholera": ["Drink clean water", "Use ORS to stay hydrated"],
        "Tetanus": ["Get vaccinated", "Avoid open wounds and cuts"],
        "Rabies": ["Avoid contact with stray animals", "Get post-exposure vaccination if bitten"],
        "Mumps": ["Isolate from others", "Rest and hydrate"],
        "Measles": ["Get vaccinated", "Avoid contact with others when sick"],
        "Diphtheria": ["Get vaccinated", "Avoid contact with infected individuals"],
        "Scarlet Fever": ["Take prescribed antibiotics", "Avoid close contact with others"],
        "Typhus": ["Wear protective clothing", "Avoid flea bites"],
        "Rotavirus": ["Practice good hygiene", "Ensure children are vaccinated"],
        "Norovirus": ["Wash hands regularly", "Avoid contaminated food"],
        "Enteric Fever": ["Drink clean water", "Wash hands before eating"],
        "H1N1 (Swine Flu)": ["Get vaccinated", "Avoid close contact with infected individuals"]
    }

# Load your trained model
# Replace 'path_to_model.h5' with the actual file path of your model
model = load_model("multi_output_disease_prediction_model.h5")

# Helper function for prediction
def predict_disease(user_symptoms):
    """
    This is a mock prediction function. Replace this with the actual model inference logic.
    """
    # Replace the below logic with model inference once trained
    match = {disease: len(set(user_symptoms).intersection(symptoms[disease])) for disease in diseases}
    predicted_disease = max(match, key=match.get)
    return predicted_disease

# Streamlit app UI
st.title("Disease Prediction App")
st.write("Provide your symptoms, and we will predict the most probable disease.")

# Symptom input
user_symptoms = st.multiselect(
    "Select your symptoms (You can choose multiple):",
    options=[symptom for symptom_list in symptoms.values() for symptom in symptom_list],
)

if st.button("Predict"):
    if user_symptoms:
        # Call the prediction function
        predicted_disease = predict_disease(user_symptoms)
        st.subheader(f"Predicted Disease: {predicted_disease}")

        # Show additional information
        st.write("### Disease Description")
        st.write(", ".join(description.get(predicted_disease, ["No specific medicines available."])))


        st.write("### Recommended Medicines")
        st.write(", ".join(Medicines.get(predicted_disease, ["No specific medicines available."])))

        st.write("### Diagnostic Tests")
        st.write(", ".join(diagnostic_Test.get(predicted_disease, ["No diagnostic tests available."])))
    else:
        st.warning("Please select at least one symptom to predict a disease.")

# Footer
st.write("---")
st.write("This app is for informational purposes only. For a proper diagnosis, consult a healthcare professional.")
