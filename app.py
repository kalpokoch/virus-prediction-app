
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import xgboost as xgb

# Page configuration
st.set_page_config(
    page_title="Virus Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Virus mapping (26 classes after filtering)
VIRUS_MAPPING = {
    0: 'Chikungunya Virus',
    1: 'Dengue Virus',
    2: 'Enterovirus',
    3: 'Hepatitis A Virus',
    4: 'Hepatitis B Virus',
    5: 'Hepatitis C Virus',
    6: 'Hepatitis E Virus',
    7: 'Herpes simplex virus',
    8: 'Influenza A H1N1',
    9: 'Influenza A H3N2',
    10: 'Influenza B Victoria',
    11: 'Japanese Encephalitis',
    12: 'Leptospira',
    13: 'Measles Virus',
    14: 'Mumps Virus',
    15: 'OtherViruses',
    16: 'Parvovirus',
    17: 'Respiratory Adenovirus',
    18: 'Respiratory Syncytial Virus RSV',
    19: 'Respiratory Syncytial Virus-A RSV-A',
    20: 'Respiratory Syncytial Virus-B RSV-B',
    21: 'Rotavirus',
    22: 'Rubella',
    23: 'SARS-Cov-2',
    24: 'Scrub typhus Orientia tsutsugamushi',
    25: 'Varicella zoster virus VZV'
}

# Symptom groups
SYMPTOM_GROUPS = {
    "Neurological": ['HEADACHE', 'IRRITABLITY', 'ALTEREDSENSORIUM', 'SOMNOLENCE', 
                     'NECKRIGIDITY', 'SEIZURES'],
    "Gastrointestinal": ['DIARRHEA', 'DYSENTERY', 'NAUSEA', 'VOMITING', 'ABDOMINALPAIN'],
    "Systemic": ['MALAISE', 'MYALGIA', 'ARTHRALGIA', 'CHILLS', 'RIGORS', 'FEVER'],
    "Respiratory": ['BREATHLESSNESS', 'COUGH', 'RHINORRHEA', 'SORETHROAT'],
    "Dermatological": ['BULLAE', 'PAPULARRASH', 'PUSTULARRASH', 'MUSCULARRASH', 
                       'MACULOPAPULARRASH', 'ESCHAR'],
    "Hepatic/Other": ['DARKURINE', 'HEPATOMEGALY', 'JAUNDICE'],
    "Ocular": ['REDEYE', 'DISCHARGEEYES', 'CRUSHINGEYES']
}

@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    try:
        with open('models/xgb_filtered_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_mappings():
    """Load state, district, and district-state mapping CSV files"""
    try:
        state_map = pd.read_csv('state_encoding_map.csv')
        district_map = pd.read_csv('district_encoding_map.csv')
        district_state_map = pd.read_csv('district_state_mapping.csv')
        return state_map, district_map, district_state_map
    except Exception as e:
        st.error(f"Error loading mapping files: {e}")
        return None, None, None

def create_feature_vector(patient_data):
    """
    Convert user inputs → 80 model features (EXACT training replica)
    """
    # Step 1: Create base DataFrame with correct column names
    feature_df = pd.DataFrame([patient_data])

    # Fill missing values
    feature_df['age'] = feature_df['age'].fillna(30).clip(0, 120)
    feature_df['duration_of_illness'] = feature_df['duration_of_illness'].fillna(0)

    # Fill all symptoms with 0
    symptom_cols = list(sum(SYMPTOM_GROUPS.values(), []))
    for col in symptom_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0
        feature_df[col] = feature_df[col].fillna(0).clip(0, 1)

    # === AGE FEATURES ===
    feature_df['age_group'] = pd.cut(feature_df['age'], 
                                   bins=[0, 5, 18, 45, 65, 150], 
                                   labels=[0, 1, 2, 3, 4]).cat.codes
    feature_df['age_group'] = feature_df['age_group'].replace(-1, 2)

    # === SYMPTOM GROUPS ===
    respiratory_cols = ['COUGH', 'BREATHLESSNESS', 'RHINORRHEA', 'SORETHROAT']
    gi_cols = ['DIARRHEA', 'DYSENTERY', 'NAUSEA', 'VOMITING', 'ABDOMINALPAIN']
    neuro_cols = ['HEADACHE', 'ALTEREDSENSORIUM', 'SEIZURES', 'SOMNOLENCE', 'NECKRIGIDITY', 'IRRITABLITY']
    skin_cols = ['PAPULARRASH', 'PUSTULARRASH', 'MACULOPAPULARRASH', 'BULLAE']
    systemic_cols = ['MYALGIA', 'ARTHRALGIA', 'CHILLS', 'RIGORS', 'MALAISE']

    # Symptom counts
    symptom_count_cols = ['HEADACHE', 'FEVER', 'COUGH', 'VOMITING', 'DIARRHEA', 'MYALGIA', 
                         'ARTHRALGIA', 'NAUSEA', 'BREATHLESSNESS', 'SORETHROAT']

    feature_df['symptom_count'] = feature_df[symptom_count_cols].sum(axis=1)
    feature_df['respiratory_symptoms'] = feature_df[respiratory_cols].sum(axis=1)
    feature_df['gi_symptoms'] = feature_df[gi_cols].sum(axis=1)
    feature_df['neuro_symptoms'] = feature_df[neuro_cols].sum(axis=1)
    feature_df['skin_symptoms'] = feature_df[skin_cols].sum(axis=1)
    feature_df['systemic_symptoms'] = feature_df[systemic_cols].sum(axis=1)
    feature_df['symptom_diversity'] = (feature_df[symptom_count_cols] > 0).sum(axis=1)

    # === GEO-TEMPORAL FEATURES ===
    month = patient_data.get('month', 1)
    feature_df['month'] = month
    feature_df['is_monsoon'] = int(month in [6, 7, 8, 9])
    feature_df['is_winter'] = int(month in [12, 1, 2])

    def get_season(m):
        if m in [12, 1, 2]: return 0
        elif m in [3, 4, 5]: return 1
        elif m in [6, 7, 8, 9]: return 2
        else: return 3

    feature_df['season'] = get_season(month)
    feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['month'] / 12)
    feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['month'] / 12)

    # === INTERACTION FEATURES ===
    # Geo-temporal interactions
    feature_df['monsoon_respiratory'] = feature_df['is_monsoon'] * feature_df['respiratory_symptoms']
    feature_df['winter_respiratory'] = feature_df['is_winter'] * feature_df['respiratory_symptoms']
    feature_df['monsoon_fever'] = feature_df['is_monsoon'] * feature_df['FEVER']

    feature_df['state_season'] = patient_data['lab_state'] * 10 + feature_df['season']
    feature_df['district_season'] = patient_data['district_encoded'] * 10 + feature_df['season']
    feature_df['district_month'] = patient_data['district_encoded'] * 100 + feature_df['month']

    feature_df['state_respiratory'] = patient_data['lab_state'] * feature_df['respiratory_symptoms']
    feature_df['state_fever'] = patient_data['lab_state'] * feature_df['FEVER']
    feature_df['state_gi'] = patient_data['lab_state'] * feature_df['gi_symptoms']

    # Fever interactions
    feature_df['fever_respiratory'] = feature_df['FEVER'] * feature_df['respiratory_symptoms']
    feature_df['fever_gi'] = feature_df['FEVER'] * feature_df['gi_symptoms']
    feature_df['fever_neuro'] = feature_df['FEVER'] * feature_df['neuro_symptoms']
    feature_df['fever_skin'] = feature_df['FEVER'] * feature_df['skin_symptoms']
    feature_df['fever_duration'] = feature_df['FEVER'] * feature_df['duration_of_illness']
    feature_df['fever_headache'] = feature_df['FEVER'] * feature_df['HEADACHE']
    feature_df['fever_cough'] = feature_df['FEVER'] * feature_df['COUGH']

    # Severity & demographic interactions
    feature_df['severity_score'] = feature_df['symptom_count'] * feature_df['duration_of_illness']
    feature_df['age_symptom'] = feature_df['age'] * feature_df['symptom_count']
    feature_df['age_duration'] = feature_df['age'] * feature_df['duration_of_illness']
    feature_df['patienttype_age'] = patient_data['PATIENTTYPE'] * feature_df['age_group']
    feature_df['sex_respiratory'] = patient_data['SEX'] * feature_df['respiratory_symptoms']
    feature_df['duration_symptom_ratio'] = feature_df['duration_of_illness'] / (feature_df['symptom_count'] + 1)

    # Year features (use current year)
    year = patient_data.get('year', 2024)
    feature_df['year'] = year
    feature_df['year_normalized'] = (year - 2012) / (2024 - 2012 + 1)  # Normalize based on training range

    # Quarter, week, day of year
    date = datetime(year, month, 1)
    feature_df['quarter'] = (month - 1) // 3 + 1
    feature_df['week_of_year'] = date.isocalendar()[1]
    feature_df['day_of_year'] = date.timetuple().tm_yday

    # Final cleanup
    feature_df = feature_df.replace([np.inf, -np.inf], 0).fillna(0)

    return feature_df.iloc[0].values.reshape(1, -1)

def main():
    st.title("Virus Detection and Classification System")
    st.markdown("---")
    st.write("Enter patient information and clinical symptoms to predict the most likely virus.")

    # Load model and mappings
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check the model file path.")
        return
    
    state_map, district_map, district_state_map = load_mappings()
    if state_map is None or district_map is None or district_state_map is None:
        st.error("Failed to load mapping files. Please check the CSV files.")
        return

    # Sidebar for patient demographics
    st.sidebar.header("Patient Information")

    patient_data = {}

    # Demographics (MATCH EXACT TRAINING COLUMN NAMES)
    patient_data['age'] = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
    patient_data['SEX'] = st.sidebar.selectbox("Sex", options=[0, 1], 
                                                format_func=lambda x: "Female" if x == 0 else "Male")
    patient_data['PATIENTTYPE'] = st.sidebar.selectbox("Patient Type", options=[0, 1], 
                                                        format_func=lambda x: "Outpatient" if x == 0 else "Inpatient")
    patient_data['duration_of_illness'] = st.sidebar.number_input("Duration of Illness (days)", 
                                                                 min_value=0, max_value=365, value=3)
    
    # State selection with names
    state_names = state_map['state_name'].tolist()
    selected_state_name = st.sidebar.selectbox("State", options=state_names, index=0)
    patient_data['lab_state'] = int(state_map[state_map['state_name'] == selected_state_name]['encoded_value'].values[0])
    
    # District selection filtered by state
    filtered_districts = district_state_map[district_state_map['state'] == selected_state_name]
    district_names = filtered_districts['district_name'].tolist()
    
    if len(district_names) > 0:
        selected_district_name = st.sidebar.selectbox("District", options=district_names, index=0)
        patient_data['district_encoded'] = int(filtered_districts[filtered_districts['district_name'] == selected_district_name]['district_encoded'].values[0])
    else:
        st.sidebar.warning("No districts available for selected state")
        patient_data['district_encoded'] = 0

    # Temporal features
    patient_data['month'] = st.sidebar.selectbox("Month of Illness", options=list(range(1, 13)), 
                                                  format_func=lambda x: datetime(2000, x, 1).strftime('%B'))
    patient_data['year'] = st.sidebar.number_input("Year", min_value=2012, max_value=2026, value=2024)

    # Main area for symptoms
    st.header("Clinical Symptoms")
    st.write("Select all symptoms present in the patient:")

    for group_name, symptoms in SYMPTOM_GROUPS.items():
        with st.expander(f"**{group_name} Symptoms**", expanded=True):
            cols = st.columns(3)
            for idx, symptom in enumerate(symptoms):
                with cols[idx % 3]:
                    patient_data[symptom] = 1 if st.checkbox(symptom.replace('_', ' ').title(), key=symptom) else 0

    st.markdown("---")

    # Prediction button
    if st.button("Predict Virus", type="primary", use_container_width=True):
        # Check if at least one symptom is selected
        all_symptoms = list(sum(SYMPTOM_GROUPS.values(), []))
        selected_symptoms = sum([patient_data.get(symptom, 0) for symptom in all_symptoms])
        
        if selected_symptoms == 0:
            st.warning("Please select at least one symptom before making a prediction.")
        else:
            with st.spinner("Analyzing patient data..."):
                try:
                    # Create feature vector
                    X = create_feature_vector(patient_data)

                    # Make prediction
                    y_pred = model.predict(X)[0]
                    y_pred_proba = model.predict_proba(X)[0]

                    # Get top 5 predictions
                    top_5_indices = np.argsort(y_pred_proba)[-5:][::-1]

                    # Display results
                    st.success("Prediction Complete!")

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.subheader("Most Likely Virus")
                        st.metric(
                            label="Predicted Virus",
                            value=VIRUS_MAPPING[y_pred],
                            delta=f"{y_pred_proba[y_pred]*100:.2f}% confidence"
                        )

                    with col2:
                        st.subheader("Top 5 Predictions")
                        for rank, idx in enumerate(top_5_indices, 1):
                            virus_name = VIRUS_MAPPING[idx]
                            confidence = y_pred_proba[idx] * 100
                            st.write(f"{rank}. **{virus_name}**: {confidence:.2f}%")

                    # Display probability distribution
                    st.markdown("---")
                    st.subheader("Probability Distribution (Top 10)")

                    top_10_indices = np.argsort(y_pred_proba)[-10:][::-1]
                    prob_df = pd.DataFrame({
                        'Virus': [VIRUS_MAPPING[i] for i in top_10_indices],
                        'Probability (%)': [y_pred_proba[i]*100 for i in top_10_indices]
                    })
                    st.bar_chart(prob_df.set_index('Virus'))

                    # Feature summary
                    with st.expander("Input Summary"):
                        st.write("**Patient Demographics:**")
                        st.write(f"- Age: {patient_data['age']} years")
                        st.write(f"- Sex: {'Male' if patient_data['SEX'] == 1 else 'Female'}")
                        st.write(f"- Patient Type: {'Inpatient' if patient_data['PATIENTTYPE'] == 1 else 'Outpatient'}")
                        st.write(f"- Duration: {patient_data['duration_of_illness']} days")

                        active_symptoms = [k.replace('_', ' ').title() for k, v in patient_data.items() 
                                         if k in sum(SYMPTOM_GROUPS.values(), []) and v == 1]
                        st.write(f"\n**Active Symptoms ({len(active_symptoms)}):**")
                        if active_symptoms:
                            st.write(", ".join(active_symptoms))
                        else:
                            st.write("None reported")

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    import traceback
                    st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
