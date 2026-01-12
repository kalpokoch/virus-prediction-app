
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import xgboost as xgb

# Page configuration
st.set_page_config(
    page_title="Virus Detection System",
    page_icon="🦠",
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

# Other Virus sub-classification mapping (13 classes)
OTHER_VIRUS_MAPPING = {
    0: 'HIV',
    1: 'Haemophilus influenzae',
    2: 'Herpes simplex virus (HSV)',
    3: 'Human Bocavirus',
    4: 'Kyasanur Forest Disease',
    5: 'Metapneumovirus',
    6: 'Norovirus',
    7: 'Other Influenza',
    8: 'Rhinovirus',
    9: 'Toxoplasma',
    10: 'Unknown',
    11: 'West Nile virus (WNV)',
    12: 'Zika'
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
    """Load the trained XGBoost models"""
    try:
        with open('models/xgb_filtered_model.pkl', 'rb') as f:
            model1 = pickle.load(f)
        with open('models/xgb_filtered_M2_model.pkl', 'rb') as f:
            model2 = pickle.load(f)
        return model1, model2
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

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
    feature_df['durationofillness'] = feature_df['durationofillness'].fillna(0)

    # Fill all symptoms with 0
    symptom_cols = list(sum(SYMPTOM_GROUPS.values(), []))
    for col in symptom_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0
        feature_df[col] = feature_df[col].fillna(0).clip(0, 1)

    # === AGE FEATURES ===
    feature_df['agegroup'] = pd.cut(feature_df['age'], 
                                   bins=[0, 5, 18, 45, 65, 150], 
                                   labels=[0, 1, 2, 3, 4]).astype(int)
    feature_df['agegroup'] = feature_df['agegroup'].fillna(2)

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
    feature_df['ismonsoon'] = int(month in [6, 7, 8, 9])
    feature_df['iswinter'] = int(month in [12, 1, 2])

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
    feature_df['monsoon_respiratory'] = feature_df['ismonsoon'] * feature_df['respiratory_symptoms']
    feature_df['winter_respiratory'] = feature_df['iswinter'] * feature_df['respiratory_symptoms']
    feature_df['monsoon_fever'] = feature_df['ismonsoon'] * feature_df['FEVER']

    feature_df['state_season'] = patient_data['labstate'] * 10 + feature_df['season']
    feature_df['district_season'] = patient_data['districtencoded'] * 10 + feature_df['season']
    feature_df['district_month'] = patient_data['districtencoded'] * 100 + feature_df['month']

    feature_df['state_respiratory'] = patient_data['labstate'] * feature_df['respiratory_symptoms']
    feature_df['state_fever'] = patient_data['labstate'] * feature_df['FEVER']
    feature_df['state_gi'] = patient_data['labstate'] * feature_df['gi_symptoms']

    # Fever interactions
    feature_df['fever_respiratory'] = feature_df['FEVER'] * feature_df['respiratory_symptoms']
    feature_df['fever_gi'] = feature_df['FEVER'] * feature_df['gi_symptoms']
    feature_df['fever_neuro'] = feature_df['FEVER'] * feature_df['neuro_symptoms']
    feature_df['fever_skin'] = feature_df['FEVER'] * feature_df['skin_symptoms']
    feature_df['fever_duration'] = feature_df['FEVER'] * feature_df['durationofillness']
    feature_df['fever_headache'] = feature_df['FEVER'] * feature_df['HEADACHE']
    feature_df['fever_cough'] = feature_df['FEVER'] * feature_df['COUGH']

    # Severity & demographic interactions
    feature_df['severity_score'] = feature_df['symptom_count'] * feature_df['durationofillness']
    feature_df['age_symptom'] = feature_df['age'] * feature_df['symptom_count']
    feature_df['age_duration'] = feature_df['age'] * feature_df['durationofillness']
    feature_df['patienttype_age'] = patient_data['PATIENTTYPE'] * feature_df['agegroup']
    feature_df['sex_respiratory'] = patient_data['SEX'] * feature_df['respiratory_symptoms']
    feature_df['duration_symptom_ratio'] = feature_df['durationofillness'] / (feature_df['symptom_count'] + 1)

    # Year features (use current year)
    year = patient_data.get('year', 2024)
    feature_df['year'] = year
    feature_df['year_normalized'] = (year - 2012) / (2024 - 2012 + 1)  # Normalize based on training range

    # Quarter, week, day of year
    date = datetime(year, month, 1)
    feature_df['quarter'] = (month - 1) // 3 + 1
    feature_df['weekofyear'] = date.isocalendar()[1]
    feature_df['dayofyear'] = date.timetuple().tm_yday

    # Final cleanup
    feature_df = feature_df.replace([np.inf, -np.inf], 0).fillna(0)

    return feature_df.iloc[0].values.reshape(1, -1)

def main():
    st.title("🦠 Virus Detection and Classification System")
    st.markdown("---")
    st.write("Enter patient information and clinical symptoms to predict the most likely virus.")

    # Load models and mappings
    model1, model2 = load_model()
    if model1 is None or model2 is None:
        st.error("Failed to load models. Please check the model file paths.")
        return
    
    state_map, district_map, district_state_map = load_mappings()
    if state_map is None or district_map is None or district_state_map is None:
        st.error("Failed to load mapping files. Please check the CSV files.")
        return

    # Sidebar for patient demographics
    st.sidebar.header("📋 Patient Information")

    patient_data = {}

    # Demographics (MATCH EXACT TRAINING COLUMN NAMES)
    patient_data['age'] = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
    patient_data['SEX'] = st.sidebar.selectbox("Sex", options=[0, 1], 
                                                format_func=lambda x: "Female" if x == 0 else "Male")
    patient_data['PATIENTTYPE'] = st.sidebar.selectbox("Patient Type", options=[0, 1], 
                                                        format_func=lambda x: "Outpatient" if x == 0 else "Inpatient")
    patient_data['durationofillness'] = st.sidebar.number_input("Duration of Illness (days)", 
                                                                 min_value=0, max_value=365, value=3)
    
    # State selection with names
    state_names = state_map['state_name'].tolist()
    selected_state_name = st.sidebar.selectbox("State", options=state_names, index=0)
    patient_data['labstate'] = int(state_map[state_map['state_name'] == selected_state_name]['encoded_value'].values[0])
    
    # District selection filtered by state
    filtered_districts = district_state_map[district_state_map['state'] == selected_state_name]
    district_names = filtered_districts['district_name'].tolist()
    
    if len(district_names) > 0:
        selected_district_name = st.sidebar.selectbox("District", options=district_names, index=0)
        patient_data['districtencoded'] = int(filtered_districts[filtered_districts['district_name'] == selected_district_name]['district_encoded'].values[0])
    else:
        st.sidebar.warning("No districts available for selected state")
        patient_data['districtencoded'] = 0

    # Temporal features
    patient_data['month'] = st.sidebar.selectbox("Month of Illness", options=list(range(1, 13)), 
                                                  format_func=lambda x: datetime(2000, x, 1).strftime('%B'))
    patient_data['year'] = st.sidebar.number_input("Year", min_value=2012, max_value=2026, value=2024)

    # Main area for symptoms
    st.header("🩺 Clinical Symptoms")
    st.write("Select all symptoms present in the patient:")

    for group_name, symptoms in SYMPTOM_GROUPS.items():
        with st.expander(f"**{group_name} Symptoms**", expanded=True):
            cols = st.columns(3)
            for idx, symptom in enumerate(symptoms):
                with cols[idx % 3]:
                    patient_data[symptom] = 1 if st.checkbox(symptom.replace('_', ' ').title(), key=symptom) else 0

    st.markdown("---")

    # Prediction button
    if st.button("🔍 Predict Virus", type="primary", use_container_width=True):
        with st.spinner("Analyzing patient data..."):
            try:
                # Create feature vector
                X = create_feature_vector(patient_data)

                # Make prediction with Model 1
                y_pred = model1.predict(X)[0]
                y_pred_proba = model1.predict_proba(X)[0]

                # Get top 5 predictions
                top_5_indices = np.argsort(y_pred_proba)[-5:][::-1]

                # Check if "Other_Viruses" (class 15) is in top 5
                other_virus_in_top5 = 15 in top_5_indices
                second_model_results = None
                
                if other_virus_in_top5:
                    # Run second model for sub-classification
                    y_pred_m2 = model2.predict(X)[0]
                    y_pred_proba_m2 = model2.predict_proba(X)[0]
                    top_5_indices_m2 = np.argsort(y_pred_proba_m2)[-5:][::-1]
                    
                    second_model_results = {
                        'prediction': y_pred_m2,
                        'probabilities': y_pred_proba_m2,
                        'top_5': top_5_indices_m2
                    }

                # Display results
                st.success("✅ Prediction Complete!")

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("🎯 Most Likely Virus")
                    
                    # Check if primary prediction is Other_Viruses
                    if y_pred == 15 and second_model_results:
                        sub_virus = OTHER_VIRUS_MAPPING[second_model_results['prediction']]
                        sub_confidence = second_model_results['probabilities'][second_model_results['prediction']] * 100
                        st.metric(
                            label="Predicted Virus",
                            value=f"Other_Viruses → {sub_virus}",
                            delta=f"{y_pred_proba[y_pred]*100:.2f}% (M1) | {sub_confidence:.2f}% (M2)"
                        )
                    else:
                        st.metric(
                            label="Predicted Virus",
                            value=VIRUS_MAPPING[y_pred],
                            delta=f"{y_pred_proba[y_pred]*100:.2f}% confidence"
                        )

                with col2:
                    st.subheader("📊 Top 5 Predictions (Model 1)")
                    for rank, idx in enumerate(top_5_indices, 1):
                        virus_name = VIRUS_MAPPING[idx]
                        confidence = y_pred_proba[idx] * 100
                        
                        # Add indicator if this is Other_Viruses
                        if idx == 15 and second_model_results:
                            sub_virus = OTHER_VIRUS_MAPPING[second_model_results['prediction']]
                            st.write(f"{rank}. **{virus_name}** → *{sub_virus}*: {confidence:.2f}%")
                        else:
                            st.write(f"{rank}. **{virus_name}**: {confidence:.2f}%")
                
                # Display second model results if available
                if second_model_results:
                    st.markdown("---")
                    st.subheader("🔬 Other Viruses Sub-Classification (Model 2)")
                    st.info("Since 'Other_Viruses' appeared in top 5, secondary classification was performed.")
                    
                    col3, col4 = st.columns([1, 1])
                    
                    with col3:
                        st.write("**Top Prediction:**")
                        top_sub = OTHER_VIRUS_MAPPING[second_model_results['prediction']]
                        top_conf = second_model_results['probabilities'][second_model_results['prediction']] * 100
                        st.metric(label="Sub-Category", value=top_sub, delta=f"{top_conf:.2f}% confidence")
                    
                    with col4:
                        st.write("**Top 5 Sub-Categories:**")
                        for rank, idx in enumerate(second_model_results['top_5'], 1):
                            sub_virus = OTHER_VIRUS_MAPPING[idx]
                            sub_confidence = second_model_results['probabilities'][idx] * 100
                            st.write(f"{rank}. **{sub_virus}**: {sub_confidence:.2f}%")

                # Display probability distribution
                st.markdown("---")
                st.subheader("📈 Probability Distribution")
                
                tab1, tab2 = st.tabs(["Model 1 (Major Classes)", "Model 2 (Other Viruses)"]) if second_model_results else st.tabs(["Model 1 (Major Classes)"])
                
                with tab1:
                    st.write("**Top 10 Major Virus Categories**")
                    top_10_indices = np.argsort(y_pred_proba)[-10:][::-1]
                    prob_df = pd.DataFrame({
                        'Virus': [VIRUS_MAPPING[i] for i in top_10_indices],
                        'Probability (%)': [y_pred_proba[i]*100 for i in top_10_indices]
                    })
                    st.bar_chart(prob_df.set_index('Virus'))
                
                if second_model_results:
                    with tab2:
                        st.write("**Top 10 Other Virus Sub-Categories**")
                        top_10_indices_m2 = np.argsort(second_model_results['probabilities'])[-10:][::-1]
                        prob_df_m2 = pd.DataFrame({
                            'Virus': [OTHER_VIRUS_MAPPING[i] for i in top_10_indices_m2],
                            'Probability (%)': [second_model_results['probabilities'][i]*100 for i in top_10_indices_m2]
                        })
                        st.bar_chart(prob_df_m2.set_index('Virus'))

                # Feature summary
                with st.expander("📋 Input Summary"):
                    st.write("**Patient Demographics:**")
                    st.write(f"- Age: {patient_data['age']} years")
                    st.write(f"- Sex: {'Male' if patient_data['SEX'] == 1 else 'Female'}")
                    st.write(f"- Patient Type: {'Inpatient' if patient_data['PATIENTTYPE'] == 1 else 'Outpatient'}")
                    st.write(f"- Duration: {patient_data['durationofillness']} days")

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
