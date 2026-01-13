import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import xgboost as xgb
from numba import jit


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
    3: 'Human papillomavirus (HPV)',
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


# Pre-computed lookup tables for performance optimization
MONTH_TO_SEASON = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 0}
MONTH_TO_QUARTER = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
MONTH_TO_WEEK = {1: 2, 2: 6, 3: 10, 4: 14, 5: 18, 6: 23, 7: 27, 8: 31, 9: 36, 10: 40, 11: 44, 12: 49}
MONTH_SIN = {m: np.sin(2 * np.pi * m / 12) for m in range(1, 13)}
MONTH_COS = {m: np.cos(2 * np.pi * m / 12) for m in range(1, 13)}
MONTH_TO_DAY = {1: 15, 2: 45, 3: 74, 4: 105, 5: 135, 6: 166, 7: 196, 8: 227, 9: 258, 10: 288, 11: 319, 12: 349}

# Pre-computed age groups for vectorized computation
AGE_GROUPS = np.array([0, 5, 18, 45, 65, 120])

# Pre-compute symptom column list (shared across all feature vector creations)
SYMPTOM_COLS = list(sum(SYMPTOM_GROUPS.values(), []))

# Pre-computed symptom indices for fast array slicing
RESPIRATORY_IDX = np.array([SYMPTOM_COLS.index(s) for s in ['COUGH', 'BREATHLESSNESS', 'RHINORRHEA', 'SORETHROAT']])
GI_IDX = np.array([SYMPTOM_COLS.index(s) for s in ['DIARRHEA', 'DYSENTERY', 'NAUSEA', 'VOMITING', 'ABDOMINALPAIN']])
NEURO_IDX = np.array([SYMPTOM_COLS.index(s) for s in ['HEADACHE', 'ALTEREDSENSORIUM', 'SEIZURES', 'SOMNOLENCE', 'NECKRIGIDITY', 'IRRITABLITY']])
SKIN_IDX = np.array([SYMPTOM_COLS.index(s) for s in ['PAPULARRASH', 'PUSTULARRASH', 'MACULOPAPULARRASH', 'BULLAE']])
SYSTEMIC_IDX = np.array([SYMPTOM_COLS.index(s) for s in ['MYALGIA', 'ARTHRALGIA', 'CHILLS', 'RIGORS', 'MALAISE']])
COUNT_IDX = np.array([SYMPTOM_COLS.index(s) for s in ['HEADACHE', 'FEVER', 'COUGH', 'VOMITING', 'DIARRHEA', 'MYALGIA', 'ARTHRALGIA', 'NAUSEA', 'BREATHLESSNESS', 'SORETHROAT']])

# Individual symptom indices
FEVER_IDX = SYMPTOM_COLS.index('FEVER')
HEADACHE_IDX = SYMPTOM_COLS.index('HEADACHE')
COUGH_IDX = SYMPTOM_COLS.index('COUGH')


# OPTIMIZATION: Lazy loading - only load primary model on startup
@st.cache_resource
def load_primary_model():
    """Load the primary XGBoost model (26 virus classes)"""
    try:
        with open('models/xgb_filtered_model.pkl', 'rb') as f:
            model1 = pickle.load(f)
        return model1
    except Exception as e:
        st.error(f"Error loading primary model: {e}")
        return None


# OPTIMIZATION: Lazy loading - only load secondary model when needed
@st.cache_resource
def load_secondary_model():
    """Load the secondary XGBoost model (13 other virus sub-classes)"""
    try:
        with open('models/xgb_filtered_M2_model.pkl', 'rb') as f:
            model2 = pickle.load(f)
        return model2
    except Exception as e:
        st.error(f"Error loading secondary model: {e}")
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


# OPTIMIZATION: Numba JIT compilation for interaction feature calculations
@jit(nopython=True)
def compute_interaction_features(symptoms, fever, respiratory_sum, gi_sum, neuro_sum, 
                                 skin_sum, symptom_count, headache, cough,
                                 age, duration, labstate, district, month, 
                                 season, ismonsoon, iswinter, sex, patienttype, agegroup):
    """
    Compute interaction features using Numba for speed
    Returns array of 24 interaction features
    """
    interactions = np.empty(24, dtype=np.float32)
    
    # Geo-temporal interactions
    interactions[0] = ismonsoon * respiratory_sum  # monsoon_respiratory
    interactions[1] = iswinter * respiratory_sum   # winter_respiratory
    interactions[2] = ismonsoon * fever            # monsoon_fever
    interactions[3] = labstate * 10 + season       # state_season
    interactions[4] = district * 10 + season       # district_season
    interactions[5] = district * 100 + month       # district_month
    
    # State-symptom interactions
    interactions[6] = labstate * respiratory_sum   # state_respiratory
    interactions[7] = labstate * fever             # state_fever
    interactions[8] = labstate * gi_sum            # state_gi
    
    # Fever-symptom interactions
    interactions[9] = fever * respiratory_sum      # fever_respiratory
    interactions[10] = fever * gi_sum              # fever_gi
    interactions[11] = fever * neuro_sum           # fever_neuro
    interactions[12] = fever * skin_sum            # fever_skin
    interactions[13] = fever * duration            # fever_duration
    interactions[14] = fever * headache            # fever_headache
    interactions[15] = fever * cough               # fever_cough
    
    # Severity and demographic interactions
    interactions[16] = symptom_count * duration    # severity_score
    interactions[17] = age * symptom_count         # age_symptom
    interactions[18] = age * duration              # age_duration
    interactions[19] = patienttype * agegroup      # patienttype_age
    interactions[20] = sex * respiratory_sum       # sex_respiratory
    interactions[21] = duration / (symptom_count + 1)  # duration_symptom_ratio
    
    # Placeholder for future features
    interactions[22] = 0.0
    interactions[23] = 0.0
    
    return interactions


def create_feature_vector(patient_data):
    """
    OPTIMIZED: Convert user inputs → 80 model features using vectorized operations
    """
    # Extract base values
    age = min(max(patient_data.get('age', 30), 0), 120)
    duration = max(patient_data.get('durationofillness', 0), 0)
    month = patient_data.get('month', 1)
    year = patient_data.get('year', 2024)
    labstate = patient_data['labstate']
    district = patient_data['districtencoded']
    sex = patient_data['SEX']
    patienttype = patient_data['PATIENTTYPE']
    
    # Get symptoms using direct lookup (vectorized)
    symptoms = np.array([patient_data.get(s, 0) for s in SYMPTOM_COLS], dtype=np.float32)
    
    # OPTIMIZATION: Vectorized age group calculation
    agegroup = np.digitize(age, AGE_GROUPS) - 1
    agegroup = min(max(agegroup, 0), 4)  # Clamp to [0, 4]
    
    # OPTIMIZATION: Pre-indexed symptom group sums (vectorized)
    respiratory_symptoms = symptoms[RESPIRATORY_IDX].sum()
    gi_symptoms = symptoms[GI_IDX].sum()
    neuro_symptoms = symptoms[NEURO_IDX].sum()
    skin_symptoms = symptoms[SKIN_IDX].sum()
    systemic_symptoms = symptoms[SYSTEMIC_IDX].sum()
    symptom_count = symptoms[COUNT_IDX].sum()
    symptom_diversity = (symptoms[COUNT_IDX] > 0).sum()
    
    # Fast symptom access
    fever = symptoms[FEVER_IDX]
    headache = symptoms[HEADACHE_IDX]
    cough = symptoms[COUGH_IDX]
    
    # Geo-temporal features (using pre-computed lookups)
    season = MONTH_TO_SEASON[month]
    ismonsoon = 1.0 if month in [6, 7, 8, 9] else 0.0
    iswinter = 1.0 if month in [12, 1, 2] else 0.0
    month_sin = MONTH_SIN[month]
    month_cos = MONTH_COS[month]
    quarter = MONTH_TO_QUARTER[month]
    weekofyear = MONTH_TO_WEEK[month]
    dayofyear = MONTH_TO_DAY[month]
    
    # OPTIMIZATION: Compute interaction features using Numba JIT
    interactions = compute_interaction_features(
        symptoms, fever, respiratory_symptoms, gi_symptoms, neuro_symptoms,
        skin_symptoms, symptom_count, headache, cough,
        age, duration, labstate, district, month,
        season, ismonsoon, iswinter, sex, patienttype, agegroup
    )
    
    year_normalized = (year - 2012) / 13.0
    
    # Build feature vector directly (no DataFrame overhead)
    feature_vector = np.array([
        # Demographics & Clinical (5)
        labstate, age, sex, patienttype, duration,
        
        # Symptoms (33) - in exact training order
        symptoms[SYMPTOM_COLS.index('HEADACHE')],
        symptoms[SYMPTOM_COLS.index('IRRITABLITY')],
        symptoms[SYMPTOM_COLS.index('ALTEREDSENSORIUM')],
        symptoms[SYMPTOM_COLS.index('SOMNOLENCE')],
        symptoms[SYMPTOM_COLS.index('NECKRIGIDITY')],
        symptoms[SYMPTOM_COLS.index('SEIZURES')],
        symptoms[SYMPTOM_COLS.index('DIARRHEA')],
        symptoms[SYMPTOM_COLS.index('DYSENTERY')],
        symptoms[SYMPTOM_COLS.index('NAUSEA')],
        symptoms[SYMPTOM_COLS.index('MALAISE')],
        symptoms[SYMPTOM_COLS.index('MYALGIA')],
        symptoms[SYMPTOM_COLS.index('ARTHRALGIA')],
        symptoms[SYMPTOM_COLS.index('CHILLS')],
        symptoms[SYMPTOM_COLS.index('RIGORS')],
        symptoms[SYMPTOM_COLS.index('BREATHLESSNESS')],
        symptoms[SYMPTOM_COLS.index('COUGH')],
        symptoms[SYMPTOM_COLS.index('RHINORRHEA')],
        symptoms[SYMPTOM_COLS.index('SORETHROAT')],
        symptoms[SYMPTOM_COLS.index('BULLAE')],
        symptoms[SYMPTOM_COLS.index('PAPULARRASH')],
        symptoms[SYMPTOM_COLS.index('PUSTULARRASH')],
        symptoms[SYMPTOM_COLS.index('MUSCULARRASH')],
        symptoms[SYMPTOM_COLS.index('MACULOPAPULARRASH')],
        symptoms[SYMPTOM_COLS.index('ESCHAR')],
        symptoms[SYMPTOM_COLS.index('DARKURINE')],
        symptoms[SYMPTOM_COLS.index('HEPATOMEGALY')],
        symptoms[SYMPTOM_COLS.index('REDEYE')],
        symptoms[SYMPTOM_COLS.index('DISCHARGEEYES')],
        symptoms[SYMPTOM_COLS.index('CRUSHINGEYES')],
        symptoms[SYMPTOM_COLS.index('JAUNDICE')],
        fever,
        symptoms[SYMPTOM_COLS.index('ABDOMINALPAIN')],
        symptoms[SYMPTOM_COLS.index('VOMITING')],
        
        # Geo-temporal (10)
        month, year, quarter, weekofyear, dayofyear,
        ismonsoon, iswinter, month_sin, month_cos, district,
        
        # Derived features (32 more to reach 80)
        agegroup,
        symptom_count, respiratory_symptoms, gi_symptoms, neuro_symptoms, 
        skin_symptoms, systemic_symptoms, symptom_diversity,
        season,
        interactions[0],   # monsoon_respiratory
        interactions[1],   # winter_respiratory
        interactions[2],   # monsoon_fever
        interactions[3],   # state_season
        interactions[4],   # district_season
        interactions[5],   # district_month
        interactions[6],   # state_respiratory
        interactions[7],   # state_fever
        interactions[8],   # state_gi
        interactions[9],   # fever_respiratory
        interactions[10],  # fever_gi
        interactions[11],  # fever_neuro
        interactions[12],  # fever_skin
        interactions[13],  # fever_duration
        interactions[14],  # fever_headache
        interactions[15],  # fever_cough
        interactions[16],  # severity_score
        interactions[17],  # age_symptom
        interactions[18],  # age_duration
        interactions[19],  # patienttype_age
        interactions[20],  # sex_respiratory
        interactions[21],  # duration_symptom_ratio
        year_normalized
    ], dtype=np.float32)
    
    return feature_vector.reshape(1, -1)


def predict_with_model(model, X, use_dmatrix=True):
    """
    OPTIMIZATION: Unified prediction function using DMatrix for speed
    """
    if use_dmatrix:
        # Convert to DMatrix for faster inference
        dmatrix = xgb.DMatrix(X)
        
        # Check if model is Booster or Classifier
        if hasattr(model, 'get_booster'):
            # XGBClassifier - get booster and predict
            booster = model.get_booster()
            y_pred_proba = booster.predict(dmatrix)
        else:
            # Direct Booster object
            y_pred_proba = model.predict(dmatrix)
    else:
        # Fallback to standard predict_proba
        y_pred_proba = model.predict_proba(X)[0]
        return y_pred_proba
    
    # Handle output shape
    if len(y_pred_proba.shape) == 1:
        return y_pred_proba
    else:
        return y_pred_proba[0]


def main():
    st.title("🦠 Virus Detection and Classification System")
    st.markdown("---")
    st.write("Enter patient information and clinical symptoms to predict the most likely virus.")

    # OPTIMIZATION: Load only primary model on startup
    model1 = load_primary_model()
    if model1 is None:
        st.error("Failed to load primary model. Please check the model file path.")
        return
    
    state_map, district_map, district_state_map = load_mappings()
    if state_map is None or district_map is None or district_state_map is None:
        st.error("Failed to load mapping files. Please check the CSV files.")
        return

    # OPTIMIZATION: Initialize session state for caching
    if 'last_patient_data' not in st.session_state:
        st.session_state.last_patient_data = None
    if 'last_features' not in st.session_state:
        st.session_state.last_features = None
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None

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
                # OPTIMIZATION: Check if we can reuse cached features
                patient_data_hash = str(sorted(patient_data.items()))
                if (st.session_state.last_patient_data == patient_data_hash and 
                    st.session_state.last_features is not None):
                    X = st.session_state.last_features
                else:
                    X = create_feature_vector(patient_data)
                    st.session_state.last_features = X
                    st.session_state.last_patient_data = patient_data_hash

                # OPTIMIZATION: Predict using DMatrix
                y_pred_proba = predict_with_model(model1, X, use_dmatrix=True)

                # Handle potential shape issues
                if len(y_pred_proba.shape) > 1:
                    y_pred_proba = y_pred_proba[0]
                
                y_pred = np.argmax(y_pred_proba)

                # Get top 5 predictions
                top_5_indices = np.argsort(y_pred_proba)[-5:][::-1]

                # OPTIMIZATION: Lazy load Model 2 only if "Other_Viruses" in top 5
                other_virus_in_top5 = 15 in top_5_indices
                second_model_results = None
                
                if other_virus_in_top5:
                    model2 = load_secondary_model()  # Lazy load
                    if model2 is not None:
                        # Run second model for sub-classification
                        y_pred_proba_m2 = predict_with_model(model2, X, use_dmatrix=True)
                        
                        # Handle potential shape issues
                        if len(y_pred_proba_m2.shape) > 1:
                            y_pred_proba_m2 = y_pred_proba_m2[0]
                        
                        y_pred_m2 = np.argmax(y_pred_proba_m2)
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
                
                if second_model_results:
                    tab1, tab2 = st.tabs(["Model 1 (Major Classes)", "Model 2 (Other Viruses)"])
                else:
                    tab1 = st.tabs(["Model 1 (Major Classes)"])[0]
                
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
