import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import xgboost as xgb

# Database imports (minimal addition)
from data_handler import save_prediction_to_db, get_db_health, get_prediction_stats, save_validation_to_db


# Page configuration - with error handling for deployment consistency
try:
    st.set_page_config(
        page_title="Virus Detection System",
        page_icon="🦠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as config_error:
    # Fallback for deployment issues
    st.set_page_config(
        page_title="Virus Detection System",
        layout="wide"
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


# Combined virus mapping for validation dropdown (Main + Other categories)
COMBINED_VIRUS_MAPPING = {
    # Main virus categories (0-25)
    **{f"main_{k}": v for k, v in VIRUS_MAPPING.items() if k != 15},  # Exclude OtherViruses
    # Other virus subcategories with prefix
    **{f"other_{k}": f"Other Viruses → {v}" for k, v in OTHER_VIRUS_MAPPING.items()}
}


# Symptom groups
SYMPTOM_GROUPS = {
    "Neurological": ['HEADACHE', 'IRRITABILITY', 'ALTERED SENSORIUM', 'SOMNOLENCE', 
                     'NECK RIGIDITY', 'SEIZURES'],
    "Gastrointestinal": ['DIARRHEA', 'DYSENTERY', 'NAUSEA', 'VOMITING', 'ABDOMINAL PAIN'],
    "Systemic": ['MALAISE', 'MYALGIA', 'ARTHRALGIA', 'CHILLS', 'RIGORS', 'FEVER'],
    "Respiratory": ['BREATHLESSNESS', 'COUGH', 'RHINORRHEA', 'SORE THROAT'],
    "Dermatological": ['BULLAE', 'PAPULAR RASH', 'PUSTULAR RASH', 'MUSCULAR RASH', 
                       'MACULOPAPULAR RASH', 'ESCHAR'],
    "Hepatic/Other": ['DARK URINE', 'HEPATOMEGALY', 'JAUNDICE'],
    "Ocular": ['RED EYE', 'DISCHARGE EYES', 'CRUSHING EYES']
}


# Pre-computed lookup tables for performance optimization
MONTH_TO_SEASON = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 0}
MONTH_TO_QUARTER = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
MONTH_TO_WEEK = {1: 2, 2: 6, 3: 10, 4: 14, 5: 18, 6: 23, 7: 27, 8: 31, 9: 36, 10: 40, 11: 44, 12: 49}
MONTH_SIN = {m: np.sin(2 * np.pi * m / 12) for m in range(1, 13)}
MONTH_COS = {m: np.cos(2 * np.pi * m / 12) for m in range(1, 13)}
MONTH_TO_DAY = {1: 15, 2: 45, 3: 74, 4: 105, 5: 135, 6: 166, 7: 196, 8: 227, 9: 258, 10: 288, 11: 319, 12: 349}


@st.cache_resource
def load_model():
    """Load the trained XGBoost models"""
    try:
        with open('models/XGB_M1_16JAN.pkl', 'rb') as f:
            model1 = pickle.load(f)
        with open('models/xgb_filtered_model2.pkl', 'rb') as f:
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
    Optimized: Convert user inputs → 80 model features using direct numpy operations
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

    # Get symptoms using direct lookup (much faster than DataFrame)
    symptom_cols = list(sum(SYMPTOM_GROUPS.values(), []))
    symptoms = np.array([patient_data.get(s, 0) for s in symptom_cols], dtype=np.float32)

    # Age group calculation (direct bins)
    if age <= 5: agegroup = 0
    elif age <= 18: agegroup = 1
    elif age <= 45: agegroup = 2
    elif age <= 65: agegroup = 3
    else: agegroup = 4

    # Pre-defined symptom indices for fast array slicing
    respiratory_idx = [symptom_cols.index(s) for s in ['COUGH', 'BREATHLESSNESS', 'RHINORRHEA', 'SORE THROAT']]
    gi_idx = [symptom_cols.index(s) for s in ['DIARRHEA', 'DYSENTERY', 'NAUSEA', 'VOMITING', 'ABDOMINAL PAIN']]
    neuro_idx = [symptom_cols.index(s) for s in ['HEADACHE', 'ALTERED SENSORIUM', 'SEIZURES', 'SOMNOLENCE', 'NECK RIGIDITY', 'IRRITABILITY']]
    skin_idx = [symptom_cols.index(s) for s in ['PAPULAR RASH', 'PUSTULAR RASH', 'MACULOPAPULAR RASH', 'BULLAE']]
    systemic_idx = [symptom_cols.index(s) for s in ['MYALGIA', 'ARTHRALGIA', 'CHILLS', 'RIGORS', 'MALAISE']]
    count_idx = [symptom_cols.index(s) for s in ['HEADACHE', 'FEVER', 'COUGH', 'VOMITING', 'DIARRHEA', 'MYALGIA', 'ARTHRALGIA', 'NAUSEA', 'BREATHLESSNESS', 'SORE THROAT']]

    # Symptom group sums (vectorized)
    respiratory_symptoms = symptoms[respiratory_idx].sum()
    gi_symptoms = symptoms[gi_idx].sum()
    neuro_symptoms = symptoms[neuro_idx].sum()
    skin_symptoms = symptoms[skin_idx].sum()
    systemic_symptoms = symptoms[systemic_idx].sum()
    symptom_count = symptoms[count_idx].sum()
    symptom_diversity = (symptoms[count_idx] > 0).sum()

    # Fast symptom access
    fever = symptoms[symptom_cols.index('FEVER')]
    headache = symptoms[symptom_cols.index('HEADACHE')]
    cough = symptoms[symptom_cols.index('COUGH')]

    # Geo-temporal features (using pre-computed lookups)
    season = MONTH_TO_SEASON[month]
    ismonsoon = 1 if month in [6, 7, 8, 9] else 0
    iswinter = 1 if month in [12, 1, 2] else 0
    month_sin = MONTH_SIN[month]
    month_cos = MONTH_COS[month]
    quarter = MONTH_TO_QUARTER[month]
    weekofyear = MONTH_TO_WEEK[month]
    dayofyear = MONTH_TO_DAY[month]

    # Interaction features (all vectorized)
    monsoon_respiratory = ismonsoon * respiratory_symptoms
    winter_respiratory = iswinter * respiratory_symptoms
    monsoon_fever = ismonsoon * fever

    state_season = labstate * 10 + season
    district_season = district * 10 + season
    district_month = district * 100 + month

    state_respiratory = labstate * respiratory_symptoms
    state_fever = labstate * fever
    state_gi = labstate * gi_symptoms

    fever_respiratory = fever * respiratory_symptoms
    fever_gi = fever * gi_symptoms
    fever_neuro = fever * neuro_symptoms
    fever_skin = fever * skin_symptoms
    fever_duration = fever * duration
    fever_headache = fever * headache
    fever_cough = fever * cough

    severity_score = symptom_count * duration
    age_symptom = age * symptom_count
    age_duration = age * duration
    patienttype_age = patienttype * agegroup
    sex_respiratory = sex * respiratory_symptoms
    duration_symptom_ratio = duration / (symptom_count + 1)

    year_normalized = (year - 2012) / 13.0

    # Build feature vector directly (no DataFrame overhead)
    # Note: Must include ALL 80 features in exact training order
    feature_vector = np.array([
        # Demographics & Clinical (5)
        labstate, age, sex, patienttype, duration,

        # Symptoms (33) - in exact training order
        symptoms[symptom_cols.index('HEADACHE')],
        symptoms[symptom_cols.index('IRRITABILITY')],
        symptoms[symptom_cols.index('ALTERED SENSORIUM')],
        symptoms[symptom_cols.index('SOMNOLENCE')],
        symptoms[symptom_cols.index('NECK RIGIDITY')],
        symptoms[symptom_cols.index('SEIZURES')],
        symptoms[symptom_cols.index('DIARRHEA')],
        symptoms[symptom_cols.index('DYSENTERY')],
        symptoms[symptom_cols.index('NAUSEA')],
        symptoms[symptom_cols.index('MALAISE')],
        symptoms[symptom_cols.index('MYALGIA')],
        symptoms[symptom_cols.index('ARTHRALGIA')],
        symptoms[symptom_cols.index('CHILLS')],
        symptoms[symptom_cols.index('RIGORS')],
        symptoms[symptom_cols.index('BREATHLESSNESS')],
        symptoms[symptom_cols.index('COUGH')],
        symptoms[symptom_cols.index('RHINORRHEA')],
        symptoms[symptom_cols.index('SORE THROAT')],
        symptoms[symptom_cols.index('BULLAE')],
        symptoms[symptom_cols.index('PAPULAR RASH')],
        symptoms[symptom_cols.index('PUSTULAR RASH')],
        symptoms[symptom_cols.index('MUSCULAR RASH')],
        symptoms[symptom_cols.index('MACULOPAPULAR RASH')],
        symptoms[symptom_cols.index('ESCHAR')],
        symptoms[symptom_cols.index('DARK URINE')],
        symptoms[symptom_cols.index('HEPATOMEGALY')],
        symptoms[symptom_cols.index('RED EYE')],
        symptoms[symptom_cols.index('DISCHARGE EYES')],
        symptoms[symptom_cols.index('CRUSHING EYES')],
        symptoms[symptom_cols.index('JAUNDICE')],
        fever,
        symptoms[symptom_cols.index('ABDOMINAL PAIN')],
        symptoms[symptom_cols.index('VOMITING')],

        # Geo-temporal (10)
        month, year, quarter, weekofyear, dayofyear,
        ismonsoon, iswinter, month_sin, month_cos, district,

        # Derived features (32 more to reach 80)
        agegroup,
        symptom_count, respiratory_symptoms, gi_symptoms, neuro_symptoms, 
        skin_symptoms, systemic_symptoms, symptom_diversity,
        season,
        monsoon_respiratory, winter_respiratory, monsoon_fever,
        state_season, district_season, district_month,
        state_respiratory, state_fever, state_gi,
        fever_respiratory, fever_gi, fever_neuro, fever_skin,
        fever_duration, fever_headache, fever_cough,
        severity_score, age_symptom, age_duration,
        patienttype_age, sex_respiratory, duration_symptom_ratio,
        year_normalized
    ], dtype=np.float32)

    return feature_vector.reshape(1, -1)


def main():
    # Top logos
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        try:
            st.image("logo_1.jpeg", width=300)
        except:
            st.write("")  # Skip if image not found
    with col3:
        try:
            st.image("Amity_logo2.png", width=250)
        except:
            st.write("")  # Skip if image not found
    with col2:
        try:
            st.image("logo_2.jpeg", width=250)
        except:
            st.write("")  # Skip if image not found
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Home", "Prediction", "About"])

    if page == "Home":
        st.markdown(
            "<h1 style='text-align: center;'>🦠 Virus Detection and Classification System</h1>",
            unsafe_allow_html=True)
        st.markdown(
            "<h2 style='text-align: center;'>Advanced AI-Powered Diagnostic Tool for Viral Infections</h2>",
            unsafe_allow_html=True)
        st.write("""
        Welcome to the Virus Detection and Classification System!
        
        This advanced AI-driven system assists healthcare professionals by analyzing patient symptoms 
        and demographic information to predict the most probable viral infection from a comprehensive 
        database of 26+ virus categories.
        
        **Key Features:**
        - **Dual-Model Architecture**: Primary classification for major virus categories and secondary classification for "Other Viruses"
        - **Comprehensive Symptom Analysis**: Covers neurological, gastrointestinal, respiratory, dermatological, and systemic symptoms
        - **Geo-temporal Intelligence**: Incorporates seasonal patterns and geographical factors
        - **Real-time Predictions**: Instant probability scores and confidence metrics
        
        Navigate to the **Prediction** page using the sidebar to input patient details 
        and get comprehensive virus classification results.
        """)
        st.warning("**Medical Disclaimer**: This system is designed to assist healthcare professionals and should not be used as a substitute for professional medical diagnosis, treatment, or advice. Always consult qualified medical personnel for patient care decisions.")

    elif page == "About":
        st.title("About Virus Detection System")
        st.write("""
        ### System Overview
        This application utilizes advanced machine learning techniques to analyze patient symptoms 
        and predict viral infections with high accuracy.
        
        ### Technical Specifications
        - **Primary Model**: XGBoost classifier for 26 major virus categories
        - **Secondary Model**: Specialized classifier for "Other Viruses" subcategorization  
        - **Feature Engineering**: 80+ engineered features including temporal, geographical, and symptom interaction variables
        - **Optimization**: Cached models and pre-computed lookup tables for real-time performance
        
        ### Supported Virus Categories
        The system can identify and classify the following major virus categories:
        - Dengue Virus, Chikungunya Virus, Japanese Encephalitis
        - Hepatitis A/B/C/E Viruses
        - Influenza variants (H1N1, H3N2, Victoria)
        - Respiratory viruses (RSV, Adenovirus, SARS-CoV-2)
        - And many more...
        
        ### Data Sources
        The models are trained on comprehensive clinical datasets with proper encoding 
        for states, districts, and symptom combinations to ensure accurate predictions 
        across different geographical regions.
        """)
        st.warning("**Medical Disclaimer**: This system provides diagnostic assistance and should not replace professional medical evaluation and treatment decisions.")

    elif page == "Prediction":
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
        st.sidebar.header("Patient Information")

        patient_data = {}

        # Demographics (MATCH EXACT TRAINING COLUMN NAMES)
        patient_data['age'] = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
        patient_data['SEX'] = st.sidebar.selectbox("Sex", options=[0, 1], 
                                                    format_func=lambda x: "Female" if x == 0 else "Male", index=1)
        patient_data['PATIENTTYPE'] = st.sidebar.selectbox("Patient Type", options=[0, 1], 
                                                            format_func=lambda x: "Outpatient" if x == 0 else "Inpatient", index=1)
        patient_data['durationofillness'] = st.sidebar.number_input("Duration of Illness (days)", 
                                                                     min_value=0, max_value=365, value=3)

        # State selection with names
        state_names = state_map['state_name'].tolist()
        # Set Tamil Nadu as default if available, otherwise use first state
        default_state_index = 0
        if 'Tamil Nadu' in state_names:
            default_state_index = state_names.index('Tamil Nadu')
        selected_state_name = st.sidebar.selectbox("State", options=state_names, index=default_state_index)
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
        patient_data['year'] = st.sidebar.number_input("Year", min_value=2012, max_value=2026, value=datetime.now().year)

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
            all_symptoms = sum(SYMPTOM_GROUPS.values(), [])
            symptoms_selected = any(patient_data.get(symptom, 0) == 1 for symptom in all_symptoms)
            
            if not symptoms_selected:
                st.warning("Please select at least one symptom before making a prediction.")
                st.info("Expand the symptom groups above and check the boxes for symptoms present in the patient.")
            else:
                with st.spinner("Analyzing patient data..."):
                    try:
                        # Create feature vector
                        X = create_feature_vector(patient_data)

                        # Make prediction with Model 1 (use only predict_proba for speed)
                        y_pred_proba = model1.predict_proba(X)[0]
                        y_pred = np.argmax(y_pred_proba)

                        # Get top 5 predictions
                        top_5_indices = np.argsort(y_pred_proba)[-5:][::-1]

                        # Check if "Other_Viruses" (class 15) is in top 5
                        other_virus_in_top5 = 15 in top_5_indices
                        second_model_results = None

                        if other_virus_in_top5:
                            # Run second model for sub-classification (use only predict_proba)
                            y_pred_proba_m2 = model2.predict_proba(X)[0]
                            y_pred_m2 = np.argmax(y_pred_proba_m2)
                            top_5_indices_m2 = np.argsort(y_pred_proba_m2)[-5:][::-1]

                            second_model_results = {
                                'prediction': y_pred_m2,
                                'probabilities': y_pred_proba_m2,
                                'top_5': top_5_indices_m2
                            }

                        # Save prediction results to session state for validation form
                        st.session_state['prediction_results'] = {
                            'y_pred': y_pred,
                            'y_pred_proba': y_pred_proba,
                            'top_5_indices': top_5_indices,
                            'second_model_results': second_model_results,
                            'patient_data': patient_data.copy(),
                            'selected_state_name': selected_state_name,
                            'selected_district_name': selected_district_name
                        }

                        # Display results
                        st.success("Prediction Complete!")
                        
                        # Save prediction to database (minimal intrusion)
                        try:
                            # Prepare prediction result for database
                            prediction_result = {
                                'predicted_virus': VIRUS_MAPPING[y_pred],
                                'predicted_virus_id': int(y_pred),
                                'confidence': float(y_pred_proba[y_pred] * 100),
                                'top_5_predictions': [
                                    {
                                        'virus': VIRUS_MAPPING[idx],
                                        'virus_id': int(idx),
                                        'confidence': float(y_pred_proba[idx] * 100)
                                    } for idx in top_5_indices
                                ]
                            }
                            
                            # Add second model results if available
                            if second_model_results:
                                prediction_result['sub_classification'] = {
                                    'predicted_sub_virus': OTHER_VIRUS_MAPPING[second_model_results['prediction']],
                                    'predicted_sub_virus_id': int(second_model_results['prediction']),
                                    'sub_confidence': float(second_model_results['probabilities'][second_model_results['prediction']] * 100),
                                    'top_5_sub_predictions': [
                                        {
                                            'virus': OTHER_VIRUS_MAPPING[idx],
                                            'virus_id': int(idx),
                                            'confidence': float(second_model_results['probabilities'][idx] * 100)
                                        } for idx in second_model_results['top_5']
                                    ]
                                }
                            
                            # Save to database (non-blocking)
                            saved_id = save_prediction_to_db(
                                patient_data=patient_data,
                                prediction_result=prediction_result,
                                model_info={'model1': 'XGB_M1_16JAN', 'model2': 'xgb_filtered_model2'}
                            )
                            
                            # Store saved_id in session state for validation
                            st.session_state['saved_id'] = saved_id
                            
                            if saved_id:
                                st.sidebar.success(f"🗄️ Prediction saved (ID: {saved_id[:8]}...)")
                            else:
                                st.sidebar.warning("⚠️ Could not save prediction to database")
                                
                        except Exception as db_error:
                            # Don't let database errors break the prediction display
                            st.sidebar.warning("⚠️ Database save failed")
                            st.sidebar.caption(f"Error: {str(db_error)[:50]}...")

                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.subheader("Most Likely Virus")

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
                            st.subheader("Top 5 Predictions (Model 1)")
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
                            st.subheader("Other Viruses Sub-Classification (Model 2)")
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
                        st.subheader("Probability Distribution")

                        if second_model_results:
                            tab1, tab2 = st.tabs(["Model 1 (Major Classes)", "Model 2 (Other Viruses)"])
                        else:
                            tabs = st.tabs(["Model 1 (Major Classes)"])
                            tab1 = tabs[0]

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
                        with st.expander("Input Summary"):
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

                        st.warning("**Medical Disclaimer**: This prediction is generated by AI and should be used only as a diagnostic aid. Always consult with qualified healthcare professionals for proper medical diagnosis and treatment decisions.")

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        import traceback
                        st.error(traceback.format_exc())

        # Validation Section - Show if prediction results exist in session state
        if 'prediction_results' in st.session_state and 'saved_id' in st.session_state and st.session_state['saved_id']:
            st.markdown("---")
            st.subheader("🩺 Medical Validation (Optional)")
            st.info("Help us improve the AI model by providing the actual diagnosis. This data will be used for model validation and improvement.")
            
            # Get prediction results from session state
            pred_results = st.session_state['prediction_results']
            saved_id = st.session_state['saved_id']
            
            # Create validation form
            with st.form(key=f"validation_form_{saved_id}"):
                col_val1, col_val2 = st.columns([2, 1])
                
                with col_val1:
                    # Combined dropdown with all virus options
                    virus_options = list(COMBINED_VIRUS_MAPPING.keys())
                    
                    selected_virus_key = st.selectbox(
                        "Actual Virus Diagnosis",
                        options=[None] + virus_options,
                        format_func=lambda x: "Please select the actual diagnosis..." if x is None else COMBINED_VIRUS_MAPPING[x],
                        help="Select the confirmed virus diagnosis from laboratory results or clinical assessment"
                    )
                
                with col_val2:
                    validation_confidence = st.selectbox(
                        "Confidence Level",
                        options=["High", "Medium", "Low"],
                        help="How confident are you in this diagnosis?"
                    )
                
                # Notes field
                validation_notes = st.text_area(
                    "Additional Notes (Optional)",
                    help="Any additional clinical observations or context",
                    placeholder="e.g., confirmed by RT-PCR, clinical presentation consistent with..."
                )
                
                # Submit validation button
                validation_submitted = st.form_submit_button(
                    "Submit Validation",
                    type="secondary",
                    use_container_width=True
                )
                
                if validation_submitted and selected_virus_key is not None:
                    # Save validation to database using session state data
                    validation_data = {
                        'prediction_id': saved_id,
                        'actual_virus_key': selected_virus_key,
                        'actual_virus_name': COMBINED_VIRUS_MAPPING[selected_virus_key],
                        'confidence_level': validation_confidence,
                        'notes': validation_notes,
                        'predicted_virus': VIRUS_MAPPING[pred_results['y_pred']],
                        'prediction_confidence': float(pred_results['y_pred_proba'][pred_results['y_pred']] * 100),
                        'patient_summary': {
                            'age': pred_results['patient_data']['age'],
                            'sex': 'Male' if pred_results['patient_data']['SEX'] == 1 else 'Female',
                            'state': pred_results['selected_state_name'],
                            'district': pred_results['selected_district_name']
                        }
                    }
                    
                    try:
                        validation_id = save_validation_to_db(validation_data)
                        if validation_id:
                            st.success("✅ Thank you! Validation submitted successfully.")
                            st.balloons()
                            st.caption(f"Validation ID: {validation_id[:8]}...")
                            # Clear the session state after successful validation
                            if 'prediction_results' in st.session_state:
                                del st.session_state['prediction_results']
                            if 'saved_id' in st.session_state:
                                del st.session_state['saved_id']
                        else:
                            st.error("❌ Failed to save validation. Please try again.")
                    except Exception as val_error:
                        st.error(f"❌ Validation error: {str(val_error)}")
                        # Add more detailed error info
                        st.error(f"Full error: {val_error}")
                
                elif validation_submitted and selected_virus_key is None:
                    st.warning("⚠️ Please select an actual diagnosis before submitting.")


if __name__ == "__main__":
    main()
