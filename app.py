import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import xgboost as xgb

# Page config
st.set_page_config(
    page_title="🦠 Virus Prediction System",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; font-weight: bold; text-align: center; margin-bottom: 2rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px;}
    .stSlider > div > div > div {background: linear-gradient(90deg, #ff6b6b, #4ecdc4);}
</style>
""", unsafe_allow_html=True)

# Load mappings
@st.cache_data
def load_mappings():
    """Load state/district mappings with fallbacks"""
    state_map = {9: "Uttar Pradesh"}
    district_map = {370: "Dadri (UP)"}
    
    try:
        if os.path.exists('models/state_mapping.pkl'):
            with open('models/state_mapping.pkl', 'rb') as f:
                state_map = pickle.load(f)
        if os.path.exists('models/district_mapping.pkl'):
            with open('models/district_mapping.pkl', 'rb') as f:
                district_map = pickle.load(f)
    except:
        pass
    return state_map, district_map

# Load model
@st.cache_resource
def load_model():
    """Load XGBoost model"""
    try:
        with open('models/xgb_filtered_model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully!")
        return model
    except:
        st.error("❌ Model file not found! Upload xgb_filtered_model.pkl to models/")
        return None

# EXACT feature engineering from training (48→80)
def create_feature_vector(patient_data):
    """
    Convert 48 user inputs → 80 model features (EXACT training replica)
    """
    # Step 1: Base DataFrame with all 48 original features
    feature_df = pd.DataFrame([patient_data])
    
    # Fill missing like training
    feature_df['age'] = feature_df['age'].fillna(30).clip(0, 120)
    feature_df['duration_of_illness'] = feature_df['duration_of_illness'].fillna(0)
    
    # All symptom columns (binary 0/1)
    symptom_cols = [
        'HEADACHE', 'IRRITABLITY', 'ALTEREDSENSORIUM', 'SOMNOLENCE', 'NECKRIGIDITY',
        'SEIZURES', 'DIARRHEA', 'DYSENTERY', 'NAUSEA', 'MALAISE', 'MYALGIA',
        'ARTHRALGIA', 'CHILLS', 'RIGORS', 'BREATHLESSNESS', 'COUGH', 'RHINORRHEA',
        'SORETHROAT', 'BULLAE', 'PAPULARRASH', 'PUSTULARRASH', 'MUSCULARRASH',
        'MACULOPAPULARRASH', 'ESCHAR', 'DARKURINE', 'HEPATOMEGALY', 'REDEYE',
        'DISCHARGEEYES', 'CRUSHINGEYES', 'JAUNDICE', 'FEVER', 'ABDOMINALPAIN', 'VOMITING'
    ]
    
    # Ensure all symptoms exist and are binary
    for col in symptom_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0
        feature_df[col] = feature_df[col].fillna(0).clip(0, 1)
    
    # === AGE GROUPS ===
    feature_df['age_group'] = pd.cut(feature_df['age'], 
                                   bins=[0, 5, 18, 45, 65, 150], 
                                   labels=[0, 1, 2, 3, 4]).astype(int)
    feature_df['age_group'] = feature_df['age_group'].fillna(2)
    
    # === SYMPTOM GROUPS ===
    respiratory_cols = ['COUGH', 'BREATHLESSNESS', 'RHINORRHEA', 'SORETHROAT']
    gi_cols = ['DIARRHEA', 'DYSENTERY', 'NAUSEA', 'VOMITING', 'ABDOMINALPAIN']
    neuro_cols = ['HEADACHE', 'ALTEREDSENSORIUM', 'SEIZURES', 'SOMNOLENCE', 'NECKRIGIDITY', 'IRRITABLITY']
    skin_cols = ['PAPULARRASH', 'PUSTULARRASH', 'MACULOPAPULARRASH', 'BULLAE']
    systemic_cols = ['MYALGIA', 'ARTHRALGIA', 'CHILLS', 'RIGORS', 'MALAISE']
    symptom_cols_main = ['HEADACHE', 'FEVER', 'COUGH', 'VOMITING', 'DIARRHEA', 'MYALGIA', 
                        'ARTHRALGIA', 'NAUSEA', 'BREATHLESSNESS', 'SORETHROAT']
    
    # Symptom counts by category
    feature_df['symptom_count'] = feature_df[symptom_cols_main].sum(axis=1)
    feature_df['respiratory_symptoms'] = feature_df[respiratory_cols].sum(axis=1)
    feature_df['gi_symptoms'] = feature_df[gi_cols].sum(axis=1)
    feature_df['neuro_symptoms'] = feature_df[neuro_cols].sum(axis=1)
    feature_df['skin_symptoms'] = feature_df[skin_cols].sum(axis=1)
    feature_df['systemic_symptoms'] = feature_df[systemic_cols].sum(axis=1)
    feature_df['symptom_diversity'] = (feature_df[symptom_cols_main] > 0).sum(axis=1)
    
    # === GEO-TEMPORAL FEATURES ===
    month = patient_data['month']
    feature_df['month'] = month
    feature_df['is_monsoon'] = int(month in [6, 7, 8, 9])
    feature_df['is_winter'] = int(month in [12, 1, 2])
    
    # Season calculation
    def get_season(m):
        if m in [12, 1, 2]: return 0  # Winter
        elif m in [3, 4, 5]: return 1  # Summer
        elif m in [6, 7, 8, 9]: return 2  # Monsoon
        else: return 3  # Post-monsoon
    feature_df['season'] = get_season(month)
    
    # Cyclical encoding
    feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['month'] / 12)
    feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['month'] / 12)
    
    # === INTERACTION FEATURES (32 new features) ===
    # Season interactions
    feature_df['monsoon_respiratory'] = feature_df['is_monsoon'] * feature_df['respiratory_symptoms']
    feature_df['winter_respiratory'] = feature_df['is_winter'] * feature_df['respiratory_symptoms']
    feature_df['monsoon_fever'] = feature_df['is_monsoon'] * feature_df['FEVER']
    
    # Geo interactions
    feature_df['state_season'] = patient_data['state'] * 10 + feature_df['season']
    feature_df['district_season'] = patient_data['district'] * 10 + feature_df['season']
    feature_df['district_month'] = patient_data['district'] * 100 + feature_df['month']
    
    feature_df['state_respiratory'] = patient_data['state'] * feature_df['respiratory_symptoms']
    feature_df['state_fever'] = patient_data['state'] * feature_df['FEVER']
    feature_df['state_gi'] = patient_data['state'] * feature_df['gi_symptoms']
    
    # Fever interactions (most predictive)
    feature_df['fever_respiratory'] = feature_df['FEVER'] * feature_df['respiratory_symptoms']
    feature_df['fever_gi'] = feature_df['FEVER'] * feature_df['gi_symptoms']
    feature_df['fever_neuro'] = feature_df['FEVER'] * feature_df['neuro_symptoms']
    feature_df['fever_skin'] = feature_df['FEVER'] * feature_df['skin_symptoms']
    feature_df['fever_duration'] = feature_df['FEVER'] * feature_df['duration_of_illness']
    feature_df['fever_headache'] = feature_df['FEVER'] * feature_df['HEADACHE']
    feature_df['fever_cough'] = feature_df['FEVER'] * feature_df['COUGH']
    
    # Severity & demographics
    feature_df['severity_score'] = feature_df['symptom_count'] * feature_df['duration_of_illness']
    feature_df['age_symptom'] = feature_df['age'] * feature_df['symptom_count']
    feature_df['age_duration'] = feature_df['age'] * feature_df['duration_of_illness']
    feature_df['patienttype_age'] = patient_data['PATIENTTYPE'] * feature_df['age_group']
    feature_df['sex_respiratory'] = patient_data['SEX'] * feature_df['respiratory_symptoms']
    feature_df['duration_symptom_ratio'] = feature_df['duration_of_illness'] / (feature_df['symptom_count'] + 1)
    
    # Final cleanup (EXACT training match)
    feature_df = feature_df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return feature_df.iloc[0].values.reshape(1, -1)  # (1, 80)

# Virus mapping (from your notebook)
VIRUS_MAPPING = {
    0: "Chikungunya Virus",
    1: "Dengue Virus", 
    2: "Enterovirus",
    3: "Hepatitis A Virus",
    4: "Hepatitis B Virus",
    5: "Hepatitis C Virus",
    6: "Hepatitis E Virus",
    7: "Herpes simplex virus",
    8: "Influenza A H1N1",
    9: "Influenza A H3N2",
    10: "Influenza B Victoria",
    11: "Japanese Encephalitis",
    12: "Leptospira",
    13: "Measles Virus",
    14: "Mumps Virus",
    15: "Other_Viruses",
    16: "Parvovirus",
    17: "Respiratory Adenovirus",
    18: "Respiratory Syncytial Virus (RSV)",
    19: "Respiratory Syncytial Virus-A (RSV-A)",
    20: "Respiratory Syncytial Virus-B (RSV-B)",
    21: "Rotavirus",
    22: "Rubella",
    23: "SARS-Cov-2",
    24: "Scrub typhus (Orientia tsutsugamushi)",
    25: "Varicella zoster virus (VZV)"
}

def main():
    st.markdown('<h1 class="main-header">🦠 Virus Prediction System</h1>', unsafe_allow_html=True)
    
    # Load resources
    model = load_model()
    state_mapping, district_mapping = load_mappings()
    
    if model is None:
        st.stop()
    
    # Sidebar - Patient Information
    st.sidebar.header("📋 Patient Information")
    
    # Location (codes internally)
    state_names = list(state_mapping.values())
    state_names.insert(0, "Select State")
    selected_state = st.sidebar.selectbox("🏛️ State", state_names, index=1)
    patient_data = {'state': list(state_mapping.keys())[state_names.index(selected_state)-1] if selected_state != "Select State" else 9}
    
    district_names = list(district_mapping.values())
    district_names.insert(0, "Select District")
    selected_district = st.sidebar.selectbox("🏘️ District", district_names, index=1)
    patient_data['district'] = list(district_mapping.keys())[district_names.index(selected_district)-1] if selected_district != "Select District" else 370
    
    # Demographics
    col1, col2 = st.sidebar.columns(2)
    with col1:
        patient_data['age'] = st.number_input("👶 Age", min_value=0, max_value=120, value=30)
        patient_data['SEX'] = st.selectbox("♀️ Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    with col2:
        patient_data['PATIENTTYPE'] = st.selectbox("🏥 Patient Type", [0, 1], format_func=lambda x: "OPD" if x == 0 else "IPD")
        patient_data['duration_of_illness'] = st.number_input("⏱️ Duration (days)", min_value=0, max_value=60, value=3)
    
    # Month (CRITICAL for geo-temporal features)
    patient_data['month'] = st.sidebar.slider("📅 Month of Illness", 1, 12, 1, 
                                             help="Monsoon (Jun-Sep) affects respiratory viruses")
    
    st.sidebar.markdown("---")
    
    # Symptoms (3 columns)
    st.sidebar.header("🤒 Symptoms (Check all that apply)")
    
    symptoms = [
        'HEADACHE', 'FEVER', 'COUGH', 'VOMITING', 'DIARRHEA', 'MYALGIA', 
        'ARTHRALGIA', 'NAUSEA', 'BREATHLESSNESS', 'SORETHROAT', 'IRRITABLITY',
        'ALTEREDSENSORIUM', 'SOMNOLENCE', 'NECKRIGIDITY', 'SEIZURES', 'CHILLS',
        'RIGORS', 'RHINORRHEA', 'BULLAE', 'PAPULARRASH', 'PUSTULARRASH',
        'MUSCULARRASH', 'MACULOPAPULARRASH', 'ESCHAR', 'DARKURINE', 'HEPATOMEGALY',
        'REDEYE', 'DISCHARGEEYES', 'CRUSHINGEYES', 'JAUNDICE', 'ABDOMINALPAIN'
    ]
    
    # Initialize symptom states
    for symptom in symptoms:
        if symptom not in patient_data:
            patient_data[symptom] = 0
    
    # Symptom checkboxes (3 columns)
    cols = st.sidebar.columns(3)
    for i, symptom in enumerate(symptoms):
        col_idx = i % 3
        with cols[col_idx]:
            patient_data[symptom] = st.checkbox(symptom.replace('RASH', 'Rash').replace('EYES', 'Eye'), 
                                              value=False, key=f"sym_{symptom}")
    
    # Predict button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Predict Virus", type="primary", use_container_width=True):
        
        # Generate 80-feature vector
        try:
            X = create_feature_vector(patient_data)
            
            # Verify shape
            if X.shape[1] != 80:
                st.error(f"❌ Feature shape mismatch! Expected 80, got {X.shape[1]}")
                st.stop()
            
            # Predict
            model.set_category(26)  # 26 filtered viruses
            probabilities = model.predict_proba(X)[0]
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction] * 100
            
            # Display results
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown("## 🎯 **Prediction Results**")
                st.error(f"**Predicted Virus:** {VIRUS_MAPPING[prediction]}")
                st.success(f"**Confidence:** {confidence:.1f}%")
            
            with col2:
                st.markdown("### 📊 Top 5 Probabilities")
                top5_idx = np.argsort(probabilities)[-5:][::-1]
                for i, idx in enumerate(top5_idx):
                    virus_name = VIRUS_MAPPING[idx]
                    prob = probabilities[idx] * 100
                    st.metric(f"{i+1}. {virus_name}", f"{prob:.1f}%")
            
            with col3:
                st.markdown("### ⚠️ **Risk Level**")
                if confidence > 85:
                    st.error("🔴 **HIGH**")
                elif confidence > 70:
                    st.warning("🟡 **MEDIUM**")
                else:
                    st.info("🟢 **LOW**")
            
            # Feature importance (top 10)
            st.markdown("### 📈 Feature Importance (Top 10)")
            importance_df = pd.DataFrame({
                'feature': [f'f{i}' for i in range(80)],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            st.bar_chart(importance_df.set_index('feature'))
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.error("Feature shape mismatch, expected: 80, got: check inputs")

if __name__ == "__main__":
    main()
