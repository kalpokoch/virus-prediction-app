import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import xgboost as xgb
from config import FEATURE_NAMES, VIRUS_MAPPING, SYMPTOM_GROUPS

# Page configuration
st.set_page_config(
    page_title="Virus Detection System",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def calculate_temporal_features(date_obj):
    """Calculate temporal features from date"""
    month = date_obj.month
    year = date_obj.year
    quarter = (month - 1) // 3 + 1
    week_of_year = date_obj.isocalendar()[1]
    day_of_year = date_obj.timetuple().tm_yday
    
    # Seasonal indicators
    is_monsoon = 1 if month in [6, 7, 8, 9] else 0
    is_winter = 1 if month in [12, 1, 2] else 0
    
    # Cyclical encoding
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    return {
        'month': month,
        'year': year,
        'quarter': quarter,
        'weekofyear': week_of_year,
        'dayofyear': day_of_year,
        'ismonsoon': is_monsoon,
        'iswinter': is_winter,
        'monthsin': month_sin,
        'monthcos': month_cos
    }

def create_feature_vector(patient_data):
    """
    Convert 48 user inputs → 80 model features (EXACT training replica)
    """
    import numpy as np
    import pandas as pd
    
    # Step 1: Create base DataFrame with 48 original features
    feature_df = pd.DataFrame([patient_data])
    
    # Fill missing values like training
    feature_df['age'] = feature_df['age'].fillna(30).clip(0, 120)  # Median age
    feature_df['duration_of_illness'] = feature_df['duration_of_illness'].fillna(0)
    
    # Fill all symptoms with 0 (binary)
    symptom_cols = [
        'HEADACHE', 'IRRITABLITY', 'ALTEREDSENSORIUM', 'SOMNOLENCE', 'NECKRIGIDITY', 
        'SEIZURES', 'DIARRHEA', 'DYSENTERY', 'NAUSEA', 'MALAISE', 'MYALGIA', 
        'ARTHRALGIA', 'CHILLS', 'RIGORS', 'BREATHLESSNESS', 'COUGH', 'RHINORRHEA', 
        'SORETHROAT', 'BULLAE', 'PAPULARRASH', 'PUSTULARRASH', 'MUSCULARRASH', 
        'MACULOPAPULARRASH', 'ESCHAR', 'DARKURINE', 'HEPATOMEGALY', 'REDEYE', 
        'DISCHARGEEYES', 'CRUSHINGEYES', 'JAUNDICE', 'FEVER', 'ABDOMINALPAIN', 'VOMITING'
    ]
    for col in symptom_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0
        feature_df[col] = feature_df[col].fillna(0).clip(0, 1)
    
    # Step 2: EXACT feature engineering from your notebook (48 → 80)
    
    # === AGE FEATURES ===
    feature_df['age_group'] = pd.cut(feature_df['age'], 
                                   bins=[0, 5, 18, 45, 65, 150], 
                                   labels=[0, 1, 2, 3, 4]).astype(int)
    feature_df['age_group'] = feature_df['age_group'].fillna(2)  # Adult default
    
    # === SYMPTOM GROUPS ===
    respiratory_cols = ['COUGH', 'BREATHLESSNESS', 'RHINORRHEA', 'SORETHROAT']
    gi_cols = ['DIARRHEA', 'DYSENTERY', 'NAUSEA', 'VOMITING', 'ABDOMINALPAIN']
    neuro_cols = ['HEADACHE', 'ALTEREDSENSORIUM', 'SEIZURES', 'SOMNOLENCE', 'NECKRIGIDITY', 'IRRITABLITY']
    skin_cols = ['PAPULARRASH', 'PUSTULARRASH', 'MACULOPAPULARRASH', 'BULLAE']
    systemic_cols = ['MYALGIA', 'ARTHRALGIA', 'CHILLS', 'RIGORS', 'MALAISE']
    
    symptom_cols = ['HEADACHE', 'FEVER', 'COUGH', 'VOMITING', 'DIARRHEA', 'MYALGIA', 
                   'ARTHRALGIA', 'NAUSEA', 'BREATHLESSNESS', 'SORETHROAT']
    
    # Symptom counts
    feature_df['symptom_count'] = feature_df[symptom_cols].sum(axis=1)
    feature_df['respiratory_symptoms'] = feature_df[respiratory_cols].sum(axis=1)
    feature_df['gi_symptoms'] = feature_df[gi_cols].sum(axis=1)
    feature_df['neuro_symptoms'] = feature_df[neuro_cols].sum(axis=1)
    feature_df['skin_symptoms'] = feature_df[skin_cols].sum(axis=1)
    feature_df['systemic_symptoms'] = feature_df[systemic_cols].sum(axis=1)
    feature_df['symptom_diversity'] = (feature_df[symptom_cols] > 0).sum(axis=1)
    
    # === GEO-TEMPORAL FEATURES (User provides month/season) ===
    # Add these to your sidebar inputs!
    month = patient_data.get('month', 1)  # Default January
    feature_df['month'] = month
    feature_df['is_monsoon'] = int(month in [6, 7, 8, 9])
    feature_df['is_winter'] = int(month in [12, 1, 2])
    
    # Season (0=Winter, 1=Summer, 2=Monsoon, 3=Post-monsoon)
    def get_season(m):
        if m in [12, 1, 2]: return 0
        elif m in [3, 4, 5]: return 1
        elif m in [6, 7, 8, 9]: return 2
        else: return 3
    feature_df['season'] = get_season(month)
    
    # Cyclical month encoding
    feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['month'] / 12)
    feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['month'] / 12)
    
    # === INTERACTION FEATURES (Key to 80 features) ===
    # Geo-temporal interactions
    feature_df['monsoon_respiratory'] = feature_df['is_monsoon'] * feature_df['respiratory_symptoms']
    feature_df['winter_respiratory'] = feature_df['is_winter'] * feature_df['respiratory_symptoms']
    feature_df['monsoon_fever'] = feature_df['is_monsoon'] * feature_df['FEVER']
    
    feature_df['state_season'] = patient_data['state'] * 10 + feature_df['season']
    feature_df['district_season'] = patient_data['district'] * 10 + feature_df['season']
    feature_df['district_month'] = patient_data['district'] * 100 + feature_df['month']
    
    feature_df['state_respiratory'] = patient_data['state'] * feature_df['respiratory_symptoms']
    feature_df['state_fever'] = patient_data['state'] * feature_df['FEVER']
    feature_df['state_gi'] = patient_data['state'] * feature_df['gi_symptoms']
    
    # Fever interactions (most important)
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
    
    # Final cleanup (EXACTLY like training)
    feature_df = feature_df.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Return first row as 80-feature vector
    return feature_df.iloc[0].values.reshape(1, -1)  # Shape: (1, 80)


def main():
    # Header
    st.title("🦠 Virus Detection and Classification System")
    st.markdown("---")
    st.write("Enter patient information and clinical symptoms to predict the most likely virus.")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check the model file path.")
        return
    
    # Sidebar for patient demographics
    st.sidebar.header("📋 Patient Information")
    
    patient_data = {}
    
   # Demographics
    patient_data['age'] = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
    patient_data['SEX'] = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    patient_data['PATIENTTYPE'] = st.sidebar.selectbox("Patient Type", options=[0, 1], format_func=lambda x: "Outpatient" if x == 0 else "Inpatient")
    patient_data['districtencoded'] = st.sidebar.number_input("Duration of Illness (days)", min_value=0, max_value=365, value=3)
    patient_data['labstate'] = st.sidebar.number_input("State Code", min_value=0, max_value=35, value=0, help="Encoded state value")
    patient_data['districtencoded'] = st.sidebar.number_input("District Code", min_value=0, max_value=740, value=370, help="Encoded district value (default: median)")
    patient_data['record_date'] = st.sidebar.date_input("Record Date", value=datetime.now())
    patient_data['month'] = st.sidebar.selectbox("Month of Illness Onset", options=list(range(1, 13)), 
                                                 format_func=lambda x: datetime(2000, x, 1).strftime('%B'))

    # Main area for symptoms
    st.header("🩺 Clinical Symptoms")
    st.write("Select all symptoms present in the patient:")
    
    # Create columns for organized symptom display
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
                
                # Make prediction
                y_pred = model.predict(X)[0]
                y_pred_proba = model.predict_proba(X)[0]
                
                # Get top 5 predictions
                top_5_indices = np.argsort(y_pred_proba)[-5:][::-1]
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🎯 Most Likely Virus")
                    st.metric(
                        label="Predicted Virus",
                        value=VIRUS_MAPPING[y_pred],
                        delta=f"{y_pred_proba[y_pred]*100:.2f}% confidence"
                    )
                
                with col2:
                    st.subheader("📊 Top 5 Predictions")
                    for rank, idx in enumerate(top_5_indices, 1):
                        virus_name = VIRUS_MAPPING[idx]
                        confidence = y_pred_proba[idx] * 100
                        st.write(f"{rank}. **{virus_name}**: {confidence:.2f}%")
                
                # Display probability distribution
                st.markdown("---")
                st.subheader("📈 Probability Distribution (Top 10)")
                
                top_10_indices = np.argsort(y_pred_proba)[-10:][::-1]
                prob_df = pd.DataFrame({
                    'Virus': [VIRUS_MAPPING[i] for i in top_10_indices],
                    'Probability (%)': [y_pred_proba[i]*100 for i in top_10_indices]
                })
                st.bar_chart(prob_df.set_index('Virus'))
                
                # Feature summary
                with st.expander("📋 Input Summary"):
                    st.write("**Patient Demographics:**")
                    st.write(f"- Age: {patient_data['age']} years")
                    st.write(f"- Sex: {'Male' if patient_data['sex'] == 1 else 'Female'}")
                    st.write(f"- Patient Type: {'Inpatient' if patient_data['patient_type'] == 1 else 'Outpatient'}")
                    st.write(f"- Duration: {patient_data['duration']} days")
                    
                    active_symptoms = [k.replace('_', ' ').title() for k, v in patient_data.items() 
                                     if k in sum(SYMPTOM_GROUPS.values(), []) and v == 1]
                    st.write(f"\n**Active Symptoms ({len(active_symptoms)}):**")
                    if active_symptoms:
                        st.write(", ".join(active_symptoms))
                    else:
                        st.write("None reported")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.error("Please ensure all required fields are filled correctly.")

if __name__ == "__main__":
    main()
