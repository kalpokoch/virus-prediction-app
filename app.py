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
    """Create feature vector matching model training format"""
    feature_dict = {}
    
    # Demographics
    feature_dict['labstate'] = patient_data['state']
    feature_dict['age'] = patient_data['age']
    feature_dict['SEX'] = patient_data['sex']
    feature_dict['PATIENTTYPE'] = patient_data['patient_type']
    feature_dict['durationofillness'] = patient_data['duration']
    
    # Symptoms (33 binary features)
    for symptom in SYMPTOM_GROUPS.values():
        for s in symptom:
            feature_dict[s] = patient_data.get(s, 0)
    
    # Temporal features
    temporal_features = calculate_temporal_features(patient_data['record_date'])
    feature_dict.update(temporal_features)
    
    # District encoding (default to median if not available)
    feature_dict['districtencoded'] = patient_data.get('district', 370)  # Median value
    
    # Create ordered feature vector
    feature_vector = [feature_dict[feat] for feat in FEATURE_NAMES]
    
    return np.array(feature_vector).reshape(1, -1)

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
    patient_data['sex'] = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    patient_data['patient_type'] = st.sidebar.selectbox("Patient Type", options=[0, 1], format_func=lambda x: "Outpatient" if x == 0 else "Inpatient")
    patient_data['duration'] = st.sidebar.number_input("Duration of Illness (days)", min_value=0, max_value=365, value=3)
    patient_data['state'] = st.sidebar.number_input("State Code", min_value=0, max_value=35, value=0, help="Encoded state value")
    patient_data['district'] = st.sidebar.number_input("District Code", min_value=0, max_value=740, value=370, help="Encoded district value (default: median)")
    patient_data['record_date'] = st.sidebar.date_input("Record Date", value=datetime.now())
    
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
