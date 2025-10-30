"""
Streamlit Web Application for Virus Prediction System
Uses unified 80/20 split knowledge base
"""

import streamlit as st
import pandas as pd
import json
import os
from model import JSONVirusPredictionModel


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Virus Prediction System",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# INITIALIZE MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load knowledge base from unified split"""
    kb_path = 'knowledge_base_unified.json'
    
    # Check if file exists
    if not os.path.exists(kb_path):
        st.error(f"❌ Knowledge base file not found: {kb_path}")
        st.info("Please ensure `knowledge_base_unified.json` is in the same directory as the app")
        st.stop()
    
    try:
        return JSONVirusPredictionModel(kb_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


try:
    model = load_model()
    kb = model.kb
except Exception as e:
    st.error(f"❌ Failed to initialize: {str(e)}")
    st.stop()


# ============================================================
# TITLE AND DESCRIPTION
# ============================================================

st.title("🦠 AI-Powered Virus Prediction System")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("""
    **Probabilistic diagnostic tool using Naive Bayes inference**
    - Trained on ICMR Virus Research Dataset
    - Unified 80/20 split methodology
    - 83 viruses, 35 symptoms, 32 states
    """)
with col2:
    st.markdown(f"""
    **✅ Model Status**
    - Version: {kb['metadata']['version']}
    - Viruses: {kb['metadata']['total_viruses']}
    - Ready for predictions
    """)
with col3:
    st.markdown(f"""
    **📊 Data Overview**
    - Training: 80%
    - Testing: 20%
    - Split: Unified
    """)


# ============================================================
# SIDEBAR - MODEL INFO
# ============================================================

with st.sidebar:
    st.header("📊 Model Statistics")
    
    # Model metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Viruses", kb['metadata']['total_viruses'])
        st.metric("States", kb['metadata']['total_states'])
    with col2:
        st.metric("Symptoms", kb['metadata']['total_symptoms'])
        st.metric("Seasons", len(kb['metadata']['seasons']))
    
    st.metric("Training Samples", f"{kb['metadata']['training_rows']['priors']:,}")
    st.metric("Test Samples", f"{kb['metadata']['test_rows']:,}")
    
    st.markdown("---")
    
    # Data split info
    with st.expander("ℹ️ Data Split Details"):
        st.info(f"""
        **Split Method**: {kb['metadata'].get('data_split', 'Unified 80/20')}
        
        **Training Data**:
        - Priors: {kb['metadata']['training_rows']['priors']} viruses
        - State patterns: {kb['metadata']['training_rows']['state']} pairs
        - Season patterns: {kb['metadata']['training_rows']['season']} pairs
        - Symptom patterns: {kb['metadata']['training_rows']['symptom']} pairs
        
        **Test Data**: {kb['metadata']['test_rows']} samples
        
        **Notes**: {kb['metadata'].get('notes', 'N/A')}
        """)
    
    st.markdown("---")
    
    with st.expander("🧠 Model Algorithm"):
        st.write("""
        **Naive Bayes Inference**
        
        Formula:
        ```
        P(V|S,Sea,Sy) ∝ P(V) × P(S|V) × P(Sea|V) × ∏P(Sy_i|V)
        ```
        
        Where:
        - V = Virus
        - S = State
        - Sea = Season
        - Sy = Symptoms
        """)
    
    st.markdown("---")
    st.caption("Built with Streamlit | Knowledge Base v2.0")


# ============================================================
# MAIN INTERFACE - TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Prediction", "📁 Batch Processing", "📊 Model Info", "ℹ️ About"])


# ============================================================
# TAB 1: SINGLE PREDICTION
# ============================================================

with tab1:
    st.header("🏥 Patient Diagnosis")
    st.markdown("Enter patient information to predict possible viral infections")
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Location & Time")
        
        # Get available states
        available_states = model.get_available_states()
        state = st.selectbox(
            "Select State",
            available_states,
            help="Select the patient's location in India"
        )
        
        season = st.selectbox(
            "Select Season",
            model.get_available_seasons(),
            help="Select the current season"
        )
    
    with col2:
        st.subheader("🩺 Symptoms")
        
        # Get all available symptoms
        available_symptoms = model.get_available_symptoms()
        
        symptoms = st.multiselect(
            "Select Patient Symptoms",
            available_symptoms,
            help="Search and select observed symptoms"
        )
    
    # Prediction section
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔬 Generate Prediction", type="primary", use_container_width=True):
            if not symptoms:
                st.warning("⚠️ Please select at least one symptom for prediction")
            else:
                with st.spinner("Analyzing patient data..."):
                    predictions = model.predict_single(state, season, symptoms)
                
                st.success("✅ Analysis Complete!")
                
                # Display results
                st.subheader("Top 5 Predicted Viruses")
                
                results_data = []
                
                for i, pred in enumerate(predictions, 1):
                    virus = pred['virus']
                    prob = pred['probability']
                    
                    results_data.append({
                        'Rank': i,
                        'Virus': virus,
                        'Probability': prob,
                        'Percentage': f"{prob*100:.2f}%"
                    })
                    
                    # Color code
                    if prob > 0.30:
                        status = "🟢 High"
                        color = "green"
                    elif prob > 0.10:
                        status = "🟡 Medium"
                        color = "orange"
                    else:
                        status = "🔴 Low"
                        color = "red"
                    
                    # Display
                    col_rank, col_virus, col_status, col_prob = st.columns([0.5, 2.5, 1.5, 1.5])
                    
                    with col_rank:
                        st.markdown(f"### #{i}")
                    with col_virus:
                        st.markdown(f"**{virus}**")
                    with col_status:
                        st.markdown(f"{status}")
                    with col_prob:
                        st.markdown(f"`{prob*100:.2f}%`")
                    
                    st.progress(prob)
                
                # Visualization
                st.subheader("📈 Probability Distribution")
                df_results = pd.DataFrame(results_data)
                st.bar_chart(df_results.set_index('Virus')['Probability'])
                
                # Summary
                st.subheader("📋 Prediction Summary")
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                
                with summary_col1:
                    st.metric("State", state)
                with summary_col2:
                    st.metric("Season", season)
                with summary_col3:
                    st.metric("Symptoms", len(symptoms))
                with summary_col4:
                    st.metric("Top Match", predictions[0]['virus'])
    
    with col2:
        st.info("""
        **How it works:**
        1. Select patient's location
        2. Choose current season
        3. Check observed symptoms
        4. Click "Generate Prediction"
        
        The system uses Naive Bayes to calculate probabilities.
        """)
    
    with col3:
        st.warning("""
        **Important Notes:**
        - Results are probabilistic
        - Not a substitute for medical diagnosis
        - Consult healthcare professionals
        - Multiple viruses may be possible
        """)


# ============================================================
# TAB 2: BATCH PROCESSING
# ============================================================

with tab2:
    st.header("📁 Batch Processing")
    st.markdown("""
    Process multiple patients at once. Upload a CSV file with:
    - **State**: Patient's state (must match available states)
    - **Season**: Season (Fall, Winter, Summer, Spring)
    - **Symptoms**: Binary columns (0=absent, 1=present)
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV with State, Season, and symptom columns"
    )
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 Input Data Preview")
            st.dataframe(input_df.head(10), use_container_width=True)
            st.info(f"Total records: {len(input_df)}")
            
            # Validate columns
            required_cols = ['State', 'Season']
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                    with st.spinner(f"Processing {len(input_df)} predictions..."):
                        output_df = model.predict_batch(input_df)
                    
                    st.success(f"✅ Successfully processed {len(output_df)} records!")
                    
                    # Results
                    st.subheader("🔍 Results Preview")
                    
                    display_cols = [col for col in output_df.columns 
                                   if col in ['State', 'Season', 'Top1_Virus', 'Top1_Probability', 
                                             'Top2_Virus', 'Top2_Probability', 'Top3_Virus']]
                    
                    st.dataframe(output_df[display_cols].head(15), use_container_width=True)
                    
                    # Statistics
                    st.subheader("📊 Batch Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Processed", len(output_df))
                    with col2:
                        top_virus = output_df['Top1_Virus'].value_counts().index[0]
                        st.metric("Most Common Prediction", top_virus)
                    with col3:
                        avg_prob = output_df['Top1_Probability'].mean() * 100
                        st.metric("Avg Top1 Confidence", f"{avg_prob:.2f}%")
                    with col4:
                        max_prob = output_df['Top1_Probability'].max() * 100
                        st.metric("Max Confidence", f"{max_prob:.2f}%")
                    
                    # Download
                    csv = output_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")


# ============================================================
# TAB 3: MODEL INFORMATION
# ============================================================

with tab3:
    st.header("📊 Model Information")
    
    # Overview
    st.subheader("🎯 Model Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Algorithm", "Naive Bayes")
    with col2:
        st.metric("Version", kb['metadata']['version'])
    with col3:
        st.metric("Created", kb['metadata']['created_date'])
    
    # Data summary
    st.subheader("📚 Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", f"{kb['metadata']['training_rows']['priors']:,}")
    with col2:
        st.metric("Test Samples", f"{kb['metadata']['test_rows']:,}")
    with col3:
        st.metric("Total Samples", f"{kb['metadata']['training_rows']['priors'] + kb['metadata']['test_rows']:,}")
    with col4:
        st.metric("Split Ratio", "80/20")
    
    # Training details
    st.subheader("📝 Training Data Details")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Viruses", kb['metadata']['training_rows']['priors'])
    with col2:
        st.metric("State-Virus Pairs", kb['metadata']['training_rows']['state'])
    with col3:
        st.metric("Season Patterns", kb['metadata']['training_rows']['season'])
    with col4:
        st.metric("Symptom Patterns", kb['metadata']['training_rows']['symptom'])
    
    # Limitations
    st.subheader("⚠️ Model Limitations")
    st.warning("""
    1. **Test Set**: 14 rare viruses excluded (had <2 samples)
    2. **Class Imbalance**: Dengue represents ~45% of training data
    3. **Assumptions**: Assumes symptom independence given virus
    4. **Geographic Limit**: Works only for 32 Indian states
    5. **Minimum Input**: Requires at least 1 symptom for prediction
    6. **Not Medical Advice**: For reference only, not diagnostic
    """)
    
    # Metadata
    st.subheader("📋 Complete Metadata")
    st.json(kb['metadata'])


# ============================================================
# TAB 4: ABOUT
# ============================================================

with tab4:
    st.header("ℹ️ About This System")
    
    st.subheader("🎯 Purpose")
    st.write("""
    This system provides probabilistic predictions for viral infections based on:
    - Patient's geographic location (state)
    - Current season
    - Observed symptoms
    
    It uses Naive Bayes inference trained on ICMR virus research data.
    """)
    
    st.subheader("🔬 Technology")
    st.write("""
    - **Algorithm**: Naive Bayes with conditional independence
    - **Training**: 80% of unified dataset (unified 80/20 split)
    - **Testing**: Evaluated on 20% holdout test set
    - **Framework**: Streamlit web application
    - **Language**: Python
    """)
    
    st.subheader("📊 Dataset")
    st.write(f"""
    - **Source**: ICMR Virus Research Dataset
    - **Total Samples**: {kb['metadata']['training_rows']['priors'] + kb['metadata']['test_rows']:,}
    - **Viruses**: {kb['metadata']['total_viruses']}
    - **Symptoms**: {kb['metadata']['total_symptoms']}
    - **Geographic Coverage**: {kb['metadata']['total_states']} Indian states
    - **Temporal Coverage**: Multiple seasons (Fall, Winter, Summer, Spring)
    """)
    
    st.subheader("⚕️ Medical Disclaimer")
    st.error("""
    **IMPORTANT**: This tool is for educational and reference purposes only.
    
    - NOT a substitute for professional medical diagnosis
    - Results are probabilistic estimates, not certainties
    - Always consult qualified healthcare professionals
    - In case of emergency, seek immediate medical attention
    - Do not rely solely on this tool for medical decisions
    """)
    
    st.subheader("📖 How to Use")
    st.write("""
    1. **Single Prediction**: Use the "Single Prediction" tab for one patient
    2. **Batch Processing**: Use the "Batch Processing" tab for multiple patients
    3. **Model Info**: Check model details and statistics
    4. **Results**: View and download predictions in CSV format
    """)
    
    st.subheader("📞 Contact & Support")
    st.info("""
    For issues or questions:
    - Check the Model Info tab for technical details
    - Ensure your input data matches required format
    - Verify knowledge_base_unified.json is present
    - Check Streamlit console for error messages
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>
    <p>🦠 <b>Virus Prediction System</b> | Knowledge Base v2.0 (Unified Split)</p>
    <p>Built with Streamlit | Powered by Naive Bayes Inference</p>
    <p>Data Source: ICMR Virus Research Dataset</p>
</div>
""", unsafe_allow_html=True)
