import streamlit as st
import pandas as pd
import json
from model import JSONVirusPredictionModel

# Page config
st.set_page_config(
    page_title="Virus Prediction System",
    page_icon="🦠",
    layout="wide"
)

# Initialize model
@st.cache_resource
def load_model():
    return JSONVirusPredictionModel('virus_probabilities.json')

model = load_model()

# Load knowledge base metadata
with open('virus_probabilities.json', 'r') as f:
    kb = json.load(f)

# Title and description
st.title("🦠 AI-Powered Virus Prediction System")
st.markdown("*Probabilistic diagnostic tool using Naive Bayes inference*")

# Sidebar - Model Info
with st.sidebar:
    st.header("📊 Model Statistics")
    st.metric("Total Viruses", kb['metadata']['total_viruses'])
    st.metric("Total Samples", f"{kb['metadata']['total_samples']:,}")
    st.metric("Positive Cases", f"{kb['metadata']['total_positive']:,}")
    st.metric("States Covered", kb['metadata']['total_states'])
    st.metric("Symptoms Tracked", kb['metadata']['total_symptoms'])
    
    st.markdown("---")
    st.info("**Data Source**: ICMR Virus Research Dataset")

# Main interface
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📁 Batch Prediction"])

# TAB 1: Single Prediction
with tab1:
    st.header("Patient Diagnosis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get available states
        all_states = sorted(set([s for v in kb['state_probabilities'].values() for s in v.keys()]))
        state = st.selectbox("Select State", all_states)
        
        season = st.selectbox("Select Season", 
                             ["Fall", "Winter", "Summer", "Spring"])
    
    with col2:
        # Get all available symptoms
        all_symptoms = sorted(set([s for v in kb['symptom_probabilities'].values() for s in v.keys()]))
        
        st.markdown("**Select Symptoms:**")
        symptoms = st.multiselect(
            "Search and select symptoms",
            all_symptoms,
            help="Start typing to search symptoms"
        )
    
    # Predict button
    if st.button("🔬 Predict Virus", type="primary", use_container_width=True):
        if not symptoms:
            st.warning("⚠️ Please select at least one symptom")
        else:
            with st.spinner("Analyzing patient data..."):
                predictions = model.predict_single(state, season, symptoms)
            
            st.success("✅ Analysis Complete!")
            
            # Display results
            st.subheader("Top 5 Predicted Viruses")
            
            for i, pred in enumerate(predictions, 1):
                virus = pred['virus']
                prob = pred['probability']
                
                # Color code by probability
                if prob > 0.3:
                    color = "green"
                elif prob > 0.1:
                    color = "orange"
                else:
                    color = "red"
                
                col1, col2, col3 = st.columns([1, 3, 2])
                with col1:
                    st.markdown(f"### #{i}")
                with col2:
                    st.markdown(f"**{virus}**")
                with col3:
                    st.progress(prob)
                    st.markdown(f":{color}[**{prob*100:.2f}%**]")
                
                st.divider()
            
            # Visualization
            df_results = pd.DataFrame(predictions)
            st.bar_chart(df_results.set_index('virus')['probability'])

# TAB 2: Batch Prediction
with tab2:
    st.header("Batch Processing")
    st.markdown("Upload a CSV file with columns: `State`, `Season`, and symptom columns (0/1 for absence/presence)")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Read uploaded file
        input_df = pd.read_csv(uploaded_file)
        
        st.subheader("Input Data Preview")
        st.dataframe(input_df.head(), use_container_width=True)
        
        if st.button("🚀 Process Batch", type="primary"):
            with st.spinner("Processing batch predictions..."):
                # Save temp file
                input_df.to_csv('temp_input.csv', index=False)
                
                # Run batch prediction
                output_df = model.predict_batch('temp_input.csv', 'temp_output.csv')
                
                st.success(f"✅ Processed {len(output_df)} records!")
                
                # Show results
                st.subheader("Results Preview")
                st.dataframe(output_df.head(10), use_container_width=True)
                
                # Download button
                csv = output_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Results",
                    data=csv,
                    file_name="virus_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit | Powered by Naive Bayes Inference</p>
</div>
""", unsafe_allow_html=True)
