# 🦠 Virus Detection System

Advanced AI-powered virus classification from patient symptoms using dual XGBoost models with MongoDB Atlas integration.

## Features

- **26 Virus Categories**: Comprehensive classification including COVID-19, Influenza, Dengue, and more
- **Dual-Model Architecture**: Primary classification + secondary sub-classification for "Other Viruses"  
- **Real-time Predictions**: Instant probability scores and confidence metrics
- **Symptom Analysis**: Covers neurological, gastrointestinal, respiratory, and dermatological symptoms
- **Database Integration**: MongoDB Atlas for prediction storage and analytics
- **Interactive UI**: User-friendly Streamlit interface with organized symptom groups

## Quick Start

### Streamlit Cloud Deployment
1. Fork this repository
2. Connect to Streamlit Cloud
3. Add your MongoDB Atlas connection string to app secrets:
   ```
   mongodb.connection_string = "mongodb+srv://username:password@cluster.mongodb.net/virus_prediction?retryWrites=true&w=majority"
   ```

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Configuration

**MongoDB Atlas Setup** (Required for data persistence):
1. Create free cluster at https://mongodb.com/atlas
2. Add connection string to `.streamlit/secrets.toml`:
   ```toml
   [mongodb]
   connection_string = "your_connection_string_here"
   ```

## Model Information

- **Algorithm**: XGBoost with optimized hyperparameters
- **Model 1**: Primary classification (26 virus categories)  
- **Model 2**: Secondary classification (13 "Other Viruses" sub-categories)
- **Features**: 80+ engineered features including demographics, symptoms, and geo-temporal data
- **Dataset**: ICMR Virus Research data (663K+ cases, 35 symptoms, 35+ states)

## Medical Disclaimer

⚠️ **Important**: This system assists healthcare professionals and should not replace professional medical diagnosis. Always consult qualified medical personnel for patient care decisions.