# Feature names (must match training data order)
FEATURE_NAMES = [
    # Demographics & Clinical (5 features)
    'labstate', 'age', 'SEX', 'PATIENTTYPE', 'durationofillness',
    
    # Symptoms (33 features)
    'HEADACHE', 'IRRITABLITY', 'ALTEREDSENSORIUM', 'SOMNOLENCE', 
    'NECKRIGIDITY', 'SEIZURES', 'DIARRHEA', 'DYSENTERY', 'NAUSEA', 
    'MALAISE', 'MYALGIA', 'ARTHRALGIA', 'CHILLS', 'RIGORS', 
    'BREATHLESSNESS', 'COUGH', 'RHINORRHEA', 'SORETHROAT', 'BULLAE', 
    'PAPULARRASH', 'PUSTULARRASH', 'MUSCULARRASH', 'MACULOPAPULARRASH', 
    'ESCHAR', 'DARKURINE', 'HEPATOMEGALY', 'REDEYE', 'DISCHARGEEYES', 
    'CRUSHINGEYES', 'JAUNDICE', 'FEVER', 'ABDOMINALPAIN', 'VOMITING',
    
    # Geo-temporal features (10 features)
    'month', 'year', 'quarter', 'weekofyear', 'dayofyear', 
    'ismonsoon', 'iswinter', 'monthsin', 'monthcos', 'districtencoded'
]

# 26 Viruses in filtered model
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

# MongoDB Configuration
MONGODB_CONFIG = {
    'database_name': 'virus_prediction',
    'collections': {
        'predictions': 'predictions',
        'patients': 'patients', 
        'usage_stats': 'usage_stats'
    },
    'connection_timeout': 10000,
    'server_selection_timeout': 5000
}

# Symptom groups for organized display
SYMPTOM_GROUPS = {
    'Neurological': ['HEADACHE', 'IRRITABLITY', 'ALTEREDSENSORIUM', 'SOMNOLENCE', 
                     'NECKRIGIDITY', 'SEIZURES'],
    'Gastrointestinal': ['DIARRHEA', 'DYSENTERY', 'NAUSEA', 'VOMITING', 'ABDOMINALPAIN'],
    'Respiratory': ['BREATHLESSNESS', 'COUGH', 'RHINORRHEA', 'SORETHROAT'],
    'Systemic': ['FEVER', 'MALAISE', 'MYALGIA', 'ARTHRALGIA', 'CHILLS', 'RIGORS'],
    'Skin': ['BULLAE', 'PAPULARRASH', 'PUSTULARRASH', 'MUSCULARRASH', 
             'MACULOPAPULARRASH', 'ESCHAR'],
    'Other': ['DARKURINE', 'HEPATOMEGALY', 'REDEYE', 'DISCHARGEEYES', 
              'CRUSHINGEYES', 'JAUNDICE']
}
