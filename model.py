"""
Virus Prediction Model
Uses Naive Bayes inference with unified 80/20 split training data
"""

import json
import pandas as pd


class JSONVirusPredictionModel:
    """
    JSON-based Logic Model for Virus Prediction
    Uses Naive Bayes approach with conditional probabilities
    Trained on unified 80/20 split from preprocessing pipeline
    """
    
    def __init__(self, knowledge_base_path):
        """
        Initialize model with JSON knowledge base
        
        Args:
            knowledge_base_path (str): Path to knowledge_base_unified.json
        """
        try:
            with open(knowledge_base_path, 'r') as f:
                self.kb = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Knowledge base not found at: {knowledge_base_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in: {knowledge_base_path}")
        
        self.viruses = list(self.kb['virus_priors'].keys())
        self.smoothing_factor = 1e-10
        
        # Store metadata
        self.metadata = self.kb.get('metadata', {})
    
    def calculate_probability(self, virus, state, season, symptoms):
        """
        Calculate P(Virus | State, Season, Symptoms) using Bayes' theorem
        
        Formula:
        P(V|S,Sea,Sy) ∝ P(V) × P(S|V) × P(Sea|V) × ∏P(Sy_i|V)
        
        Args:
            virus (str): Virus name
            state (str): Patient's state
            season (str): Current season
            symptoms (list): List of symptom names
        
        Returns:
            float: Unnormalized probability
        """
        # Prior probability: P(Virus)
        prior = self.kb['virus_priors'].get(virus, self.smoothing_factor)
        
        # Likelihood: P(State | Virus)
        state_prob = self.kb['state_probabilities'].get(virus, {}).get(state, self.smoothing_factor)
        
        # Likelihood: P(Season | Virus)
        season_prob = self.kb['season_probabilities'].get(virus, {}).get(season, self.smoothing_factor)
        
        # Likelihood: P(Symptoms | Virus)
        symptom_prob = 1.0
        if symptoms:
            for symptom in symptoms:
                symp_given_virus = self.kb['symptom_probabilities'].get(virus, {}).get(symptom, self.smoothing_factor)
                if symp_given_virus == 0:
                    symp_given_virus = self.smoothing_factor
                symptom_prob *= symp_given_virus
        
        # Compute combined probability
        probability = prior * state_prob * season_prob * symptom_prob
        return probability
    
    def predict_single(self, state, season, symptoms):
        """
        Predict top 5 viruses for given inputs
        
        Args:
            state (str): Patient's state
            season (str): Current season
            symptoms (list): List of symptom names
        
        Returns:
            list: List of dicts with virus names and normalized probabilities
                  Format: [{"virus": "Dengue", "probability": 0.45}, ...]
        """
        results = {}
        
        # Calculate probability for each virus
        for virus in self.viruses:
            prob = self.calculate_probability(virus, state, season, symptoms)
            results[virus] = prob
        
        # Normalize probabilities (convert to posterior)
        total = sum(results.values())
        if total > 0:
            results = {k: v/total for k, v in results.items()}
        else:
            # If all probabilities are 0, use uniform distribution
            uniform_prob = 1.0 / len(self.viruses)
            results = {virus: uniform_prob for virus in self.viruses}
        
        # Get top 5
        top5 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return [{"virus": virus, "probability": prob} for virus, prob in top5]
    
    def predict_batch(self, input_df):
        """
        Process batch predictions from DataFrame
        
        Args:
            input_df (pd.DataFrame): DataFrame with columns:
                - State (str): Patient's state
                - Season (str): Current season
                - Symptom columns (0/1): Binary indicator for each symptom
        
        Returns:
            pd.DataFrame: Original dataframe + Top1-5 predictions with probabilities
        """
        df = input_df.copy()
        
        # Add prediction columns
        for i in range(1, 6):
            df[f'Top{i}_Virus'] = ''
            df[f'Top{i}_Probability'] = 0.0
        
        # Identify symptom columns (everything except State, Season, True_Virus, and Top columns)
        symptom_cols = [col for col in df.columns 
                       if col not in ['State', 'Season', 'True_Virus'] 
                       and not col.startswith('Top')]
        
        # Process each row
        for idx, row in df.iterrows():
            # Extract symptoms for this patient
            symptoms = [col for col in symptom_cols 
                       if row[col] == 1 or row[col] == True or str(row[col]).upper() == '1']
            
            # Get predictions
            predictions = self.predict_single(row['State'], row['Season'], symptoms)
            
            # Store predictions
            for i, pred in enumerate(predictions, 1):
                df.at[idx, f'Top{i}_Virus'] = pred['virus']
                df.at[idx, f'Top{i}_Probability'] = round(pred['probability'], 6)
        
        return df
    
    def get_model_info(self):
        """
        Get model metadata and statistics
        
        Returns:
            dict: Model information
        """
        return {
            "version": self.metadata.get('version', 'Unknown'),
            "created_date": self.metadata.get('created_date', 'Unknown'),
            "description": self.metadata.get('description', 'Unknown'),
            "total_viruses": self.metadata.get('total_viruses', len(self.viruses)),
            "total_states": self.metadata.get('total_states', 'Unknown'),
            "total_symptoms": self.metadata.get('total_symptoms', 'Unknown'),
            "data_split": self.metadata.get('data_split', 'Unknown'),
            "notes": self.metadata.get('notes', 'Unknown')
        }
    
    def get_available_states(self):
        """Get list of available states"""
        return sorted(set([s for v in self.kb['state_probabilities'].values() for s in v.keys()]))
    
    def get_available_symptoms(self):
        """Get list of available symptoms"""
        return sorted(set([s for v in self.kb['symptom_probabilities'].values() for s in v.keys()]))
    
    def get_available_seasons(self):
        """Get list of available seasons"""
        return self.metadata.get('seasons', ['Fall', 'Winter', 'Summer', 'Spring'])
