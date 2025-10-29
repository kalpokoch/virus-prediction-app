import json
import pandas as pd

class JSONVirusPredictionModel:
    """
    JSON-based Logic Model for Virus Prediction
    Uses Naive Bayes approach with conditional probabilities
    """
    
    def __init__(self, knowledge_base_path):
        """Initialize model with JSON knowledge base"""
        with open(knowledge_base_path, 'r') as f:
            self.kb = json.load(f)
        
        self.viruses = list(self.kb['virus_priors'].keys())
        self.smoothing_factor = 1e-10
    
    def calculate_probability(self, virus, state, season, symptoms):
        """Calculate P(Virus | State, Season, Symptoms) using Bayes' theorem"""
        prior = self.kb['virus_priors'].get(virus, self.smoothing_factor)
        state_prob = self.kb['state_probabilities'].get(virus, {}).get(state, self.smoothing_factor)
        season_prob = self.kb['season_probabilities'].get(virus, {}).get(season, self.smoothing_factor)
        
        symptom_prob = 1.0
        if symptoms:
            for symptom in symptoms:
                symp_given_virus = self.kb['symptom_probabilities'].get(virus, {}).get(symptom, self.smoothing_factor)
                if symp_given_virus == 0:
                    symp_given_virus = self.smoothing_factor
                symptom_prob *= symp_given_virus
        
        probability = prior * state_prob * season_prob * symptom_prob
        return probability
    
    def predict_single(self, state, season, symptoms):
        """Predict top 5 viruses for given inputs"""
        results = {}
        
        for virus in self.viruses:
            prob = self.calculate_probability(virus, state, season, symptoms)
            results[virus] = prob
        
        total = sum(results.values())
        if total > 0:
            results = {k: v/total for k, v in results.items()}
        
        top5 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
        return [{"virus": virus, "probability": prob} for virus, prob in top5]
    
    def predict_batch(self, input_csv, output_csv):
        """Process batch predictions from CSV"""
        df = pd.read_csv(input_csv)
        
        for i in range(1, 6):
            df[f'Top{i}_Virus'] = ''
            df[f'Top{i}_Probability'] = 0.0
        
        symptom_cols = [col for col in df.columns if col not in ['State', 'Season'] 
                       and not col.startswith('Top')]
        
        for idx, row in df.iterrows():
            symptoms = [col for col in symptom_cols if row[col] == 1 or row[col] == True or row[col] == '1']
            predictions = self.predict_single(row['State'], row['Season'], symptoms)
            
            for i, pred in enumerate(predictions, 1):
                df.at[idx, f'Top{i}_Virus'] = pred['virus']
                df.at[idx, f'Top{i}_Probability'] = round(pred['probability'], 6)
        
        df.to_csv(output_csv, index=False)
        return df
