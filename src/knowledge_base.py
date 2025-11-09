import pandas as pd
import numpy as np
from collections import defaultdict

class KnowledgeBase:
    def __init__(self, csv_path: str, target_col: str = "prognosis", smoothing: float = 1.0):
        """
        Builds a probabilistic knowledge base from a symptom–disease dataset.
        """
        self.csv_path = csv_path
        self.target_col = target_col
        self.smoothing = smoothing
        
        # Internal structures
        self.df = None
        self.symptoms = []
        self.diseases = []
        self.P_symptom_given_disease = {}
        self.P_disease = {}
    
    # -----------------------------
    # Load dataset and preprocess
    # -----------------------------
    def load_dataset(self):
        self.df = pd.read_csv(self.csv_path)
        self.df.fillna(0, inplace=True)
        
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")
        
        # Identify symptoms and diseases
        self.symptoms = [col for col in self.df.columns if col != self.target_col]
        self.diseases = sorted(self.df[self.target_col].unique())
        
        print(f"✅ Loaded dataset with {len(self.df)} samples, {len(self.symptoms)} symptoms, {len(self.diseases)} diseases.")

    # -----------------------------
    # Compute probabilities
    # -----------------------------
    def compute_probabilities(self):
        """
        Computes:
        P(Disease)
        P(Symptom | Disease)
        Using Laplace smoothing.
        """
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")
        
        # P(Disease)
        disease_counts = self.df[self.target_col].value_counts()
        total = len(self.df)
        self.P_disease = {d: disease_counts[d] / total for d in self.diseases}
        
        # P(Symptom|Disease)
        prob_matrix = defaultdict(dict)
        
        for disease in self.diseases:
            subset = self.df[self.df[self.target_col] == disease]
            for symptom in self.symptoms:
                count_yes = subset[symptom].sum()
                prob_matrix[disease][symptom] = (count_yes + self.smoothing) / (len(subset) + 2 * self.smoothing)
        
        self.P_symptom_given_disease = dict(prob_matrix)
        print("✅ Computed conditional probabilities P(Symptom|Disease).")

    # -----------------------------
    # Utility: Accessors
    # -----------------------------
    def get_symptom_list(self):
        return self.symptoms
    
    def get_disease_list(self):
        return self.diseases
    
    def get_P_symptom_given_disease(self, disease, symptom):
        return self.P_symptom_given_disease.get(disease, {}).get(symptom, 0.5)
    
    def get_P_disease(self, disease):
        return self.P_disease.get(disease, 1.0 / len(self.diseases))

    # -----------------------------
    # Optional: Export for visualization
    # -----------------------------
    def export_matrix(self):
        """
        Returns a DataFrame representation of P(Symptom|Disease).
        """
        df_export = pd.DataFrame(self.P_symptom_given_disease).T
        return df_export
