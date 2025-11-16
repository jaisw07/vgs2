import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any


class KnowledgeBaseService:
    """
    FastAPI-friendly version of KnowledgeBase.
    - No print() calls
    - Methods return JSON-safe responses
    - Supports structured inspection APIs
    """

    def __init__(self, csv_path: str, target_col: str = "prognosis", smoothing: float = 1.0):
        self.csv_path = csv_path
        self.target_col = target_col
        self.smoothing = smoothing

        # Internal structures
        self.df = None
        self.symptoms: List[str] = []
        self.diseases: List[str] = []
        self.P_symptom_given_disease: Dict[str, Dict[str, float]] = {}
        self.P_disease: Dict[str, float] = {}

    # ---------------------------------------------------
    # Load and preprocess dataset
    # ---------------------------------------------------
    def load_dataset(self) -> Dict[str, Any]:
        """
        Loads CSV and initializes the symptoms/diseases list.
        """
        self.df = pd.read_csv(self.csv_path)
        self.df.fillna(0, inplace=True)

        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")

        # Identify symptoms and diseases
        self.symptoms = [col for col in self.df.columns if col != self.target_col]
        self.diseases = sorted(self.df[self.target_col].unique())

        return {
            "status": "loaded",
            "n_samples": len(self.df),
            "n_symptoms": len(self.symptoms),
            "n_diseases": len(self.diseases),
            "symptoms": self.symptoms,
            "diseases": self.diseases,
        }

    # ---------------------------------------------------
    # Compute probability models
    # ---------------------------------------------------
    def compute_probabilities(self) -> Dict[str, Any]:
        """
        Computes P(Disease) and P(Symptom|Disease) using Laplace smoothing.
        """
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")

        # Compute P(Disease)
        disease_counts = self.df[self.target_col].value_counts()
        total = len(self.df)
        self.P_disease = {d: disease_counts[d] / total for d in self.diseases}

        # Compute P(Symptom|Disease)
        prob_matrix = defaultdict(dict)

        for disease in self.diseases:
            subset = self.df[self.df[self.target_col] == disease]

            for symptom in self.symptoms:
                count_yes = subset[symptom].sum()

                # Laplace smoothing: (count + 1) / (N + 2)
                prob_matrix[disease][symptom] = (
                    count_yes + self.smoothing
                ) / (len(subset) + 2 * self.smoothing)

        self.P_symptom_given_disease = dict(prob_matrix)

        return {
            "status": "probabilities_computed",
            "n_diseases": len(self.diseases),
            "n_symptoms": len(self.symptoms),
        }

    # ---------------------------------------------------
    # Accessors
    # ---------------------------------------------------
    def get_symptom_list(self) -> List[str]:
        return list(self.symptoms)

    def get_disease_list(self) -> List[str]:
        return list(self.diseases)

    def get_P_symptom_given_disease(self, disease: str, symptom: str) -> float:
        return float(self.P_symptom_given_disease.get(disease, {}).get(symptom, 0.5))

    def get_P_disease(self, disease: str) -> float:
        if disease not in self.P_disease:
            return 1.0 / max(len(self.diseases), 1)
        return float(self.P_disease[disease])

    # ---------------------------------------------------
    # Export for visualization
    # ---------------------------------------------------
    def export_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Return JSON-ready matrix of P(Symptom|Disease).
        """
        return {
            disease: {
                symptom: float(prob)
                for symptom, prob in symptom_probs.items()
            }
            for disease, symptom_probs in self.P_symptom_given_disease.items()
        }

    # ---------------------------------------------------
    # Debug / state
    # ---------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        return {
            "csv_path": self.csv_path,
            "symptoms": self.symptoms,
            "diseases": self.diseases,
            "P_disease": self.P_disease,
            "P_symptom_given_disease": self.P_symptom_given_disease,
        }