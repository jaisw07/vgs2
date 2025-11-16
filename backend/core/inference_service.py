import numpy as np
from typing import Dict, Any, List, Tuple


class InferenceService:
    """
    FastAPI-friendly wrapper around your original InferenceEngine.
    - Removes any CLI coupling
    - Returns JSON-safe structures
    - Integrates with EntropyService and CSPService
    """

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.diseases = self.kb.get_disease_list()
        self.symptoms = self.kb.get_symptom_list()

        # Initialize priors
        self.priors = np.array([self.kb.get_P_disease(d) for d in self.diseases])
        self.posteriors = self.priors.copy()

    # ---------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        """
        Reset posterior to priors.
        """
        self.posteriors = self.priors.copy()
        return {"status": "reset", "posteriors": self._posterior_dict()}

    # ---------------------------------------------------
    def update_beliefs(self, symptom_name: str, user_response: int) -> Dict[str, Any]:
        """
        Update posterior distribution using Bayes' rule.
        Used by EntropyService and front-end answers.

        user_response:
            1  → Yes
            0  → No
           -1  → Unknown (ignored)
        """

        if user_response == -1:
            # No update for unknown response
            return {
                "status": "ignored",
                "message": f"Unknown response for symptom '{symptom_name}'",
                "posteriors": self._posterior_dict()
            }

        likelihoods = []
        for disease in self.diseases:
            p_symptom = self.kb.get_P_symptom_given_disease(disease, symptom_name)
            likelihood = p_symptom if user_response == 1 else (1 - p_symptom)
            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods)

        numerators = self.posteriors * likelihoods
        if numerators.sum() == 0:
            numerators += 1e-9  # numeric safety

        self.posteriors = numerators / numerators.sum()

        return {
            "status": "updated",
            "symptom": symptom_name,
            "response": user_response,
            "posteriors": self._posterior_dict()
        }

    # ---------------------------------------------------
    def get_top_diseases(self, top_k=5) -> Dict[str, Any]:
        """
        Returns the top-K diseases with probabilities.
        JSON-friendly {disease: prob}
        """
        sorted_idx = np.argsort(self.posteriors)[::-1]
        result = [
            {
                "disease": self.diseases[i],
                "probability": float(self.posteriors[i])
            }
            for i in sorted_idx[:top_k]
        ]

        return {"top_diseases": result}

    # ---------------------------------------------------
    def get_entropy(self) -> float:
        """
        Shannon entropy of posterior distribution.
        """
        p = self.posteriors[self.posteriors > 0]
        return float(-np.sum(p * np.log2(p)))

    # ---------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """
        Debug endpoint — exposes current inference state.
        """
        return {
            "symptoms": self.symptoms,
            "diseases": self.diseases,
            "posteriors": self._posterior_dict(),
            "entropy": self.get_entropy()
        }

    # ---------------------------------------------------
    def _posterior_dict(self) -> Dict[str, float]:
        """
        Convert numpy posterior array to JSON-friendly dict.
        """
        return {
            disease: float(prob)
            for disease, prob in zip(self.diseases, self.posteriors)
        }