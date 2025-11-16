import numpy as np
from typing import Dict, Any, List, Tuple

class EntropyService:
    """
    FastAPI-friendly wrapper for EntropyEngine.
    - Returns dicts instead of tuples or raw values
    - Removes print() and CLI usage
    - Works with session-based inference engines
    """

    def __init__(self, inference_engine):
        self.engine = inference_engine
        self.kb = inference_engine.kb
        self.diseases = inference_engine.diseases
        self.symptoms = inference_engine.symptoms
        self.asked_symptoms = set()

    # ----------------------------------------------
    def get_unasked_symptoms(self) -> List[str]:
        """
        Return list of symptoms that have not been asked yet.
        """
        return [s for s in self.symptoms if s not in self.asked_symptoms]

    # ----------------------------------------------
    def compute_expected_entropy(self, symptom_name: str) -> float:
        """
        Compute expected posterior entropy for symptom_name.
        """
        priors = self.engine.posteriors.copy()

        # Compute P(S=1)
        p_s1 = np.sum([
            priors[i] * self.kb.get_P_symptom_given_disease(self.diseases[i], symptom_name)
            for i in range(len(self.diseases))
        ])
        p_s0 = 1 - p_s1

        # Posterior if YES
        post_yes = np.array([
            priors[i] * self.kb.get_P_symptom_given_disease(self.diseases[i], symptom_name)
            for i in range(len(self.diseases))
        ])
        post_yes /= (post_yes.sum() + 1e-9)

        # Posterior if NO
        post_no = np.array([
            priors[i] * (1 - self.kb.get_P_symptom_given_disease(self.diseases[i], symptom_name))
            for i in range(len(self.diseases))
        ])
        post_no /= (post_no.sum() + 1e-9)

        # Entropy
        def entropy(p):
            p = p[p > 0]
            return -np.sum(p * np.log2(p))

        H_yes = entropy(post_yes)
        H_no = entropy(post_no)
        H_exp = p_s1 * H_yes + p_s0 * H_no

        return float(H_exp)

    # ----------------------------------------------
    def select_next_symptom(self) -> Dict[str, Any]:
        """
        Returns the symptom with the maximum expected information gain.
        JSON-friendly format for FastAPI.
        """
        current_entropy = self.engine.get_entropy()

        best_symptom = None
        best_gain = -np.inf

        for s in self.get_unasked_symptoms():
            H_exp = self.compute_expected_entropy(s)
            IG = current_entropy - H_exp
            if IG > best_gain:
                best_gain = IG
                best_symptom = s

        return {
            "symptom": best_symptom,
            "info_gain": float(best_gain)
        }

    # ----------------------------------------------
    def mark_asked(self, symptom_name: str) -> Dict[str, Any]:
        """
        Records that a symptom has been asked.
        """
        self.asked_symptoms.add(symptom_name)
        return {
            "status": "marked_asked",
            "symptom": symptom_name
        }

    # ----------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """
        Returns the internal state for debugging or frontend.
        """
        return {
            "asked_symptoms": sorted(list(self.asked_symptoms)),
            "unasked_symptoms": self.get_unasked_symptoms()
        }