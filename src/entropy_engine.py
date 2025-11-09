import numpy as np

class EntropyEngine:
    def __init__(self, inference_engine):
        """
        Parameters:
            inference_engine : InferenceEngine
                An instance of the current inference engine.
        """
        self.engine = inference_engine
        self.kb = inference_engine.kb
        self.diseases = inference_engine.diseases
        self.symptoms = inference_engine.symptoms
        
        # Track symptoms already asked
        self.asked_symptoms = set()

    # ---------------------------------------------------
    def get_unasked_symptoms(self):
        """Return list of symptoms not yet asked."""
        return [s for s in self.symptoms if s not in self.asked_symptoms]

    # ---------------------------------------------------
    def compute_expected_entropy(self, symptom_name):
        """
        Compute expected posterior entropy if we were to ask about 'symptom_name'.
        """
        priors = self.engine.posteriors.copy()
        
        # Compute P(S=1) and P(S=0)
        p_s1 = np.sum([priors[i] * self.kb.get_P_symptom_given_disease(self.diseases[i], symptom_name)
                       for i in range(len(self.diseases))])
        p_s0 = 1 - p_s1
        
        # Compute posterior if user says "Yes"
        post_yes = np.array([
            priors[i] * self.kb.get_P_symptom_given_disease(self.diseases[i], symptom_name)
            for i in range(len(self.diseases))
        ])
        post_yes /= (post_yes.sum() + 1e-9)
        
        # Compute posterior if user says "No"
        post_no = np.array([
            priors[i] * (1 - self.kb.get_P_symptom_given_disease(self.diseases[i], symptom_name))
            for i in range(len(self.diseases))
        ])
        post_no /= (post_no.sum() + 1e-9)
        
        # Entropy helper
        def entropy(p):
            p = p[p > 0]
            return -np.sum(p * np.log2(p))
        
        H_yes = entropy(post_yes)
        H_no = entropy(post_no)
        H_exp = p_s1 * H_yes + p_s0 * H_no
        
        return H_exp

    # ---------------------------------------------------
    def select_next_symptom(self):
        """
        Choose the symptom that maximizes expected information gain.
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
        
        return best_symptom, best_gain

    # ---------------------------------------------------
    def mark_asked(self, symptom_name):
        """Mark a symptom as asked (regardless of response)."""
        self.asked_symptoms.add(symptom_name)