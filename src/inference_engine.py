import numpy as np

class InferenceEngine:
    def __init__(self, knowledge_base):
        """
        Parameters:
            knowledge_base : KnowledgeBase
                An instance of the KnowledgeBase class containing probabilities.
        """
        self.kb = knowledge_base
        self.diseases = knowledge_base.get_disease_list()
        self.symptoms = knowledge_base.get_symptom_list()
        
        # Initialize priors
        self.priors = np.array([self.kb.get_P_disease(d) for d in self.diseases])
        self.posteriors = self.priors.copy()

    # ---------------------------------------------------
    def reset(self):
        """Reset posterior to priors (start new session)."""
        self.posteriors = self.priors.copy()

    # ---------------------------------------------------
    def update_beliefs(self, symptom_name: str, user_response: int):
        """
        Update P(Disease | observed symptoms) using Bayes' rule.

        Parameters:
            symptom_name : str
                The symptom being answered.
            user_response : int
                1 = Yes, symptom present
                0 = No, symptom absent
               -1 = Unknown (ignored)
        """
        if user_response == -1:
            # Don't update for unknown
            return self.posteriors
        
        likelihoods = []
        for disease in self.diseases:
            p_symptom_given_disease = self.kb.get_P_symptom_given_disease(disease, symptom_name)
            if user_response == 1:  # user says Yes
                likelihood = p_symptom_given_disease
            elif user_response == 0:  # user says No
                likelihood = 1 - p_symptom_given_disease
            likelihoods.append(likelihood)
        
        likelihoods = np.array(likelihoods)
        
        # Apply Bayes’ rule
        numerators = self.posteriors * likelihoods
        if numerators.sum() == 0:
            numerators += 1e-9  # numerical safeguard
        
        self.posteriors = numerators / numerators.sum()
        return self.posteriors

    # ---------------------------------------------------
    def get_top_diseases(self, top_k=5):
        """Return top-k diseases with their probabilities."""
        sorted_indices = np.argsort(self.posteriors)[::-1]
        top = [(self.diseases[i], float(self.posteriors[i])) for i in sorted_indices[:top_k]]
        return top

    # ---------------------------------------------------
    def get_entropy(self):
        """Compute entropy of the current posterior distribution."""
        p = self.posteriors[self.posteriors > 0]
        entropy = -np.sum(p * np.log2(p))
        return entropy