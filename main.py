from src.csp_module import CSPModule
from src.knowledge_base import KnowledgeBase

kb = KnowledgeBase("data/symptoms_dataset.csv")
kb.load_dataset()
kb.compute_probabilities()

# Create CSP with verbose debugging on
csp = CSPModule(kb, verbose=True)

# Print a small sample of symptom names to confirm exact strings:
symptoms_sample = list(kb.get_symptom_list())[:20]
print("Sample symptoms (first 20):", symptoms_sample)

# Show all symptom names that look like 'fever'
matches = [s for s in kb.get_symptom_list() if "fever" in s.lower()]
print("Matching symptom names containing 'fever':", matches)

# # Add constraints (these will validate names on add)
csp.add_dependency("mild_fever", "fatigue")
csp.add_mutual_exclusion("high_fever", "fatigue")

# Simulate user responses
user_answers = {"mild_fever": 1, "fatigue": 0, "dry_cough": 1, "high_fever": 1}
valid, violations = csp.is_valid_state(user_answers)
print("State valid?", valid)