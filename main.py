from src.knowledge_base import KnowledgeBase
from src.inference_engine import InferenceEngine

kb = KnowledgeBase("data/symptoms_dataset.csv")
kb.load_dataset()
kb.compute_probabilities()

engine = InferenceEngine(kb)

print("\n--- Initial Priors ---")
for d, p in engine.get_top_diseases(5):
    print(f"{d}: {p:.4f}")

# Simulate answers
print("\nUser says: Fever = Yes")
engine.update_beliefs("fever", 1)

print("\nUser says: Cough = Yes")
engine.update_beliefs("cough", 1)

print("\n--- Updated Top Diseases ---")
for d, p in engine.get_top_diseases(5):
    print(f"{d}: {p:.4f}")