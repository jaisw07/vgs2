from src.knowledge_base import KnowledgeBase
from src.inference_engine import InferenceEngine
from src.entropy_engine import EntropyEngine

kb = KnowledgeBase("data/symptoms_dataset.csv")
kb.load_dataset()
kb.compute_probabilities()

engine = InferenceEngine(kb)
entropy_engine = EntropyEngine(engine)

symptom, gain = entropy_engine.select_next_symptom()
print(f"🧩 Next best symptom to ask: '{symptom}' (Expected information gain: {gain:.4f})")