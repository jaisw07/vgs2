from src.knowledge_base import KnowledgeBase

kb = KnowledgeBase("data/symptoms_dataset.csv")
kb.load_dataset()
kb.compute_probabilities()
kb.save_disease_similarity()
kb.visualize()