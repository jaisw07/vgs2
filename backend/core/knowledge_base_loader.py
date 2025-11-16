from functools import lru_cache
from src.knowledge_base import KnowledgeBase

@lru_cache()
def get_knowledge_base():
    kb = KnowledgeBase("path/to/your/dataset.csv")
    kb.load_dataset()
    kb.compute_probabilities()
    return kb