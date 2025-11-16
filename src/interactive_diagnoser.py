import numpy as np
from src.knowledge_base import KnowledgeBase
from src.inference_engine import InferenceEngine
from src.entropy_engine import EntropyEngine
from src.csp_module import CSPModule
from src.logger import SessionLogger
from src.nlp_parser import SymptomNLPParser

class InteractiveDiagnoser:
    def __init__(self, dataset_path: str, confidence_threshold: float = 0.8, max_questions: int = 20):
        # === Initialize components ===
        print("🧩 Initializing Knowledge Base...")
        self.kb = KnowledgeBase(dataset_path)
        self.kb.load_dataset()
        self.kb.compute_probabilities()
        self.nlp_mode = False
        self.nlp_input_text = None
        self.nlp_parsed = None
        self.nlp_skipped = []

        print("🧠 Initializing Inference Engine...")
        self.engine = InferenceEngine(self.kb)

        print("🔍 Initializing Entropy Engine...")
        self.entropy = EntropyEngine(self.engine)

        print("⚙️ Initializing CSP Module...")
        self.csp = CSPModule(self.kb, verbose=False)

        print("🗄️  Initializing Session Logger...")
        self.logger = SessionLogger(verbose=True)

        print("📝 Initializing NLP Parser...")
        self.parser = SymptomNLPParser(self.kb.get_symptom_list(), verbose=False)

        self.confidence_threshold = confidence_threshold
        self.max_questions = max_questions
        self.user_answers = {}   # {symptom_name: 1/0/-1}

    # ---------------------------------------------------
    def update_state(self, symptom, response):
        """Update inference + CSP validation."""
        # Tentatively update answers
        temp_answers = self.user_answers.copy()
        temp_answers[symptom] = response

        # Validate with CSP
        valid, violations = self.csp.is_valid_state(temp_answers)
        if not valid:
            print("❌ Invalid combination detected due to constraints:")
            for v in violations:
                print("   -", v)
            print("Skipping this answer. Try again.")
            return False

        # Update belief system
        self.user_answers[symptom] = response
        self.engine.update_beliefs(symptom, response)
        self.entropy.mark_asked(symptom)
        return True

    # ---------------------------------------------------
    def show_top_diseases(self, top_k=5):
        top = self.engine.get_top_diseases(top_k)
        print("\n🩺 Current top possible diseases:")
        for d, p in top:
            print(f" - {d:25s} : {p*100:.2f}%")
        print()