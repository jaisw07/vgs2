import numpy as np
from src.knowledge_base import KnowledgeBase
from src.inference_engine import InferenceEngine
from src.entropy_engine import EntropyEngine
from src.csp_module import CSPModule
from src.logger import SessionLogger

class InteractiveDiagnoser:
    def __init__(self, dataset_path: str, confidence_threshold: float = 0.8, max_questions: int = 20):
        # === Initialize components ===
        print("🧩 Initializing Knowledge Base...")
        self.kb = KnowledgeBase(dataset_path)
        self.kb.load_dataset()
        self.kb.compute_probabilities()

        print("🧠 Initializing Inference Engine...")
        self.engine = InferenceEngine(self.kb)

        print("🔍 Initializing Entropy Engine...")
        self.entropy = EntropyEngine(self.engine)

        print("⚙️ Initializing CSP Module...")
        self.csp = CSPModule(self.kb, verbose=True)

        print("🗄️  Initializing Session Logger...")
        self.logger = SessionLogger(verbose=True)

        self.confidence_threshold = confidence_threshold
        self.max_questions = max_questions
        self.user_answers = {}   # {symptom_name: 1/0/-1}

    # ---------------------------------------------------
    def ask_question(self, symptom: str):
        """Prompt user and record their response."""
        while True:
            ans = input(f"Do you have {symptom.replace('_', ' ')}? (y/n/u to skip): ").strip().lower()
            if ans in ["y", "n", "u"]:
                return {"y": 1, "n": 0, "u": -1}[ans]
            print("Please answer with 'y' (yes), 'n' (no), or 'u' (unknown).")

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

    # ---------------------------------------------------
    def run(self):
        """Main interactive diagnosis loop."""
        print("\n=== Welcome to the Interactive Diagnoser ===")
        print("Answer with (y)es, (n)o, or (u)nknown to each symptom question.\n")

        for step in range(self.max_questions):
            # Select next best symptom to ask
            symptom, gain = self.entropy.select_next_symptom()
            if not symptom:
                print("No further informative questions. Stopping.")
                break

            print(f"\n[{step+1}/{self.max_questions}] 🔎 Next question (IG={gain:.4f}):")
            response = self.ask_question(symptom)

            # Update beliefs + validate constraints
            updated = self.update_state(symptom, response)
            if not updated:
                continue

            # Display current belief state
            self.show_top_diseases(top_k=5)

            # Stopping condition: high confidence
            top_disease, top_prob = self.engine.get_top_diseases(1)[0]
            if top_prob >= self.confidence_threshold:
                print(f"✅ Confidence threshold reached ({top_prob*100:.2f}%).")
                break

        print("\n=== Final Diagnostic Report ===")
        self.show_top_diseases(top_k=5)
        print("Diagnosis complete. Thank you for using the system!")
        
        # Log session
        final_topk = self.engine.get_top_diseases(5)
        session_file = self.logger.log_session(
            user_answers=self.user_answers,
            final_topk=final_topk,
            engine=self.engine,
            confidence_threshold=self.confidence_threshold
        )
        self.logger.append_summary(final_topk, self.confidence_threshold, session_file)