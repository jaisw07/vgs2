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
        """Main interactive diagnosis loop (supports free-text + questions)."""
        print("\n=== Welcome to the Interactive Diagnoser ===")
        print("Answer with (y)es, (n)o, or (u)nknown to each symptom question.\n")

        # --- STEP 1: Optional NLP input ---
        mode = input("Would you like to describe your symptoms in free text? (y/n): ").strip().lower()
        if mode == 'y':
            self.nlp_mode = True
            text = input("Please describe your symptoms: ")
            self.nlp_input_text = text
            parsed = self.parser.parse_text(text)
            # ✅ Filter to only include positive (v == 1) symptoms
            positive_parsed = {s: v for s, v in parsed.items() if v == 1}
            self.nlp_parsed = positive_parsed

            print("\n🧠 Parsed symptoms:")
            if positive_parsed:
                for s in positive_parsed:
                    print(f" - {s.replace('_', ' ')} ✅")
            else:
                print("No clear positive symptoms detected.")
            
            skipped_symptoms = []

            # Validate + update the inference model with NLP results
            for symptom, value in parsed.items():
                if value == 1:
                    valid, violations = self.csp.is_valid_state({**self.user_answers, symptom: value})
                    if valid:
                        self.user_answers[symptom] = value
                        self.engine.update_beliefs(symptom, value)
                        self.entropy.mark_asked(symptom)
                    else:
                        skipped_symptoms.append(symptom)
                        print(f"⚠️ Skipped {symptom} due to constraint conflict: {violations}")


            # Show current top diseases before continuing
            self.show_top_diseases(top_k=5)

            cont = input("\nWould you like to continue answering more questions? (y/n): ").strip().lower()
            if cont != 'y':
                final_topk = self.engine.get_top_diseases(5)
                session_file = self.logger.log_session(
                    user_answers=self.user_answers,
                    final_topk=final_topk,
                    engine=self.engine,
                    confidence_threshold=self.confidence_threshold,
                    nlp_input_text=text if mode == 'y' else None,
                    nlp_parsed_symptoms=parsed if mode == 'y' else None,
                    csp_skipped=skipped_symptoms if mode == 'y' else None
                )
                self.logger.append_summary(final_topk, self.confidence_threshold, session_file, nlp_used=(mode == 'y'))
                print("Diagnosis complete. Thank you for using the system!")
                return

        # --- STEP 2: Interactive question loop ---
        for step in range(self.max_questions):
            symptom, gain = self.entropy.select_next_symptom()

            # Skip symptoms already answered via NLP or user input
            if not symptom or symptom in self.user_answers:
                continue

            print(f"\n[{step+1}/{self.max_questions}] 🔎 Next question (IG={gain:.4f}):")
            response = self.ask_question(symptom)

            # Update beliefs + validate constraints
            updated = self.update_state(symptom, response)
            if not updated:
                continue

            self.show_top_diseases(top_k=5)

            # Stopping condition
            top_disease, top_prob = self.engine.get_top_diseases(1)[0]
            if top_prob >= self.confidence_threshold:
                print(f"✅ Confidence threshold reached ({top_prob*100:.2f}%).")
                break

        # --- STEP 3: Final summary ---
        print("\n=== Final Diagnostic Report ===")
        self.show_top_diseases(top_k=5)
        print("Diagnosis complete. Thank you for using the system!  ")

        final_topk = self.engine.get_top_diseases(5)
        session_file = self.logger.log_session(
        user_answers=self.user_answers,
        final_topk=final_topk,
        engine=self.engine,
        confidence_threshold=self.confidence_threshold,
        nlp_input_text=self.nlp_input_text if self.nlp_mode else None,
        nlp_parsed_symptoms=self.nlp_parsed if self.nlp_mode else None,
        csp_skipped=self.nlp_skipped if self.nlp_mode else None
        )
        self.logger.append_summary(final_topk, self.confidence_threshold, session_file, nlp_used=self.nlp_mode)