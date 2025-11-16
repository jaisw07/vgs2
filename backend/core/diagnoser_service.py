from typing import Dict, Any, List, Optional


class DiagnoserService:
    """
    FastAPI-compatible refactor of InteractiveDiagnoser.
    Orchestrates: KB → NLP → CSP → Inference → Entropy → Logging
    """

    def __init__(
        self,
        kb,
        inference,
        entropy,
        csp,
        nlp,
        logger,
        confidence_threshold: float = 0.8,
        max_questions: int = 20,
    ):
        self.kb = kb
        self.inference = inference
        self.entropy = entropy
        self.csp = csp
        self.logger = logger
        self.nlp = nlp

        self.user_answers: Dict[str, int] = {}
        self.nlp_input_text: Optional[str] = None
        self.nlp_parsed: Optional[Dict[str, int]] = None
        self.nlp_skipped: List[str] = []

        self.confidence_threshold = confidence_threshold
        self.max_questions = max_questions
        self.num_questions_asked = 0

    # --------------------------------------------------------
    # NLP processing
    # --------------------------------------------------------
    def process_free_text(self, text: str) -> Dict[str, Any]:
        """
        Parse free-text symptoms via NLP and pre-apply valid symptoms.
        """
        parsed = self.nlp.parse_text(text)
        symptom_states = parsed["symptom_states"]

        self.nlp_input_text = text
        self.nlp_parsed = symptom_states

        skipped = []

        for symptom, value in symptom_states.items():
            if value == 1:
                temp_state = {**self.user_answers, symptom: 1}
                csp_valid = self.csp.is_valid_state(temp_state)

                if not csp_valid["valid"]:
                    skipped.append(symptom)
                    continue

                # Accept NLP-positive symptom
                self.user_answers[symptom] = 1
                self.inference.update_beliefs(symptom, 1)
                self.entropy.mark_asked(symptom)

        self.nlp_skipped = skipped

        return {
            "parsed_symptoms": symptom_states,
            "positive_symptoms": parsed["positive_symptoms"],
            "skipped_due_to_csp": skipped,
            "top_diseases": self.inference.get_top_diseases(5),
        }

    # --------------------------------------------------------
    # Entropy-based question selection
    # --------------------------------------------------------
    def get_next_question(self) -> Dict[str, Any]:
        """
        Returns next symptom with max information gain.
        """
        next_q = self.entropy.select_next_symptom()
        symptom = next_q["symptom"]
        ig = next_q["info_gain"]

        if symptom is None:
            return {"done": True, "reason": "no_symptoms_left"}

        return {
            "symptom": symptom,
            "info_gain": ig,
            "num_asked": self.num_questions_asked,
        }

    # --------------------------------------------------------
    # Answering / updating state
    # --------------------------------------------------------
    def submit_answer(self, symptom: str, response: int) -> Dict[str, Any]:
        """
        Update inference and CSP state from user answer.
        """
        temp = {**self.user_answers, symptom: response}
        validity = self.csp.is_valid_state(temp)

        if not validity["valid"]:
            return {"accepted": False, "violations": validity["violations"]}

        # Apply update
        self.user_answers[symptom] = response
        self.inference.update_beliefs(symptom, response)
        self.entropy.mark_asked(symptom)

        self.num_questions_asked += 1

        # Check for stopping condition
        top = self.inference.get_top_diseases(1)["top_diseases"][0]
        if top["probability"] >= self.confidence_threshold:
            return {
                "accepted": True,
                "stopping": True,
                "top_disease": top,
                "top5": self.inference.get_top_diseases(5)
            }

        return {
            "accepted": True,
            "stopping": False,
            "top5": self.inference.get_top_diseases(5)
        }

    # --------------------------------------------------------
    # Utility methods for frontend
    # --------------------------------------------------------
    def get_progress(self) -> Dict[str, Any]:
        return {
            "num_asked": self.num_questions_asked,
            "max_questions": self.max_questions,
            "user_answers": self.user_answers,
            "top5": self.inference.get_top_diseases(5),
        }

    def is_done(self) -> Dict[str, Any]:
        top1 = self.inference.get_top_diseases(1)["top_diseases"][0]
        if top1["probability"] >= self.confidence_threshold:
            return {"done": True, "top_disease": top1}
        return {"done": False}

    # --------------------------------------------------------
    # Final summary + logging
    # --------------------------------------------------------
    def finalize_session(self, session_id: str) -> Dict[str, Any]:
        """
        Log full diagnostic session.
        """
        final_topk = self.inference.get_top_diseases(5)["top_diseases"]

        log = self.logger.log_session(
            user_answers=self.user_answers,
            final_topk=[(d["disease"], d["probability"]) for d in final_topk],
            engine=self.inference,
            confidence_threshold=self.confidence_threshold,
            session_id=session_id,
            nlp_input_text=self.nlp_input_text,
            nlp_parsed_symptoms=self.nlp_parsed,
            csp_skipped=self.nlp_skipped,
        )

        summary = self.logger.append_summary(
            final_topk=[(d["disease"], d["probability"]) for d in final_topk],
            confidence_threshold=self.confidence_threshold,
            session_file=log["file"],
            nlp_used=bool(self.nlp_input_text)
        )

        return {
            "status": "session_completed",
            "final_top5": final_topk,
            "log_file": log["file"],
            "summary_file": summary["file"]
        }