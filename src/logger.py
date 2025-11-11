import json
import os
import csv
from datetime import datetime


class SessionLogger:
    """
    Handles saving diagnostic sessions, model state, and metadata.
    Designed to be reusable across multiple modules (interactive, simulation, batch, etc.)
    """

    def __init__(self, base_dir: str = "results/sessions", verbose: bool = True):
        self.base_dir = base_dir
        self.verbose = verbose
        self._ensure_dirs()

    # -------------------------
    def _ensure_dirs(self):
        """Make sure session directory exists."""
        os.makedirs(self.base_dir, exist_ok=True)

    # -------------------------
    def log_session(
        self,
        user_answers,
        final_topk,
        engine,
        confidence_threshold,
        session_id=None,
        nlp_input_text=None,
        nlp_parsed_symptoms=None,
        csp_skipped=None,
    ):
        """
        Save a session summary to JSON.
        Args:
            user_answers: dict[str, int]
            final_topk: list[(disease, probability)]
            engine: InferenceEngine object (for priors/posteriors)
            confidence_threshold: float
            session_id: optional custom ID (for simulation runs)
            nlp_input_text: str (raw free-text input)
            nlp_parsed_symptoms: dict[str, int] (parsed NLP results before CSP)
            csp_skipped: list[str] (symptoms ignored due to constraints)
        """
        ts = datetime.utcnow()
        if not session_id:
            session_id = ts.strftime('%Y%m%dT%H%M%S')

        # Compute counts
        total_symptoms = len(user_answers)
        nlp_count = len(nlp_parsed_symptoms or {})
        nlp_used = bool(nlp_input_text)

        session = {
            "session_id": session_id,
            "timestamp": ts.isoformat() + "Z",
            "confidence_threshold": confidence_threshold,
            "num_questions_asked": total_symptoms,
            "nlp_used": nlp_used,
            "nlp_input_text": nlp_input_text,
            "nlp_parsed_symptoms": nlp_parsed_symptoms,
            "csp_skipped_symptoms": csp_skipped or [],
            "num_symptoms_from_nlp": nlp_count,
            "answers": {k: int(v) for k, v in user_answers.items()},
            "final_topk": [(d, float(p)) for d, p in final_topk],
            "priors": {d: float(p) for d, p in zip(engine.diseases, engine.priors)},
            "posteriors": {d: float(p) for d, p in engine.get_top_diseases(len(engine.diseases))},
        }

        # Save to JSON
        fname = os.path.join(self.base_dir, f"session_{session_id}.json")
        with open(fname, "w") as f:
            json.dump(session, f, indent=2)

        if self.verbose:
            print(f"🗂️  Session logged to {fname}")

        return fname

    # -------------------------
    def append_summary(self, final_topk, confidence_threshold, session_file, nlp_used=False):
        """Append key session info to results/sessions/summary.csv"""
        summary_path = os.path.join(self.base_dir, "summary.csv")
        top_disease, top_prob = final_topk[0]
        write_header = not os.path.exists(summary_path)

        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "session_file", "top_disease", "confidence", "threshold", "nlp_used"])
            writer.writerow([
                datetime.utcnow().isoformat() + "Z",
                os.path.basename(session_file),
                top_disease,
                round(top_prob, 3),
                confidence_threshold,
                int(nlp_used)
            ])

        if self.verbose:
            print(f"📈 Summary updated: {summary_path}")