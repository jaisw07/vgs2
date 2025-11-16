import json
import os
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional


class LoggerService:
    """
    FastAPI-compatible version of SessionLogger.
    - No print statements
    - Returns JSON-safe results
    - Writes files safely
    """

    def __init__(self, base_dir: str = "results/sessions", verbose: bool = False):
        self.base_dir = base_dir
        self.verbose = verbose
        self._ensure_dirs()

    # -------------------------------------------------------
    def _ensure_dirs(self):
        """Ensure the session directory exists."""
        os.makedirs(self.base_dir, exist_ok=True)

    # -------------------------------------------------------
    def log_session(
        self,
        user_answers: Dict[str, int],
        final_topk: List[tuple],
        inference_engine,
        confidence_threshold: float,
        session_id: Optional[str] = None,
        nlp_input_text: Optional[str] = None,
        nlp_parsed_symptoms: Optional[Dict[str, int]] = None,
        csp_skipped: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Save a diagnostic session into JSON log.

        Returns:
            {
              "status": "logged",
              "session_id": ...,
              "file": ...,
              "summary": { ...small metadata... }
            }
        """

        ts = datetime.utcnow()

        if not session_id:
            session_id = ts.strftime("%Y%m%dT%H%M%S")

        # Compute summary items
        total_symptoms = len(user_answers)
        nlp_count = len(nlp_parsed_symptoms or {})
        nlp_used = bool(nlp_input_text)

        session_json = {
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
            "priors": {
                d: float(p) for d, p in zip(inference_engine.diseases, inference_engine.priors)
            },
            "posteriors": {
                d: float(p)
                for d, p in zip(
                    [d for d, _ in final_topk],
                    [p for _, p in final_topk]
                )
            },
        }

        # Save JSON file
        fname = os.path.join(self.base_dir, f"session_{session_id}.json")
        with open(fname, "w") as f:
            json.dump(session_json, f, indent=2)

        return {
            "status": "logged",
            "session_id": session_id,
            "file": fname,
            "summary": {
                "top_disease": final_topk[0][0] if final_topk else None,
                "confidence": float(final_topk[0][1]) if final_topk else None,
                "num_answers": total_symptoms,
                "nlp_used": nlp_used,
            }
        }

    # -------------------------------------------------------
    def append_summary(
        self,
        final_topk: List[tuple],
        confidence_threshold: float,
        session_file: str,
        nlp_used: bool = False
    ) -> Dict[str, Any]:
        """
        Append a compact session summary to summary.csv.

        Returns:
            {
              "status": "summary_appended",
              "file": "results/sessions/summary.csv"
            }
        """

        summary_path = os.path.join(self.base_dir, "summary.csv")
        write_header = not os.path.exists(summary_path)

        top_disease, top_prob = final_topk[0]

        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    "timestamp",
                    "session_file",
                    "top_disease",
                    "confidence",
                    "threshold",
                    "nlp_used"
                ])

            writer.writerow([
                datetime.utcnow().isoformat() + "Z",
                os.path.basename(session_file),
                top_disease,
                round(float(top_prob), 3),
                confidence_threshold,
                int(nlp_used)
            ])

        return {
            "status": "summary_appended",
            "file": summary_path
        }