import json
import os
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
    def log_session(self, user_answers, final_topk, engine, confidence_threshold, session_id=None):
        """
        Save a session summary to JSON.
        Args:
            user_answers: dict[str, int]
            final_topk: list[(disease, probability)]
            engine: InferenceEngine object (for priors/posteriors)
            confidence_threshold: float
            session_id: optional custom ID (for simulation runs)
        """
        ts = datetime.utcnow()
        if not session_id:
            session_id = ts.strftime('%Y%m%dT%H%M%S')

        session = {
            'session_id': session_id,
            'timestamp': ts.isoformat() + 'Z',
            'confidence_threshold': confidence_threshold,
            'num_questions_asked': len(user_answers),
            'answers': {k: int(v) for k, v in user_answers.items()},
            'final_topk': [(d, float(p)) for d, p in final_topk],
            'priors': {d: float(p) for d, p in zip(engine.diseases, engine.priors)},
            'posteriors': {d: float(p) for d, p in engine.get_top_diseases(len(engine.diseases))}
        }

        # Save to JSON
        fname = os.path.join(self.base_dir, f"session_{session_id}.json")
        with open(fname, 'w') as f:
            json.dump(session, f, indent=2)

        if self.verbose:
            print(f"🗂️  Session logged to {fname}")

        return fname