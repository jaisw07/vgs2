from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid

from backend.core.session_manager import SessionManager
from backend.core.knowledge_base_service import KnowledgeBaseService
from backend.core.inference_service import InferenceService
from backend.core.csp_service import CSPService
from backend.core.entropy_service import EntropyService
from backend.core.logger_service import LoggerService
from backend.core.nlp_service import NLPService
from backend.core.diagnoser_service import DiagnoserService

router = APIRouter(prefix="/init", tags=["Initializer"])

session_manager = SessionManager()


# -------------------------------
# Request Model
# -------------------------------
class InitRequest(BaseModel):
    csv_path: str
    target_col: str = "prognosis"
    smoothing: float = 1.0
    confidence_threshold: float = 0.8
    max_questions: int = 20


# -------------------------------
# Full-System Initializer
# -------------------------------
@router.post("/full_pipeline")
def initialize_full(req: InitRequest):
    """
    Creates a fully-operational diagnosis session in ONE CALL.
    Initializes:
    - KB
    - Inference
    - CSP
    - Entropy
    - NLP
    - Logger
    - Diagnoser
    """
    session_id = str(uuid.uuid4())

    # --------------------------
    # Load Knowledge Base
    # --------------------------
    kb = KnowledgeBaseService(
        csv_path=req.csv_path,
        target_col=req.target_col,
        smoothing=req.smoothing
    )
    kb.load_dataset()
    kb.compute_probabilities()

    # --------------------------
    # Create Inference Engine
    # --------------------------
    inference = InferenceService(kb)

    # --------------------------
    # CSP & Entropy Engines
    # --------------------------
    csp = CSPService(kb)
    entropy = EntropyService(inference)

    # --------------------------
    # Logger
    # --------------------------
    logger = LoggerService(verbose=False)

    # --------------------------
    # NLP Parser
    # --------------------------
    nlp = NLPService(symptom_list=kb.get_symptom_list())

    # --------------------------
    # Diagnoser (final orchestrator)
    # --------------------------
    diagnoser = DiagnoserService(
        kb=kb,
        inference=inference,
        entropy=entropy,
        csp=csp,
        logger=logger,
        nlp=nlp,
        confidence_threshold=req.confidence_threshold,
        max_questions=req.max_questions
    )

    # Store everything in session
    session_manager.set(session_id, {
        "kb": kb,
        "inference": inference,
        "entropy": entropy,
        "csp": csp,
        "logger": logger,
        "nlp": nlp,
        "diagnoser": diagnoser
    })

    return {
        "status": "pipeline_initialized",
        "session_id": session_id,
        "details": {
            "num_symptoms": len(kb.get_symptom_list()),
            "num_diseases": len(kb.get_disease_list()),
            "confidence_threshold": req.confidence_threshold,
            "max_questions": req.max_questions
        }
    }