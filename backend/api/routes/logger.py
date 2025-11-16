from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional

from backend.core.session_manager import SessionManager
from backend.core.logger_service import LoggerService

router = APIRouter(prefix="/logger", tags=["Logger"])

session_manager = SessionManager()

# -----------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------

class CreateLoggerSessionRequest(BaseModel):
    session_id: str                  # Must already exist
    base_dir: Optional[str] = None   # Optional override


class LogSessionRequest(BaseModel):
    session_id: str
    user_answers: Dict[str, int]
    final_topk: List[tuple]
    confidence_threshold: float
    nlp_input_text: Optional[str] = None
    nlp_parsed_symptoms: Optional[Dict[str, int]] = None
    csp_skipped: Optional[List[str]] = None


class AppendSummaryRequest(BaseModel):
    session_id: str
    final_topk: List[tuple]
    confidence_threshold: float
    session_file: str
    nlp_used: bool = False


class GenericSession(BaseModel):
    session_id: str


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def get_logger(session_id: str) -> LoggerService:
    session = session_manager.get(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Invalid session_id")

    if "logger" not in session:
        raise HTTPException(status_code=500, detail="LoggerService not initialized")

    return session["logger"]


def get_inference_service(session_id: str):
    session = session_manager.get(session_id)
    if session is None or "inference" not in session:
        raise HTTPException(status_code=500, detail="InferenceService missing for this session")
    return session["inference"]


# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------

@router.post("/create_session")
def create_logger_session(req: CreateLoggerSessionRequest):
    """
    Attach a LoggerService to an existing session.
    Creates base_dir if needed.
    """
    session = session_manager.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    logger = LoggerService(
        base_dir=req.base_dir if req.base_dir else "results/sessions",
        verbose=False
    )

    session["logger"] = logger
    session_manager.set(req.session_id, session)

    return {
        "status": "logger_initialized",
        "session_id": req.session_id,
        "base_dir": logger.base_dir
    }


@router.post("/log")
def log_session(req: LogSessionRequest):
    """
    Writes a JSON log for the current diagnostic session.
    """
    logger = get_logger(req.session_id)
    inference = get_inference_service(req.session_id)

    return logger.log_session(
        user_answers=req.user_answers,
        final_topk=req.final_topk,
        engine=inference,
        confidence_threshold=req.confidence_threshold,
        session_id=req.session_id,                    # unify session log filenames
        nlp_input_text=req.nlp_input_text,
        nlp_parsed_symptoms=req.nlp_parsed_symptoms,
        csp_skipped=req.csp_skipped,
    )


@router.post("/append_summary")
def append_summary(req: AppendSummaryRequest):
    """
    Appends session summary to summary.csv
    """
    logger = get_logger(req.session_id)
    return logger.append_summary(
        final_topk=req.final_topk,
        confidence_threshold=req.confidence_threshold,
        session_file=req.session_file,
        nlp_used=req.nlp_used,
    )


@router.get("/state")
def logger_state(session_id: str):
    """
    Since logger doesn't have a complex state,
    just return base_dir if needed.
    """
    logger = get_logger(session_id)
    return {
        "base_dir": logger.base_dir,
        "status": "logger_ready"
    }