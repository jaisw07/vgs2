from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from backend.core.session_manager import SessionManager
from backend.core.diagnoser_service import DiagnoserService

router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])

session_manager = SessionManager()

# -----------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------

class StartDiagnosisRequest(BaseModel):
    session_id: str
    confidence_threshold: float = 0.8
    max_questions: int = 20


class NLPRequest(BaseModel):
    session_id: str
    text: str


class AnswerRequest(BaseModel):
    session_id: str
    symptom: str
    response: int    # 1=yes, 0=no, -1=unknown


class GenericSession(BaseModel):
    session_id: str


# -----------------------------------------------------------
# Helper
# -----------------------------------------------------------

def get_diagnoser(session_id: str) -> DiagnoserService:
    """
    Retrieve DiagnoserService from session.
    """
    session = session_manager.get(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Invalid session_id")

    if "diagnoser" not in session:
        raise HTTPException(status_code=500, detail="DiagnoserService not initialized")

    return session["diagnoser"]


def ensure_components_exist(session_id: str):
    """
    Ensures that all required backend components are initialized.
    """
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    required = ["kb", "inference", "entropy", "csp", "nlp", "logger"]
    missing = [name for name in required if name not in session]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing components for session {session_id}: {missing}"
        )

    return session


# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------

@router.post("/start")
def start_diagnosis(req: StartDiagnosisRequest):
    """
    Initialize a DiagnoserService for this session.
    All underlying components must already exist:
    - KnowledgeBaseService
    - InferenceService
    - EntropyService
    - CSPService
    - NLPService
    - LoggerService
    """

    session = ensure_components_exist(req.session_id)

    diagnoser = DiagnoserService(
        kb=session["kb"],
        inference=session["inference"],
        entropy=session["entropy"],
        csp=session["csp"],
        logger=session["logger"],
        nlp=session["nlp"],
        confidence_threshold=req.confidence_threshold,
        max_questions=req.max_questions,
    )

    session["diagnoser"] = diagnoser
    session_manager.set(req.session_id, session)

    return {
        "status": "diagnosis_initialized",
        "session_id": req.session_id,
        "confidence_threshold": req.confidence_threshold,
        "max_questions": req.max_questions,
    }


@router.post("/nlp")
def process_nlp(req: NLPRequest):
    diagnoser = get_diagnoser(req.session_id)
    return diagnoser.process_free_text(req.text)


@router.get("/next_question")
def next_question(session_id: str):
    diagnoser = get_diagnoser(session_id)
    return diagnoser.get_next_question()


@router.post("/answer")
def answer(req: AnswerRequest):
    diagnoser = get_diagnoser(req.session_id)

    if req.response not in [1, 0, -1]:
        raise HTTPException(status_code=400, detail="Response must be 1, 0, or -1")

    return diagnoser.submit_answer(req.symptom, req.response)


@router.get("/progress")
def progress(session_id: str):
    diagnoser = get_diagnoser(session_id)
    return diagnoser.get_progress()


@router.get("/is_done")
def is_done(session_id: str):
    diagnoser = get_diagnoser(session_id)
    return diagnoser.is_done()


@router.post("/finish")
def finish(req: GenericSession):
    diagnoser = get_diagnoser(req.session_id)
    return diagnoser.finalize_session(req.session_id)