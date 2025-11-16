from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import uuid

from backend.core.session_manager import SessionManager
from backend.core.entropy_service import EntropyService
from backend.core.knowledge_base_loader import get_knowledge_base
from backend.core.inference_service import InferenceService

router = APIRouter(prefix="/entropy", tags=["Entropy"])

# Shared session store
session_manager = SessionManager()

# -------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------

class CreateEntropySessionRequest(BaseModel):
    session_id: str  # Must already exist in inference session


class GenericSessionResponse(BaseModel):
    session_id: str


class MarkAskedRequest(BaseModel):
    session_id: str
    symptom: str


class GetNextSymptomRequest(BaseModel):
    session_id: str


# -------------------------------------------------------
# Helper
# -------------------------------------------------------

def get_entropy_service(session_id: str) -> EntropyService:
    """
    This retrieves the EntropyService tied to an existing
    InferenceService session.
    """
    session = session_manager.get(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Invalid session_id.")

    if "entropy" not in session:
        raise HTTPException(status_code=500, detail="Entropy service not initialized.")

    return session["entropy"]


def get_inference_service(session_id: str) -> InferenceService:
    session = session_manager.get(session_id)
    if session is None or "inference" not in session:
        raise HTTPException(status_code=404, detail="Inference session not found.")
    return session["inference"]


# -------------------------------------------------------
# Routes
# -------------------------------------------------------

@router.post("/create_session", response_model=GenericSessionResponse)
def create_entropy_session(req: CreateEntropySessionRequest):
    """
    Create an EntropyService for a session that already has an InferenceService.
    """
    session_id = req.session_id

    inf = get_inference_service(session_id)

    entropy = EntropyService(inference_engine=inf)

    # Add entropy service to the user session
    session = session_manager.get(session_id)
    session["entropy"] = entropy
    session_manager.set(session_id, session)

    return GenericSessionResponse(session_id=session_id)


# -------------------------------------------------------
# Get next best symptom
# -------------------------------------------------------

@router.get("/next")
def get_next_symptom(session_id: str):
    entropy = get_entropy_service(session_id)
    return entropy.select_next_symptom()


# -------------------------------------------------------
# Mark a symptom as asked
# -------------------------------------------------------

@router.post("/mark_asked")
def mark_asked(req: MarkAskedRequest):
    entropy = get_entropy_service(req.session_id)
    return entropy.mark_asked(req.symptom)


# -------------------------------------------------------
# Return all unasked symptoms
# -------------------------------------------------------

@router.get("/unasked")
def get_unasked_symptoms(session_id: str):
    entropy = get_entropy_service(session_id)
    return {
        "unasked_symptoms": entropy.get_unasked_symptoms()
    }


# -------------------------------------------------------
# Internal debug: state snapshot
# -------------------------------------------------------

@router.get("/state")
def get_state(session_id: str):
    entropy = get_entropy_service(session_id)
    return entropy.get_state()