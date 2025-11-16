from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from backend.core.session_manager import SessionManager
from backend.core.nlp_service import NLPService

router = APIRouter(prefix="/nlp", tags=["NLP"])

session_manager = SessionManager()

# -----------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------

class CreateNLPSessionRequest(BaseModel):
    session_id: str              # Must already exist (KB loaded)
    symptoms: List[str]          # Inject symptom list (usually from KB)


class ParseTextRequest(BaseModel):
    session_id: str
    text: str


class UpdateSymptomsRequest(BaseModel):
    session_id: str
    symptoms: List[str]


class SessionOnly(BaseModel):
    session_id: str


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def get_nlp(session_id: str) -> NLPService:
    session = session_manager.get(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Invalid session_id")

    if "nlp" not in session:
        raise HTTPException(status_code=500, detail="NLPService not initialized")

    return session["nlp"]


# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------

@router.post("/create_session")
def create_nlp_session(req: CreateNLPSessionRequest):
    """
    Attach NLPService to an existing knowledge/inference session.
    """
    session = session_manager.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    nlp = NLPService(symptom_list=req.symptoms, verbose=False)

    session["nlp"] = nlp
    session_manager.set(req.session_id, session)

    return {
        "session_id": req.session_id,
        "status": "nlp_initialized",
        "n_symptoms": len(req.symptoms)
    }


@router.post("/parse")
def parse_text(req: ParseTextRequest):
    """
    Parse free-text user input into symptom states.
    """
    nlp = get_nlp(req.session_id)
    return nlp.parse_text(req.text)


@router.post("/update_symptoms")
def update_symptoms(req: UpdateSymptomsRequest):
    """
    Update the NLP vocabulary (usually after KB reload).
    """
    nlp = get_nlp(req.session_id)
    return nlp.update_symptom_list(req.symptoms)


@router.get("/state")
def get_state(session_id: str):
    """
    Debug endpoint: return internal NLP parser settings.
    """
    nlp = get_nlp(session_id)
    return nlp.get_state()