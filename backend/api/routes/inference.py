from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid

from backend.core.session_manager import SessionManager
from backend.core.inference_service import InferenceService
from backend.core.knowledge_base_loader import get_knowledge_base

router = APIRouter(prefix="/inference", tags=["Inference"])

session_manager = SessionManager()

# -----------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------

class CreateSessionResponse(BaseModel):
    session_id: str


class UpdateBeliefRequest(BaseModel):
    session_id: str
    symptom: str
    response: int  # 1=yes, 0=no, -1=unknown


class GenericSessionRequest(BaseModel):
    session_id: str


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def get_inference_service(session_id: str) -> InferenceService:
    session = session_manager.get(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Invalid session_id")

    if "inference" not in session:
        raise HTTPException(status_code=500, detail="InferenceService not initialized")

    return session["inference"]


# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------

@router.post("/create_session", response_model=CreateSessionResponse)
def create_inference_session():
    """
    Creates a complete inference session:
    - Loads knowledge base
    - Creates InferenceService
    - Stores in session manager

    Other services (entropy, csp) will attach to this session_id later.
    """
    session_id = str(uuid.uuid4())

    kb = get_knowledge_base()
    inference = InferenceService(kb)

    session_manager.set(session_id, {"inference": inference})

    return CreateSessionResponse(session_id=session_id)


# -----------------------------------------------------------
# Reset posterior
# -----------------------------------------------------------

@router.post("/reset")
def reset(req: GenericSessionRequest):
    inf = get_inference_service(req.session_id)
    return inf.reset()


# -----------------------------------------------------------
# Update beliefs
# -----------------------------------------------------------

@router.post("/update")
def update_beliefs(req: UpdateBeliefRequest):
    inf = get_inference_service(req.session_id)

    if req.response not in [1, 0, -1]:
        raise HTTPException(status_code=400, detail="Response must be 1, 0, or -1")

    if req.symptom not in inf.symptoms:
        raise HTTPException(status_code=400, detail=f"Unknown symptom '{req.symptom}'")

    return inf.update_beliefs(req.symptom, req.response)


# -----------------------------------------------------------
# Get top diseases
# -----------------------------------------------------------

@router.get("/top")
def get_top_diseases(session_id: str, k: int = 5):
    inf = get_inference_service(session_id)
    return inf.get_top_diseases(top_k=k)


# -----------------------------------------------------------
# Get current inference state
# -----------------------------------------------------------

@router.get("/state")
def get_state(session_id: str):
    inf = get_inference_service(session_id)
    return inf.get_state()