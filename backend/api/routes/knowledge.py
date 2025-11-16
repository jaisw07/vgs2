from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid

from backend.core.session_manager import SessionManager
from backend.core.knowledge_base_service import KnowledgeBaseService

router = APIRouter(prefix="/knowledge", tags=["KnowledgeBase"])

session_manager = SessionManager()

# -----------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------

class CreateKBRequest(BaseModel):
    csv_path: str
    target_col: str = "prognosis"
    smoothing: float = 1.0

class CreateKBResponse(BaseModel):
    session_id: str
    status: str

class GenericSession(BaseModel):
    session_id: str

class QueryDiseaseRequest(BaseModel):
    session_id: str
    disease: str

class QueryConditionalRequest(BaseModel):
    session_id: str
    disease: str
    symptom: str


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def get_kb(session_id: str) -> KnowledgeBaseService:
    session = session_manager.get(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Invalid session_id")

    if "kb" not in session:
        raise HTTPException(status_code=500, detail="KnowledgeBase not initialized")

    return session["kb"]


# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------

@router.post("/create", response_model=CreateKBResponse)
def create_kb(req: CreateKBRequest):
    """
    Create a new KnowledgeBase session.
    Loads dataset + stores KB instance in session manager.
    """
    session_id = str(uuid.uuid4())

    kb = KnowledgeBaseService(
        csv_path=req.csv_path,
        target_col=req.target_col,
        smoothing=req.smoothing
    )

    # Store the KB instance in session
    session_manager.set(session_id, {"kb": kb})

    return CreateKBResponse(session_id=session_id, status="created")


# -----------------------------------------------------------
# Load Dataset
# -----------------------------------------------------------

@router.post("/load")
def load_dataset(req: GenericSession):
    kb = get_kb(req.session_id)
    return kb.load_dataset()


# -----------------------------------------------------------
# Compute Probabilities
# -----------------------------------------------------------

@router.post("/compute")
def compute_probabilities(req: GenericSession):
    kb = get_kb(req.session_id)
    return kb.compute_probabilities()


# -----------------------------------------------------------
# Get symptom list
# -----------------------------------------------------------

@router.get("/symptoms")
def get_symptoms(session_id: str):
    kb = get_kb(session_id)
    return {"symptoms": kb.get_symptom_list()}


# -----------------------------------------------------------
# Get disease list
# -----------------------------------------------------------

@router.get("/diseases")
def get_diseases(session_id: str):
    kb = get_kb(session_id)
    return {"diseases": kb.get_disease_list()}


# -----------------------------------------------------------
# Get P(Disease)
# -----------------------------------------------------------

@router.post("/prob_disease")
def prob_disease(req: QueryDiseaseRequest):
    kb = get_kb(req.session_id)
    return {
        "disease": req.disease,
        "probability": kb.get_P_disease(req.disease)
    }


# -----------------------------------------------------------
# Get P(Symptom|Disease)
# -----------------------------------------------------------

@router.post("/prob_symptom_given_disease")
def prob_symptom_given_disease(req: QueryConditionalRequest):
    kb = get_kb(req.session_id)
    return {
        "disease": req.disease,
        "symptom": req.symptom,
        "probability": kb.get_P_symptom_given_disease(req.disease, req.symptom)
    }


# -----------------------------------------------------------
# Export full matrix
# -----------------------------------------------------------

@router.get("/matrix")
def export_matrix(session_id: str):
    kb = get_kb(session_id)
    return kb.export_matrix()


# -----------------------------------------------------------
# Debug: Return entire KB state
# -----------------------------------------------------------

@router.get("/state")
def kb_state(session_id: str):
    kb = get_kb(session_id)
    return kb.get_state()