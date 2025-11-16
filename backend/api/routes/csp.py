from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uuid

from backend.core.csp_service import CSPService
from backend.core.session_manager import SessionManager
from backend.core.knowledge_base_loader import get_knowledge_base

# Create router
router = APIRouter(prefix="/csp", tags=["CSP"])

# Session manager (simple version)
session_manager = SessionManager()

# -------------------------------------------
# Pydantic models
# -------------------------------------------

class CreateSessionResponse(BaseModel):
    session_id: str

class AddDependencyRequest(BaseModel):
    session_id: str
    cause: str
    effect: str

class MutualExclusionRequest(BaseModel):
    session_id: str
    s1: str
    s2: str

class DiseaseRequirementRequest(BaseModel):
    session_id: str
    disease: str
    symptom: str

class StateValidationRequest(BaseModel):
    session_id: str
    symptom_values: Dict[str, int]

# -------------------------------------------
# Helpers
# -------------------------------------------

def get_csp(session_id: str) -> CSPService:
    csp = session_manager.get(session_id)
    if csp is None:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    return csp

# -------------------------------------------
# Routes
# -------------------------------------------

@router.post("/create_session", response_model=CreateSessionResponse)
def create_session():
    """
    Create a new CSP session with its own constraint state.
    """
    kb = get_knowledge_base()  # your KB loader (cached)
    session_id = str(uuid.uuid4())

    csp = CSPService(kb, verbose=False)
    session_manager.set(session_id, csp)

    return CreateSessionResponse(session_id=session_id)


# ------------------- Add constraints -------------------

@router.post("/add_dependency")
def add_dependency(req: AddDependencyRequest):
    csp = get_csp(req.session_id)
    try:
        return csp.add_dependency(req.cause, req.effect)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/add_mutual_exclusion")
def add_mutual_exclusion(req: MutualExclusionRequest):
    csp = get_csp(req.session_id)
    try:
        return csp.add_mutual_exclusion(req.s1, req.s2)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/add_disease_requirement")
def add_disease_requirement(req: DiseaseRequirementRequest):
    csp = get_csp(req.session_id)
    try:
        return csp.add_required_symptom_for_disease(req.symptom, req.disease)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ------------------- Listing -------------------

@router.get("/list_constraints")
def list_constraints(session_id: str):
    csp = get_csp(session_id)
    return csp.list_constraints()


# ------------------- State Validation -------------------

@router.post("/validate_state")
def validate_state(req: StateValidationRequest):
    csp = get_csp(req.session_id)
    return csp.is_valid_state(req.symptom_values)


# ------------------- Consistency Check -------------------

@router.get("/check_consistency")
def check_consistency(session_id: str):
    csp = get_csp(session_id)
    return csp.check_consistency()