# main.py
import uuid
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.interactive_diagnoser import InteractiveDiagnoser
from fastapi.middleware.cors import CORSMiddleware

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Interactive Diagnostic API",
    description="An API to interact with the diagnostic engine.",
    version="1.0.0",
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("diagnostic_api")

# In-memory storage for active sessions.
# In a production environment, you would use a more persistent store like Redis.
sessions = {}

# --- Pydantic Models for API Payloads ---
class StartResponse(BaseModel):
    session_id: str
    question: str
    ig: float

class DescribeRequest(BaseModel):
    session_id: str
    text: str

class DescribeResponse(BaseModel):
    parsed_symptoms: dict
    top_diseases: list
    question: str
    ig: float

class AnswerRequest(BaseModel):
    session_id: str
    symptom: str
    answer: int  # 1 for yes, 0 for no, -1 for unknown

class AnswerResponse(BaseModel):
    question: str | None
    ig: float | None
    top_diseases: list
    is_finished: bool
    finish_reason: str | None

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.post("/start", response_model=StartResponse)
async def start_session():
    """
    Starts a new diagnostic session.
    """
    session_id = str(uuid.uuid4())
    logger.debug(f"Starting new session with ID: {session_id}")

    try:
        # Create a new diagnoser instance for this session
        diagnoser = InteractiveDiagnoser(dataset_path="data/symptoms_dataset.csv")
        sessions[session_id] = diagnoser
        logger.debug("InteractiveDiagnoser instance created successfully.")

        # Get the first question
        symptom, gain = diagnoser.entropy.select_next_symptom()
        logger.debug(f"First symptom selected: {symptom}, Information Gain: {gain}")

        if not symptom:
            raise ValueError("Could not select an initial symptom.")

        return {
            "session_id": session_id,
            "question": f"Do you have {symptom.replace('_', ' ')}?",
            "ig": gain,
        }
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail="Failed to start session due to server error.")

@app.post("/describe", response_model=DescribeResponse)
async def describe_symptoms(request: DescribeRequest):
    """
    Processes a free-text description of symptoms.
    """
    diagnoser = sessions.get(request.session_id)
    if not diagnoser:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Parse the text
    parsed = diagnoser.parser.parse_text(request.text)
    positive_symptoms = {s: v for s, v in parsed.items() if v == 1}
    
    # Update beliefs for each parsed symptom
    for symptom, value in positive_symptoms.items():
        diagnoser.update_state(symptom, value)

    top_diseases = diagnoser.engine.get_top_diseases(5)
    
    # Get the next question
    symptom, gain = diagnoser.entropy.select_next_symptom()

    return {
        "parsed_symptoms": positive_symptoms,
        "top_diseases": top_diseases,
        "question": f"Do you have {symptom.replace('_', ' ')}?" if symptom else None,
        "ig": gain if symptom else 0.0,
    }

@app.post("/answer", response_model=AnswerResponse)
async def process_answer(request: AnswerRequest):
    """
    Processes a user's answer to a symptom question.
    """
    diagnoser = sessions.get(request.session_id)
    if not diagnoser:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Update the state with the user's answer
    updated = diagnoser.update_state(request.symptom, request.answer)
    if not updated:
        # This could happen if a CSP constraint is violated
        raise HTTPException(status_code=400, detail="Invalid answer due to a constraint violation.")

    # Check for stopping conditions
    top_disease, top_prob = diagnoser.engine.get_top_diseases(1)[0]
    if top_prob >= diagnoser.confidence_threshold:
        return {
            "question": None,
            "ig": None,
            "top_diseases": diagnoser.engine.get_top_diseases(5),
            "is_finished": True,
            "finish_reason": f"Diagnosis complete. Confidence threshold of {diagnoser.confidence_threshold*100:.0f}% reached for {top_disease}.",
        }

    if len(diagnoser.user_answers) >= diagnoser.max_questions:
        return {
            "question": None,
            "ig": None,
            "top_diseases": diagnoser.engine.get_top_diseases(5),
            "is_finished": True,
            "finish_reason": f"Diagnosis stopped. Maximum number of questions ({diagnoser.max_questions}) reached.",
        }

    # Otherwise, select the next best question
    symptom, gain = diagnoser.entropy.select_next_symptom()
    
    if not symptom:
        return {
            "question": None,
            "ig": None,
            "top_diseases": diagnoser.engine.get_top_diseases(5),
            "is_finished": True,
            "finish_reason": "Diagnosis complete. No more informative questions to ask.",
        }

    return {
        "question": f"Do you have {symptom.replace('_', ' ')}?",
        "ig": gain,
        "top_diseases": diagnoser.engine.get_top_diseases(5),
        "is_finished": False,
        "finish_reason": None,
    }
