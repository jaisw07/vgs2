# main.py
import uuid
import logging
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.interactive_diagnoser import InteractiveDiagnoser
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Interactive Diagnostic API",
    description="An API to interact with the diagnostic engine.",
    version="1.0.0",
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("diagnostic_api")

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
else:
    logger.warning("GEMINI_API_KEY not found in environment variables")
    model = None

# Load symptoms list
try:
    with open("frontend/src/context/symptoms.json", "r") as f:
        symptoms_data = json.load(f)
        SYMPTOMS_LIST = symptoms_data["symptoms"]
        FORMATTED_SYMPTOMS = symptoms_data["formatted_symptoms"]
except Exception as e:
    logger.error(f"Failed to load symptoms.json: {e}")
    SYMPTOMS_LIST = []
    FORMATTED_SYMPTOMS = []

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

class SuggestRequest(BaseModel):
    text: str
    cursor_position: int

class SuggestResponse(BaseModel):
    suggestions: list[dict]

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

@app.post("/suggest", response_model=SuggestResponse)
async def suggest_symptoms(request: SuggestRequest):
    """
    Provides intelligent symptom suggestions using Gemini LLM.
    Only suggests when there's meaningful context (3+ characters).
    """
    text = request.text.strip()
    
    # Don't suggest for very short input
    if len(text) < 3:
        return {"suggestions": []}
    
    # Extract the current word being typed
    words = text.split()
    if not words:
        return {"suggestions": []}
    
    current_word = words[-1].lower()
    
    # Don't suggest for single characters or very short words
    if len(current_word) < 3:
        return {"suggestions": []}
    
    # Simple fuzzy matching first
    simple_matches = []
    for symptom_obj in FORMATTED_SYMPTOMS:
        symptom_value = symptom_obj["value"]
        symptom_label = symptom_obj["label"].lower()
        
        # Check if current word matches beginning of symptom
        if symptom_label.startswith(current_word) or symptom_value.startswith(current_word):
            simple_matches.append({
                "value": symptom_value,
                "label": symptom_obj["label"],
                "confidence": "high",
                "match_type": "prefix"
            })
        # Check if current word is contained in symptom
        elif current_word in symptom_label or current_word in symptom_value:
            simple_matches.append({
                "value": symptom_value,
                "label": symptom_obj["label"],
                "confidence": "medium",
                "match_type": "contains"
            })
    
    # If we have simple matches, return them (faster)
    if simple_matches:
        # Sort by confidence and limit to 5
        simple_matches.sort(key=lambda x: (x["confidence"] != "high", x["label"]))
        return {"suggestions": simple_matches[:5]}
    
    # If no simple matches and Gemini is available, use LLM for intelligent suggestions
    if model and len(text) >= 5:
        try:
            prompt = f"""You are a medical symptom matcher. Given a partial or misspelled symptom description, match it to the closest actual medical symptoms from this list.

User input: "{text}"

Available symptoms:
{', '.join([s['label'] for s in FORMATTED_SYMPTOMS[:50]])}... (and more)

Rules:
1. Only suggest if the input seems to be describing a medical symptom
2. Match based on meaning, not just spelling
3. Return ONLY the top 3 most relevant symptom names from the list
4. Return as a comma-separated list
5. If no good match, return "NONE"

Response (comma-separated symptom names only):"""

            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
            if result_text != "NONE" and result_text:
                suggested_labels = [s.strip() for s in result_text.split(",")]
                llm_matches = []
                
                for label in suggested_labels[:3]:
                    # Find matching symptom from our list
                    for symptom_obj in FORMATTED_SYMPTOMS:
                        if symptom_obj["label"].lower() == label.lower():
                            llm_matches.append({
                                "value": symptom_obj["value"],
                                "label": symptom_obj["label"],
                                "confidence": "ai_suggested",
                                "match_type": "llm"
                            })
                            break
                
                if llm_matches:
                    return {"suggestions": llm_matches}
        
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
    
    return {"suggestions": []}
