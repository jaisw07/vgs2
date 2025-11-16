import re
import spacy
from typing import Dict, List, Any


class NLPService:
    """
    FastAPI-friendly wrapper around SymptomNLPParser.
    - No prints
    - Returns JSON-safe response dicts
    - Ready for interactive diagnosis workflow
    """

    def __init__(self, symptom_list: List[str], use_lemmas: bool = True, verbose: bool = False):
        self.symptom_list = symptom_list
        self.use_lemmas = use_lemmas
        self.verbose = verbose

        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            raise RuntimeError(
                "SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm"
            )

        # A simpler keyword lookup (underscores → spaces)
        self.symptom_keywords = {
            s: s.replace("_", " ").lower() for s in self.symptom_list
        }

    # ------------------------------------------------------
    def parse_text(self, text: str) -> Dict[str, Any]:
        """
        Parse user free text -> symptom presence dict.
        Returns:
            {
                "symptom_states": {symptom: 0/1},
                "positive_symptoms": [...],
                "raw_text": ...
            }
        """
        text = text.lower().strip()
        doc = self.nlp(text)
        tokens = [t.lemma_ if self.use_lemmas else t.text for t in doc]

        result = {s: 0 for s in self.symptom_list}

        for symptom, phrase in self.symptom_keywords.items():

            # Positive signal
            if re.search(rf"\b{re.escape(phrase)}\b", text):
                result[symptom] = 1
                continue

            # Negation patterns
            neg_patterns = [
                rf"no {phrase}",
                rf"without {phrase}",
                rf"not {phrase}",
                rf"haven't (had )?{phrase}",
            ]
            if any(re.search(p, text) for p in neg_patterns):
                result[symptom] = 0

        positives = [s for s, v in result.items() if v == 1]

        response = {
            "raw_text": text,
            "symptom_states": result,
            "positive_symptoms": positives
        }

        return response

    # ------------------------------------------------------
    def update_symptom_list(self, symptoms: List[str]) -> Dict[str, Any]:
        """
        Update symptom vocabulary dynamically.
        """
        self.symptom_list = symptoms
        self.symptom_keywords = {
            s: s.replace("_", " ").lower() for s in self.symptom_list
        }

        return {
            "status": "updated",
            "n_symptoms": len(self.symptom_list)
        }

    # ------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """
        Debug state: returns internal lexical items and settings.
        """
        return {
            "symptoms": self.symptom_list,
            "keywords": self.symptom_keywords,
            "use_lemmas": self.use_lemmas,
            "verbose": self.verbose
        }