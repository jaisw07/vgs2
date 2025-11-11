# src/nlp_parser.py
import re
import spacy
from typing import Dict, List

class SymptomNLPParser:
    """
    Parses free-text medical symptom descriptions into binary symptom features.
    """

    def __init__(self, symptom_list: List[str], use_lemmas=True, verbose=True):
        """
        Args:
            symptom_list: list of known symptom names (from KnowledgeBase)
            use_lemmas: whether to lemmatize for matching
        """
        self.symptom_list = symptom_list
        self.use_lemmas = use_lemmas
        self.verbose = verbose

        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            raise RuntimeError("SpaCy model not found. Run: python -m spacy download en_core_web_sm")

        # Build a lookup for easier matching
        self.symptom_keywords = {s: s.replace("_", " ").lower() for s in self.symptom_list}

    # -------------------------
    def parse_text(self, text: str) -> Dict[str, int]:
        """
        Convert free text into binary symptom states.
        Returns dict {symptom: 0/1}.
        """
        text = text.lower().strip()
        doc = self.nlp(text)
        tokens = [t.lemma_ if self.use_lemmas else t.text for t in doc]

        result = {s: 0 for s in self.symptom_list}

        for symptom, phrase in self.symptom_keywords.items():
            # Positive detection
            if re.search(rf"\b{re.escape(phrase)}\b", text):
                result[symptom] = 1
                continue

            # Negative patterns (simple negation detection)
            neg_patterns = [
                rf"no {phrase}",
                rf"without {phrase}",
                rf"not {phrase}",
                rf"haven't (had )?{phrase}",
            ]
            if any(re.search(p, text) for p in neg_patterns):
                result[symptom] = 0

        if self.verbose:
            found = [s for s, v in result.items() if v == 1]
            print(f"🧠 NLP detected positive symptoms: {found}")

        return result