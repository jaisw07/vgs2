# src/csp_module.py
from typing import Dict, List, Tuple

class CSPModule:
    def __init__(self, knowledge_base, verbose: bool = True):
        """
        CSPModule (lightweight)
        - Validates symptom/disease names when adding constraints.
        - Efficiently checks whether a given symptom assignment violates constraints.
        - Does NOT attempt to enumerate all solutions (that would be exponential).
        """
        self.kb = knowledge_base
        self.symptom_list = set(self.kb.get_symptom_list())
        self.disease_list = set(self.kb.get_disease_list())
        self.constraints: List[Tuple[str, str, str]] = []  # (a, relation, b)
        self.verbose = verbose

    # -------------------------
    def _check_symptom_exists(self, name: str):
        if name not in self.symptom_list:
            raise ValueError(f"Symptom '{name}' not found in KB symptom list. Available symptoms: (show with csp.symptom_list)")

    # -------------------------
    def add_dependency(self, cause: str, effect: str):
        """
        cause=1 => effect must be 1
        """
        self._check_symptom_exists(cause)
        self._check_symptom_exists(effect)
        self.constraints.append((cause, "->", effect))
        if self.verbose:
            print(f"Added dependency constraint: {cause} -> {effect}")

    # -------------------------
    def add_mutual_exclusion(self, s1: str, s2: str):
        """
        s1 and s2 cannot both be 1
        """
        self._check_symptom_exists(s1)
        self._check_symptom_exists(s2)
        self.constraints.append((s1, "XOR", s2))
        if self.verbose:
            print(f"Added mutual exclusion constraint: {s1} XOR {s2}")

    # -------------------------
    def add_required_symptom_for_disease(self, symptom: str, disease: str):
        """
        Records that disease requires symptom=1.
        This is stored as a special constraint type (disease, "requires", symptom).
        Note: this is checked by is_valid_state only if the disease name is provided in the symptom_values
        (you can extend to check diseases vs inferred disease probabilities later).
        """
        self._check_symptom_exists(symptom)
        if disease not in self.disease_list:
            raise ValueError(f"Disease '{disease}' not found in KB disease list.")
        self.constraints.append((disease, "requires", symptom))
        if self.verbose:
            print(f"Added disease requirement: {disease} requires {symptom}")

    # -------------------------
    def is_valid_state(self, symptom_values: Dict[str, int]):
        """
        Efficiently check a *given* assignment for violations.
        - symptom_values: dict mapping symptom name -> 0/1 (may include disease names for 'requires' checks)
        Returns: (valid: bool, violations: List[str])
        """
        relevant = {k: v for k, v in symptom_values.items() if (k in self.symptom_list) or (k in self.disease_list)}

        violations: List[str] = []

        # Evaluate constraints
        for c in self.constraints:
            a, rel, b = c

            if rel == "->":
                # a (cause) => b (effect)
                a_val = int(relevant.get(a, 0))
                b_val = int(relevant.get(b, 0))
                if a_val == 1 and b_val == 0:
                    violations.append(f"Violation: {a}=1 requires {b}=1")

            elif rel == "XOR":
                s1_val = int(relevant.get(a, 0))
                s2_val = int(relevant.get(b, 0))
                if s1_val == 1 and s2_val == 1:
                    violations.append(f"Violation: {a} and {b} cannot both be 1")

            elif rel == "requires":
                # (disease, "requires", symptom)
                disease_name = a
                symptom_name = b
                # Only check this if disease was explicitly provided in symptom_values as present (1).
                disease_val = int(relevant.get(disease_name, 0))
                symptom_val = int(relevant.get(symptom_name, 0))
                if disease_val == 1 and symptom_val == 0:
                    violations.append(f"Violation: disease {disease_name} requires symptom {symptom_name}=1")

            else:
                # Unknown relation — skip or raise
                raise RuntimeError(f"Unknown constraint relation '{rel}' in constraint {c}")

        if self.verbose:
            # For debugging: show what was checked
            print("CSP check debug:")
            print("  Provided symptom_values keys:", sorted(list(symptom_values.keys())))
            print("  Relevant used keys:", sorted(list(relevant.keys())))
            if violations:
                print("  Violations found:")
                for v in violations:
                    print("   -", v)
            else:
                print("  No violations.")

        return (len(violations) == 0, violations)

    # -------------------------
    def list_constraints(self):
        return list(self.constraints)   