from typing import Dict, List, Tuple, Any

class CSPService:
    """
    API-friendly version of CSPModule.
    - No print()
    - No terminal interaction
    - Returns JSON-ready dicts
    - Validates incoming requests
    """

    def __init__(self, knowledge_base, verbose: bool = False):
        self.kb = knowledge_base
        self.symptom_list = set(self.kb.get_symptom_list())
        self.disease_list = set(self.kb.get_disease_list())
        self.constraints: List[Tuple[str, str, str]] = []
        self.verbose = verbose

    # -------------------------
    def _check_symptom_exists(self, name: str):
        if name not in self.symptom_list:
            raise ValueError(f"Symptom '{name}' not found in Knowledge Base.")

    def _check_disease_exists(self, name: str):
        if name not in self.disease_list:
            raise ValueError(f"Disease '{name}' not found in Knowledge Base.")

    # -------------------------
    def add_dependency(self, cause: str, effect: str) -> Dict[str, Any]:
        self._check_symptom_exists(cause)
        self._check_symptom_exists(effect)

        self.constraints.append((cause, "->", effect))
        return {
            "status": "success",
            "added": {"cause": cause, "effect": effect, "type": "dependency"},
            "message": f"Added dependency: {cause} → {effect}"
        }

    # -------------------------
    def add_mutual_exclusion(self, s1: str, s2: str) -> Dict[str, Any]:
        self._check_symptom_exists(s1)
        self._check_symptom_exists(s2)

        self.constraints.append((s1, "XOR", s2))
        return {
            "status": "success",
            "added": {"s1": s1, "s2": s2, "type": "mutual_exclusion"},
            "message": f"Added mutual exclusion: {s1} XOR {s2}"
        }

    # -------------------------
    def add_required_symptom_for_disease(self, symptom: str, disease: str) -> Dict[str, Any]:
        self._check_symptom_exists(symptom)
        self._check_disease_exists(disease)

        self.constraints.append((disease, "requires", symptom))
        return {
            "status": "success",
            "added": {"disease": disease, "symptom": symptom, "type": "requires"},
            "message": f"Added disease requirement: {disease} requires {symptom}"
        }

    # -------------------------
    def is_valid_state(self, symptom_values: Dict[str, int]) -> Dict[str, Any]:
        """
        API version: returns dict instead of (bool, violations)
        """

        relevant = {k: v for k, v in symptom_values.items()
                    if (k in self.symptom_list) or (k in self.disease_list)}

        violations: List[str] = []

        # Evaluate constraints
        for (a, rel, b) in self.constraints:

            if rel == "->":
                if symptom_values.get(a, 0) == 1 and symptom_values.get(b, 0) == 0:
                    violations.append(f"{a}=1 requires {b}=1")

            elif rel == "XOR":
                if symptom_values.get(a, 0) == 1 and symptom_values.get(b, 0) == 1:
                    violations.append(f"{a} and {b} cannot both be 1")

            elif rel == "requires":
                disease = a
                symptom = b
                if symptom_values.get(disease, 0) == 1 and symptom_values.get(symptom, 0) == 0:
                    violations.append(f"Disease {disease} requires symptom {symptom}=1")

        return {
            "valid": len(violations) == 0,
            "violations": violations
        }

    # -------------------------
    def list_constraints(self) -> Dict[str, Any]:
        return {
            "constraints": [
                {"a": a, "relation": rel, "b": b}
                for (a, rel, b) in self.constraints
            ]
        }

    # -------------------------
    def check_consistency(self) -> Dict[str, Any]:
        """
        API version of your long consistency checking method.
        Now returns JSON-ready dictionaries.
        """

        issues = []
        warnings = []

        deps = [(a, b) for (a, rel, b) in self.constraints if rel == "->"]
        xors = [(a, b) for (a, rel, b) in self.constraints if rel == "XOR"]
        requires = [(a, b) for (a, rel, b) in self.constraints if rel == "requires"]

        # 1. Undefined entities
        for a, b in deps + xors:
            for s in (a, b):
                if s not in self.symptom_list and s not in self.disease_list:
                    issues.append(f"Undefined entity '{s}' in constraint ({a}, {b})")

        for d, s in requires:
            if d not in self.disease_list:
                issues.append(f"Disease '{d}' missing in KB")
            if s not in self.symptom_list:
                issues.append(f"Symptom '{s}' missing in KB")

        # 2. Direct contradictions
        for (a, b) in deps:
            if (a, b) in xors or (b, a) in xors:
                issues.append(f"Dependency {a}->{b} contradicts XOR {a} XOR {b}")

        for (a, b) in deps:
            if (b, a) in deps:
                issues.append(f"Circular dependency between {a} and {b}")

        # 3. Build dependency graph
        dep_graph = {}
        for (a, b) in deps:
            dep_graph.setdefault(a, set()).add(b)

        changed = True
        while changed:
            changed = False
            for a in dep_graph:
                new = set()
                for b in dep_graph[a]:
                    new |= dep_graph.get(b, set())
                if not new.issubset(dep_graph[a]):
                    dep_graph[a] |= new
                    changed = True

        # 4. Indirect XOR contradictions
        for (a, b) in xors:
            if b in dep_graph.get(a, set()):
                issues.append(f"{a} leads to {b} but also XORs with it")
            if a in dep_graph.get(b, set()):
                issues.append(f"{b} leads to {a} but also XORs with it")

        # 5. Redundant constraints
        seen = set()
        for c in self.constraints:
            if c in seen:
                warnings.append(f"Duplicate constraint: {c}")
            seen.add(c)

        return {
            "consistent": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "dependency_graph": {
                node: sorted(list(children))
                for node, children in dep_graph.items()
            }
        }