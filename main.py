import json
from src.interactive_diagnoser import InteractiveDiagnoser

if __name__ == "__main__":
    diagnoser = InteractiveDiagnoser("data/symptoms_dataset.csv", confidence_threshold=0.85, max_questions=15)
    csp = diagnoser.csp

    with open("config/constraints.json") as f:
        rules = json.load(f)

    for cause, effect in rules.get("dependencies", []):
        csp.add_dependency(cause, effect)
    for s1, s2 in rules.get("mutual_exclusions", []):
        csp.add_mutual_exclusion(s1, s2)
    for rule in rules.get("disease_requirements", []):
        disease = rule[0]
        for symptom in rule[1:]:
            csp.add_required_symptom_for_disease(symptom, disease)
    
    # After all add_dependency, add_mutual_exclusion, add_required_symptom_for_disease calls
    ok, issues = csp.check_consistency()
    if not ok:
        print("❌ Inconsistent constraints detected:")
        for issue in issues:
            print("   -", issue)
        print("Please fix the configuration before running the diagnoser.")
        exit(1)
    else:
        print("✅ All constraints are consistent.\n")

    diagnoser.run()