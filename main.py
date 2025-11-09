from src.knowledge_base import KnowledgeBase

csv_path = "data/symptoms_dataset.csv"  # path to your dataset

kb = KnowledgeBase(csv_path)
kb.load_dataset()
kb.compute_probabilities()

# Display sample probabilities
print("\n--- Sample P(Symptom|Disease) ---")
for disease in kb.get_disease_list()[:3]:
    print(f"\n{disease}:")
    for symptom in kb.get_symptom_list()[:5]:
        print(f"  {symptom}: {kb.get_P_symptom_given_disease(disease, symptom):.3f}")

# Export matrix to CSV for inspection
matrix = kb.export_matrix()
matrix.to_csv("results/P_symptom_given_disease.csv")