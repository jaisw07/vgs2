from src.interactive_diagnoser import InteractiveDiagnoser

if __name__ == "__main__":
    diagnoser = InteractiveDiagnoser(
        dataset_path="data/symptoms_dataset.csv",
        confidence_threshold=0.85,
        max_questions=15
    )
    diagnoser.run()