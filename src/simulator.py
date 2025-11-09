# src/simulator.py
import random
import pandas as pd
import numpy as np
from src.interactive_diagnoser import InteractiveDiagnoser

class DiagnoserSimulator:
    def __init__(self, dataset_path, diagnoser_kwargs=None):
        self.kb = None
        self.dataset_path = dataset_path
        self.diagnoser_kwargs = diagnoser_kwargs or {}
        # we will create diagnoser instances per simulation run to reset state cleanly

    def _make_diagnoser(self):
        return InteractiveDiagnoser(self.dataset_path, **self.diagnoser_kwargs)

    def simulate_on_dataset(self, n_samples=100, random_seed=0):
        df = pd.read_csv(self.dataset_path)
        rng = random.Random(random_seed)
        idxs = rng.sample(list(df.index), min(n_samples, len(df)))
        results = []
        for i in idxs:
            row = df.loc[i]
            # build truth symptom map (use same symptom naming as KB)
            symptom_cols = [c for c in df.columns if c != 'prognosis']
            truth_map = {s: int(row[s]) for s in symptom_cols}

            # create fresh diagnoser
            diagnoser = self._make_diagnoser()
            engine = diagnoser.engine
            entropy = diagnoser.entropy
            csp = diagnoser.csp

            # Simulate interactive loop until threshold or max_questions
            asked = 0
            max_q = diagnoser.max_questions
            while asked < max_q:
                symptom, gain = entropy.select_next_symptom()
                if not symptom:
                    break
                # the simulator answers from truth_map. Unknown not used.
                response = truth_map.get(symptom, -1)
                # attempt CSP check: if invalid, mark as unknown (skip)
                valid, violations = csp.is_valid_state({**diagnoser.user_answers, symptom: response})
                if not valid:
                    # skip this symptom (simulate "unknown" or skip)
                    response = -1
                    # still mark as asked to avoid loops
                    entropy.mark_asked(symptom)
                    asked += 1
                    continue

                # apply
                diagnoser.update_state(symptom, response)
                asked += 1

                top_d = engine.get_top_diseases(5)[0]
                if top_d[1] >= diagnoser.confidence_threshold:
                    break

            topk = [d for d, p in engine.get_top_diseases(5)]
            true_label = row['prognosis']
            results.append({
                'index': i,
                'true': true_label,
                'top1': topk[0],
                'top1_correct': topk[0] == true_label,
                'top3_correct': true_label in topk[:3],
                'questions_asked': asked
            })

        res_df = pd.DataFrame(results)
        summary = {
            'n': len(res_df),
            'top1_acc': res_df['top1_correct'].mean(),
            'top3_acc': res_df['top3_correct'].mean(),
            'avg_questions': res_df['questions_asked'].mean()
        }
        return res_df, summary

if __name__ == "__main__":
    sim = DiagnoserSimulator("data/symptoms_dataset.csv", diagnoser_kwargs={'confidence_threshold':0.85, 'max_questions':20})
    df, summary = sim.simulate_on_dataset(n_samples=4961, random_seed=42)
    print("Summary:", summary)
    df.to_csv("results/simulation_results.csv", index=False)