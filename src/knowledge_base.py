import pandas as pd
import numpy as np
from collections import defaultdict

class KnowledgeBase:
    def __init__(self, csv_path: str, target_col: str = "prognosis", smoothing: float = 1.0):
        """
        Builds a probabilistic knowledge base from a symptom–disease dataset.
        """
        self.csv_path = csv_path
        self.target_col = target_col
        self.smoothing = smoothing
        
        # Internal structures
        self.df = None
        self.symptoms = []
        self.diseases = []
        self.P_symptom_given_disease = {}
        self.P_disease = {}
    
    # -----------------------------
    # Load dataset and preprocess
    # -----------------------------
    def load_dataset(self):
        self.df = pd.read_csv(self.csv_path)
        self.df.fillna(0, inplace=True)
        
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")
        
        # Identify symptoms and diseases
        self.symptoms = [col for col in self.df.columns if col != self.target_col]
        self.diseases = sorted(self.df[self.target_col].unique())
        
        print(f"✅ Loaded dataset with {len(self.df)} samples, {len(self.symptoms)} symptoms, {len(self.diseases)} diseases.")

    # -----------------------------
    # Compute probabilities
    # -----------------------------
    def compute_probabilities(self):
        """
        Computes:
        P(Disease)
        P(Symptom | Disease)
        Using Laplace smoothing.
        """
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")
        
        # P(Disease)
        disease_counts = self.df[self.target_col].value_counts()
        total = len(self.df)
        self.P_disease = {d: disease_counts[d] / total for d in self.diseases}
        
        # P(Symptom|Disease)
        prob_matrix = defaultdict(dict)
        
        for disease in self.diseases:
            subset = self.df[self.df[self.target_col] == disease]
            for symptom in self.symptoms:
                count_yes = subset[symptom].sum()
                prob_matrix[disease][symptom] = (count_yes + self.smoothing) / (len(subset) + 2 * self.smoothing)
        
        self.P_symptom_given_disease = dict(prob_matrix)
        print("✅ Computed conditional probabilities P(Symptom|Disease).")

    # -----------------------------
    # Utility: Accessors
    # -----------------------------
    def get_symptom_list(self):
        return self.symptoms
    
    def get_disease_list(self):
        return self.diseases
    
    def get_P_symptom_given_disease(self, disease, symptom):
        return self.P_symptom_given_disease.get(disease, {}).get(symptom, 0.5)
    
    def get_P_disease(self, disease):
        return self.P_disease.get(disease, 1.0 / len(self.diseases))

    # -----------------------------
    # Optional: Export for visualization
    # -----------------------------
    def export_matrix(self):
        """
        Returns a DataFrame representation of P(Symptom|Disease).
        """
        df_export = pd.DataFrame(self.P_symptom_given_disease).T
        return df_export

    # visualizations
    def visualize_heatmap(self, figsize=(14, 10), save_path=None):
        import matplotlib.pyplot as plt
        import seaborn as sns

        matrix = pd.DataFrame(self.P_symptom_given_disease).T

        plt.figure(figsize=figsize)
        sns.heatmap(matrix, cmap="viridis", linewidths=0.1)
        plt.xlabel("Symptoms")
        plt.ylabel("Diseases")
        plt.title("P(Symptom | Disease) Heatmap")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Heatmap saved to {save_path}")
        else:
            plt.show()

    def visualize_bipartite_graph(self, threshold=0.5, save_path=None):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.Graph()

        # Add nodes
        G.add_nodes_from(self.diseases, bipartite=0)
        G.add_nodes_from(self.symptoms, bipartite=1)

        # Add edges with weights
        for disease in self.diseases:
            for symptom in self.symptoms:
                p = self.P_symptom_given_disease[disease][symptom]
                if p >= threshold:
                    G.add_edge(disease, symptom, weight=p)

        # Layout bipartite
        pos = {}
        d_step = 1 / len(self.diseases)
        s_step = 1 / len(self.symptoms)

        for i, d in enumerate(self.diseases):
            pos[d] = (0, i * d_step)

        for i, s in enumerate(self.symptoms):
            pos[s] = (1, i * s_step)

        plt.figure(figsize=(16, 12))
        nx.draw(G, pos, with_labels=True, node_size=600, font_size=8)
        plt.title(f"Bipartite Disease–Symptom Graph (threshold ≥ {threshold})")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"🕸️ Bipartite graph saved to {save_path}")
        else:
            plt.show()

    def visualize_disease_similarity(self, threshold=0.4, save_path=None):
        import networkx as nx
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # Build matrix
        mat = pd.DataFrame(self.P_symptom_given_disease).T
        sim = cosine_similarity(mat.values)

        # Create graph
        G = nx.Graph()
        for i, d1 in enumerate(self.diseases):
            for j, d2 in enumerate(self.diseases):
                if i < j and sim[i][j] >= threshold:
                    G.add_edge(d1, d2, weight=sim[i][j])

        # --- Cluster Detection ---
        # try:
        #     # If python-louvain is installed (community_louvain)
        #     import community as community_louvain
        #     communities = community_louvain.best_partition(G)
        # except ImportError:
            # Fallback: greedy modularity communities
        comms = nx.algorithms.community.greedy_modularity_communities(G)
        communities = {}
        for idx, cluster in enumerate(comms):
            for node in cluster:
                communities[node] = idx

        # Colors by community id
        unique_ids = sorted(set(communities.values()))
        color_map = plt.cm.get_cmap("tab10", len(unique_ids))
        node_colors = [color_map(communities[n]) for n in G.nodes()]

        # Layout
        pos = nx.spring_layout(G, seed=42, k=0.5)

        # Plot graph
        plt.figure(figsize=(14, 12))
        nx.draw_networkx(
            G, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=1400,
            font_size=9,
            width=[G[u][v]['weight'] * 2 for u, v in G.edges()],
            edge_color="#555",
        )

        plt.title("Disease Similarity Graph (Color-Coded by Clusters)")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"🌈 Disease similarity graph saved to {save_path}")
        else:
            plt.show()

    def save_disease_similarity(self, save_path="results/disease_similarity.csv"):
        from sklearn.metrics.pairwise import cosine_similarity
        import pandas as pd
        import os

        # Create results directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        mat = pd.DataFrame(self.P_symptom_given_disease).T
        sim = cosine_similarity(mat.values)

        # Build DataFrame with diseases as both rows and columns
        df_sim = pd.DataFrame(sim, index=self.diseases, columns=self.diseases)

        # Save
        df_sim.to_csv(save_path)
        print(f"📄 Disease similarity matrix saved to: {save_path}")

        # Optional pretty print
        print("\n=== Pairwise Similarities (i < j) ===")
        for i, d1 in enumerate(self.diseases):
            for j, d2 in enumerate(self.diseases):
                if i < j:
                    print(f"{d1:20s} ↔ {d2:20s} : {sim[i][j]:.4f}")


    def visualize_symptom_cooccurrence(self, threshold=0.5, save_path=None):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.Graph()

        corr = self.df[self.symptoms].corr()

        for s1 in self.symptoms:
            for s2 in self.symptoms:
                if s1 < s2 and corr.loc[s1, s2] >= threshold:
                    G.add_edge(s1, s2, weight=corr.loc[s1, s2])

        plt.figure(figsize=(14, 12))
        nx.draw_networkx(G, with_labels=True, node_size=600, font_size=7)
        plt.title("Symptom Co-Occurrence Graph")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"🤝 Symptom co-occurrence graph saved to {save_path}")
        else:
            plt.show()

    def visualize(self):
        print("Choose visualization:")
        print("1. Heatmap")
        print("2. Bipartite Graph")
        print("3. Disease Similarity Graph")
        print("4. Symptom Co-occurrence Graph")

        choice = input("Enter choice: ")

        if choice == "1":
            self.visualize_heatmap()
        elif choice == "2":
            self.visualize_bipartite_graph()
        elif choice == "3":
            self.visualize_disease_similarity()
        elif choice == "4":
            self.visualize_symptom_cooccurrence()
        else:
            print("Invalid choice.")
