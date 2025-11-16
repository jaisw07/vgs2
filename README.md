<div align="center">

```
  __      __   _______   _________  ____  ____
 /  \    /  \ /  _____| /   _____/ /   \/   /
 \   \/\/   /|  |  __  |   __  \  |   \  /   |
  \        / |  | |_ | |  |__|  | |   |\/|   |
   \__/\  /  \______|  \_______/  \__/  \__/
        \/
```
# VGS-2: AI-Powered Interactive Diagnoser

**An intelligent diagnostic system that uses Bayesian inference and information theory to conduct a conversation and identify potential diseases from symptoms.**

</div>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-009688.svg?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-13.x-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.x-09a3d5.svg?style=for-the-badge&logo=spacy)](https://spacy.io/)

---

## ✨ Overview

Welcome to VGS-2, a sophisticated medical diagnostic assistant. This project isn't just another symptom checker; it's a dynamic conversational AI that intelligently guides the user through a diagnostic process. It uses a probabilistic model to reason about symptoms and an entropy-based engine to decide the most informative question to ask next, ensuring the diagnosis is as quick and accurate as possible.

The entire diagnostic logic is wrapped in a clean **FastAPI** backend, making it ready for integration with any modern web or mobile frontend.

## 🚀 Key Features

- **🧠 Bayesian Inference Engine**: At its core, the system uses Bayes' theorem to update its beliefs about potential diseases as new symptom information is provided.
- **🔍 Intelligent Questioning**: Powered by an **Entropy Engine**, the system doesn't ask random questions. It calculates the "Information Gain" for each potential question and picks the one that will reduce the most uncertainty.
- **🗣️ Natural Language Understanding**: Users can describe their symptoms in plain English, and our **spaCy-powered NLP Parser** will understand and structure the information.
- **🧩 Constraint-Based Logic**: A **Constraint Satisfaction Problem (CSP)** module ensures that the system considers logical rules, such as dependencies or mutual exclusions between symptoms.
- **🌐 API-Ready**: A robust **FastAPI** server exposes the diagnostic logic through a clean, well-documented API, ready for any UI to consume.
- **🧪 Simulator Included**: A built-in simulation module allows for testing the diagnostic engine's accuracy and efficiency against the entire dataset.

## 🏗️ System Architecture

The project is designed with a modular and decoupled architecture, making it easy to maintain and extend.

```mermaid
graph TD
    subgraph Frontend
        A[Browser UI<br>(Next.js/React)]
    end

    subgraph Backend
        B[FastAPI Server<br>(main.py)]
    end

    subgraph Core Logic
        C[InteractiveDiagnoser]
        D[Inference Engine]
        E[Entropy Engine]
        F[NLP Parser]
        G[CSP Module]
    end

    subgraph Data Layer
        H[Knowledge Base]
        I[symptoms_dataset.csv]
    end

    A -- HTTP Requests --> B
    B -- Manages Session --> C
    C -- Integrates --> D
    C -- Integrates --> E
    C -- Integrates --> F
    C -- Integrates --> G
    D -- Uses --> H
    E -- Uses --> D
    F -- Uses --> H
    H -- Learns from --> I
```

## 🏁 Getting Started

Follow these steps to get the project up and running on your local machine.

### 1. Prerequisites

- **Conda**: You must have `conda` installed to manage the environment. You can get it by installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd vgs2
```

### 3. Set Up the Conda Environment

Create the conda environment from the `environment.yml` file. This will install all the necessary Python dependencies.

```bash
conda env create -f environment.yml
```

Once the installation is complete, activate the environment:

```bash
conda activate vgs2-env
```

### 4. Download the NLP Model

The NLP parser requires a `spaCy` language model. Download it with this command:

```bash
python -m spacy download en_core_web_sm
```

### 5. Run the Backend Server

Now, you can start the FastAPI server.

```bash
uvicorn main:app --reload
```

The server will be running at `http://127.0.0.1:8000`.

### 6. Explore the API

You can now access the interactive API documentation (Swagger UI) in your browser at:
**[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

From here, you can test the API endpoints directly.

## 🔌 API Usage

The API is designed to be used in a conversational manner. For full details on the workflow and endpoint models, please refer to the comprehensive documentation file:

**[➡️ API_DOCUMENTATION.md](API_DOCUMENTATION.md)**

## 📂 Project Structure

```
vgs2/
├── API_DOCUMENTATION.md      # Detailed API documentation for UI developers/AI.
├── main.py                   # FastAPI application entry point.
├── environment.yml           # Conda environment definition.
├── .gitignore                # Files and folders to be ignored by Git.
├── README.md                 # This file.
├── data/
│   └── symptoms_dataset.csv  # The core dataset of symptoms and diseases.
├── frontend/                 # (Placeholder for Next.js/React frontend)
│   └── ...
├── src/                      # Core Python source code for the diagnostic engine.
│   ├── knowledge_base.py     # Loads data and computes probabilities.
│   ├── inference_engine.py   # Performs Bayesian inference.
│   ├── entropy_engine.py     # Selects the best questions to ask.
│   ├── nlp_parser.py         # Handles free-text symptom parsing.
│   ├── csp_module.py         # Manages logical constraints.
│   └── ...
├── results/                  # (Ignored by Git) Output from simulations and logs.
└── visualizations/           # (Ignored by Git) Generated graphs and heatmaps.
```

## 💡 Future Enhancements

This project has a solid foundation, but there are many ways it could be extended:

- **Persistent Session Storage**: Use Redis or a database to store session data, allowing users to resume a diagnosis.
- **User Accounts**: Implement user authentication to save diagnostic history.
- **Advanced NLP**: Use more advanced NLP techniques to handle more complex sentence structures and medical terminologies.
- **Frontend UI**: Build a complete, responsive, and user-friendly web interface using the Next.js framework.
- **Model Retraining**: Create a pipeline to periodically retrain the knowledge base with new data.

---
<div align="center">
Made with ❤️ and lots of code.
</div>
