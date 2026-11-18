# PathSnap: Distributed Map-Matching & Sequential Route Reconstruction

**UE24CS352A: Machine Learning Hackathon | PES University**

PathSnap is an end-to-end distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy GPS observations to a physical road network graph. It couples supervised candidate road segment classification with a Hidden Markov Model (HMM) Viterbi decoder, deployed across a two-laptop distributed architecture.

---

## 1. System Architecture

```
                    TWO-LAPTOP DEPLOYMENT ARCHITECTURE

  +--------------------------------+          +----------------------------------+
  |            LAPTOP 1            |          |             LAPTOP 2             |
  |     FastAPI Server & Model     |          |       Streamlit Evaluator UI     |
  |                                |          |                                  |
  |  - Candidate Generation        |  HTTP    |  - 10 Evaluation Trip Selector   |
  |  - Feature Extraction Pipeline | <======> |  - Interactive Folium Visualizer |
  |  - Tuned ML Classifier        |   REST   |  - Performance Scorecards        |
  |  - HMM & Viterbi Decoder       |          |    (Top-1 Acc, Macro F1, Cont.)  |
  |  - Ground-Truth Evaluation     |          |                                  |
  +--------------------------------+          +----------------------------------+
```

---

## 2. Repository Structure

This repository strictly adheres to the official PathSnap submission guidelines:

* `data_processing/`: Raw GPS and road-network parsing, coordinate projections, and spatial candidate generation.
* `feature_engineering/`: Geometric and kinematic feature extraction (perpendicular distance, heading error, speed, projection ratios).
* `model_training/`: Training scripts for 5 distinct machine learning model families (Linear, $k$-NN, Random Forest, LightGBM/XGBoost, MLP).
* `model_evaluation/`: Group-aware cross-validation harness, comparison tables, and confusion matrices.
* `hyperparameter_optimisation/`: Bayesian hyperparameter optimization (Optuna) and parameter search history.
* `hmm_viterbi/`: Road graph construction (`NetworkX`), transition and emission probability models, and Viterbi decoding.
* `api/`: FastAPI backend implementation running on Laptop 1 with evaluation endpoints.
* `ui/`: Evaluator-facing interface running on Laptop 2 with interactive Folium map displays.
* `visualisations/`: Folium multi-layer map generation and performance metric plotting scripts.
* `deployment/`: Network configuration guide, startup scripts, and verification commands for the 2-laptop setup.
* `results/`: Experiment logs, benchmark tables, and saved model configurations.
* `report/`: Technical report sources and compiled PDF.

---

## 3. Installation & Setup

### Prerequisites
* Python 3.10+
* Git

### Step-by-Step Installation
```bash
# Clone the repository
git clone https://github.com/Saatwik-V/PathSnap-MapMatching.git
cd PathSnap-MapMatching

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: .\venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 4. Execution & Deployment

### Laptop 1: Starting the Model & Evaluation API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Laptop 2: Launching the Evaluator Interface
```bash
streamlit run ui/app.py
```
*(Enter Laptop 1's LAN IP address in the UI to connect and evaluate any test trip).*

---

## 5. Academic Integrity
This repository represents the independent work of the project team. No model weights, code, or hyperparameter configurations have been shared across teams. Evaluator ground-truth labels remain strictly server-side.
