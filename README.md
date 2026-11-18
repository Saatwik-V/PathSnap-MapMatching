# PathSnap: Distributed Map-Matching & Sequential Route Reconstruction

**UE24CS352A: Machine Learning Hackathon | PES University**  
*An end-to-end production-ready distributed system for map-matching noisy vehicle GPS trajectories onto physical road network graphs.*

---

## 1. System Overview & Problem Formulation

Vehicle GPS units in urban environments suffer from severe multipath reflections, satellite shadowing, low sampling frequencies (~1 Hz), and sensor noise. As a result, raw observations appear to jump across buildings, drift onto parallel streets, or breach road topology.

**PathSnap** solves this through a multi-stage machine learning and graph-theoretic pipeline:
1. **Spatial Indexing & Candidate Generation**: A `cKDTree` built on road arc midpoints retrieves candidate segments within an adaptive radius buffer in $\mathcal{O}(\log M)$ time.
2. **Kinematic & Geometric Feature Extraction**: Computes a 12-dimensional feature vector per candidate (distance, projection ratios, heading angles, directional cosines, velocity, endpoint symmetry, and graph topological ranks).
3. **Supervised Binary Classification**: Trains across 5 distinct algorithm families with strict `GroupKFold(groups=trip_id)` cross-validation to eliminate spatial data leakage.
4. **Bayesian Hyperparameter Optimization**: Tunes the champion model via **Optuna** (TPE sampler, 30 trials) maximizing generalization on held-out partitions.
5. **Sequential Route Reconstruction (HMM + Viterbi)**: Decodes the globally optimal continuous route path over a directed road graph (`NetworkX`), improving Route Continuity from 76.4% to **91.9%**.
6. **Distributed 2-Laptop Deployment**: Deployed across a local area network using a **FastAPI** model server (Laptop 1) and a **Streamlit + Folium** interactive evaluator client (Laptop 2).

---

## 2. Distributed Architecture

```
                    TWO-LAPTOP DISTRIBUTED SYSTEM ARCHITECTURE

  +---------------------------------------+          +---------------------------------------+
  |               LAPTOP 1                |          |               LAPTOP 2                |
  |     FastAPI Backend & Model Server    |          |         Streamlit Evaluator UI        |
  |             (Saatwik)                 |          |                (Uttam)                |
  |                                       |          |                                       |
  |  - Host: 0.0.0.0:8000                 |  HTTP    |  - Host: localhost:8501               |
  |  - Spatial KD-Tree Indexing Engine    | <======> |  - Evaluator 10-Trip Dropdown Selector|
  |  - 12-Feature Extraction Module       |   REST   |  - Real-Time HTTP Client              |
  |  - Tuned Champion Model (Joblib)     |   (LAN)  |  - Interactive 4-Layer Folium Map     |
  |  - NetworkX HMM & Viterbi Decoder     |          |  - Metric Scorecards:                 |
  |  - Server-Side Ground-Truth (.route)  |          |    * Top-1 Observation Accuracy       |
  |  - Live Request Logging with Latency  |          |    * Route Continuity %               |
  |                                       |          |    * Macro F1 & Server Latency (ms)   |
  +---------------------------------------+          +---------------------------------------+
```

---

## 3. Detailed Repository & Folder Structure

This repository strictly complies with the official PathSnap submission guidelines:

```
PathSnap-MapMatching/
├── api/
│   ├── main.py                     # Production FastAPI server with request middleware and live logging
│   └── utils.py                    # Dataset discovery, KD-Tree indexing, graph decoding, and metric engine
├── data_processing/
│   ├── candidate_gen.py            # Spatial KD-Tree candidate generation & ground-truth route alignment
│   └── eda.py                      # Comprehensive automated EDA pipeline (temporal, spatial, multi-region)
├── feature_engineering/
│   └── extract_features.py         # Vectorized feature engine computing 12 kinematic & geometric features
├── model_training/
│   ├── baseline.py                 # Initial baseline verification script
│   └── train_5_models.py           # 5-family benchmark suite with GroupKFold cross-validation
├── hyperparameter_optimisation/
│   └── tune_optuna.py              # 30-trial Bayesian TPE hyperparameter optimization via Optuna
├── hmm_viterbi/
│   └── viterbi.py                  # NetworkX road graph loader, emission/transition models, and Viterbi decoder
├── ui/
│   └── app.py                      # Streamlit evaluator portal running on Laptop 2
├── visualisations/
│   └── folium_map.py               # Reusable 4-layer Folium visualizer with layer controls & legends
├── deployment/
│   └── setup_network.md            # Step-by-step 2-laptop LAN hotspot configuration and IP setup guide
├── report/
│   ├── TECHNICAL_REPORT.md         # Mandatory 7-section consolidated technical report source
│   └── generate_report_pdf.py      # Automated script compiling the technical report into a submission PDF
├── results/
│   ├── best_model.joblib           # Pre-tuned champion model weights
│   ├── best_tuned_model.joblib     # Optuna-tuned champion model artifact
│   ├── five_model_comparison.csv   # Official 5-model benchmark metrics table
│   ├── tuned_vs_untuned.csv        # Optuna tuning improvement table (71.6% -> 82.7%)
│   ├── hmm_vs_classifier_studenttest.csv # Per-trip HMM continuity comparison (76.4% -> 91.9%)
│   ├── optuna_trials.csv           # 30-trial optimization audit log
│   ├── optuna_best_params.json     # Serialized winning hyperparameters
│   ├── data_quality_notes.md       # Comprehensive multi-region and dropout audit documentation
│   ├── train_features.csv          # Master training dataset (80 trips, 995,190 candidate rows)
│   ├── test_features.csv           # Held-out student-test dataset (10 trips, 95,801 candidate rows)
│   └── plots/
│       ├── sampling_intervals.png  # GPS delta-t histogram with dropout analysis
│       ├── trajectory_lengths.png  # GPS points per trip volume distribution
│       ├── geographic_regions.png  # Global regional clusters plot (Europe, Russia, USA, Japan)
│       ├── model_macro_f1_comparison.png # 5-model Macro F1 comparison chart
│       ├── model_top1_acc_comparison.png # 5-model Top-1 Accuracy comparison chart
│       └── model_logloss_comparison.png  # 5-model cross-entropy log loss chart
├── docs/
│   ├── README.md                   # Team role breakdown & dependency graph
│   ├── saatwik.md                  # Lead ML Architect & HMM Core playbook
│   ├── uttam.md                    # Frontend UI & Folium Visualizer playbook
│   ├── vikas_jadar.md              # Data Pipeline & Feature Engineering playbook
│   ├── vikas_aihole.md             # FastAPI Backend & Report Coordinator playbook
│   └── FIRST_REVIEW_MASTER_GUIDE.md# End-to-end system guide and comprehensive viva manual
├── requirements.txt                # System dependencies
├── .gitignore                      # Git hygiene rules (dataset and materials excluded)
└── README.md                       # Master system documentation
```

---

## 4. Benchmark Results & Key Findings

### A. 5-Model Baseline Benchmark (Without Optimization)
*Evaluated across all 80 training trips (995,190 candidate observations) under 3-fold `GroupKFold(groups=trip_id)`:*

| Model Family | Specific Algorithm | CV Macro F1 | Observation Top-1 Acc | Log Loss | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Linear** | Logistic Regression (Standardized) | **0.7562** | 70.52% | 0.5033 | 0.000 ms |
| **Instance-Based** | Distance-Weighted $k$-NN ($k=5$) | **0.7697** | 69.76% | 2.7509 | 0.003 ms |
| **Ensemble (Bagging)** | Random Forest (100 Trees, Depth 12) | **0.7905** | 71.70% | 0.4437 | 0.002 ms |
| **Boosting** | HistGradientBoosting | **0.7879** | 71.59% | 0.4445 | 0.001 ms |
| **Neural Network** | **Multi-Layer Perceptron (MLP)** | **0.7984** ⭐ | **71.86%** | **0.4402** | 0.000 ms |

### B. Bayesian Hyperparameter Optimization (Optuna)
*Tuning HistGradientBoosting over 30 trials with TPESampler:*
* **Held-out Student-Test Top-1 Accuracy**: Improved from **71.59% $\to$ 82.69%** *(+11.10% absolute gain)*.
* **Best Hyperparameters**: `learning_rate=0.1055, max_depth=7, max_leaf_nodes=39, l2_reg=0.00216, min_samples_leaf=48`.

### C. HMM & Viterbi Sequential Route Reconstruction
*Tested across all 10 held-out student-test trips:*
* **Mean Observation Accuracy**: Improved from **83.10% $\to$ 87.60%**.
* **Mean Route Continuity %**: Surged from **76.38% (classifier-only) $\to$ 91.94% (classifier + HMM)** *(+15.56% gain)*, reaching **98.23%** on Trip `00000007`.

---

## 5. Installation & Execution

### 1. Installation
```bash
git clone https://github.com/Saatwik-V/PathSnap-MapMatching.git
cd PathSnap-MapMatching

python3 -m venv venv
source venv/bin/activate       # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Running on Laptop 1 (Model Server & API — Saatwik)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
*Health Check*: `curl http://localhost:8000/health`  
*API Docs*: `http://localhost:8000/docs`

### 3. Running on Laptop 2 (Evaluator UI — Uttam)
```bash
streamlit run ui/app.py
```
*(Enter Laptop 1's LAN IP address e.g. `http://192.168.1.x:8000` in the sidebar and select any test trip).*

---

## 6. Academic Integrity & Data Hygiene
* **Zero Cross-Trip Leakage**: All validation is conducted under `GroupKFold` strictly grouped by `trip_id`.
* **Private Ground Truth**: Reference `.route` files remain strictly server-side on Laptop 1; the evaluator client receives only predicted routes and calculated performance scorecards.
* **Independent Implementation**: All algorithms, network models, feature formulations, and API endpoints are independently developed.
