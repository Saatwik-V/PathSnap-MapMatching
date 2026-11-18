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
5. **Sequential Route Reconstruction (HMM + Viterbi)**: Decodes the globally optimal continuous route path over a directed road graph (`NetworkX`), followed by a deterministic connectivity-repair step, improving Route Continuity from 76.8% (classifier-only) to 90.4% (HMM) to **99.0%** (HMM + repair).
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
│   └── utils.py                    # Dataset discovery + server-side metrics; delegates route reconstruction
│                                    #   to hmm_viterbi.viterbi.reconstruct_route (the same 12-feature pipeline
│                                    #   used for every offline evaluation number in this README)
├── data_processing/
│   ├── candidate_gen.py            # Spatial KD-Tree candidate generation & ground-truth route alignment
│   └── eda.py                      # Comprehensive automated EDA pipeline (temporal, spatial, multi-region)
├── feature_engineering/
│   └── extract_features.py         # Vectorized feature engine computing 12 kinematic & geometric features
├── model_training/
│   └── train_5_models.py           # 5-family benchmark suite with GroupKFold CV + student-test/HMM eval
├── hyperparameter_optimisation/
│   └── tune_optuna.py              # 30-trial Bayesian TPE hyperparameter optimization via Optuna
├── hmm_viterbi/
│   └── viterbi.py                  # NetworkX road graph loader, emission/transition models, and Viterbi decoder
├── model_evaluation/
│   └── generate_diagnostic_plots.py # Confusion matrix, tuned-vs-untuned, HMM continuity gain, static route map
├── ui/
│   └── app.py                      # Streamlit evaluator portal running on Laptop 2 (download button + HMM status)
├── visualisations/
│   ├── folium_map.py               # Reusable 4-layer Folium visualizer with layer controls & floating legend
│   └── generate_production_artifacts.py # Batch-generates interactive HTML maps for all 10 student-test trips
├── deployment/
│   └── setup_network.md            # Step-by-step 2-laptop LAN hotspot configuration and IP setup guide
├── report/
│   ├── TECHNICAL_REPORT.md         # Mandatory 7-section consolidated technical report source
│   ├── generate_report_pdf.py      # Compiles the report into a PDF, reading all numbers live from results/
│   └── PathSnap_Technical_Report.pdf # Generated submission PDF
├── results/
│   ├── best_model.joblib           # Untuned champion model weights (MLP, highest CV Macro F1)
│   ├── best_tuned_model.joblib     # Optuna-tuned production model (HistGradientBoosting) -- used by the API
│   ├── five_model_comparison.csv   # Official 5-model benchmark: CV + held-out student-test + route continuity
│   ├── tuned_vs_untuned.csv        # Fair untuned-vs-tuned table (both on student-test)
│   ├── hmm_vs_classifier_studenttest.csv # Per-trip continuity: classifier-only -> HMM -> HMM+repair (76.8% -> 90.4% -> 99.0%)
│   ├── calibration_comparison.csv  # Untuned / tuned / calibrated, all on student-test
│   ├── hmm_param_sweep.csv         # HMM sigma x beta grid results
│   ├── optuna_trials.csv           # 30-trial optimization audit log
│   ├── optuna_best_params.json     # Serialized winning hyperparameters
│   ├── data_quality_notes.md       # Comprehensive multi-region and dropout audit documentation
│   ├── PHASE2_SUMMARY.md           # Audit trail: 6->12 feature expansion, tuning, and production-polish log
│   ├── test_features.csv           # Held-out student-test dataset (10 trips, 191,600 candidate rows)
│   ├── maps/                       # Standalone interactive Folium HTML maps, all 10 student-test trips
│   └── plots/
│       ├── sampling_intervals.png         # GPS delta-t histogram with dropout analysis
│       ├── trajectory_lengths.png         # GPS points per trip volume distribution
│       ├── geographic_regions.png         # Global regional clusters plot (Europe, Russia, USA, Japan, ...)
│       ├── model_macro_f1_comparison.png  # 5-model Macro F1 comparison chart
│       ├── model_top1_acc_comparison.png  # 5-model Top-1 Accuracy comparison chart
│       ├── model_logloss_comparison.png   # 5-model cross-entropy log loss chart
│       ├── confusion_matrix_champion.png  # Tuned champion model's confusion matrix (student-test)
│       ├── tuned_vs_untuned_comparison.png # Optuna tuning impact chart
│       ├── hmm_continuity_gain.png        # Per-trip classifier-only vs classifier+HMM continuity
│       └── static_route_map.png           # Static rendering of the Folium map layers (for the PDF report)
│
│   `results/train_features.csv` (2,084,170 rows, 80 trips, ~208MB) is regenerated, not committed --
│   see "Reproducing results" below; it exceeds GitHub's 100MB push limit.
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

All numbers below are read directly from `results/*.csv` and `results/*.json` --
see `report/TECHNICAL_REPORT.md` for full methodology, per-model analysis, and
the compulsory per-model Q&A.

### A. 5-Model Baseline Benchmark (12 features)
*Trained on all 80 training trips (2,084,170 candidate rows) under `GroupKFold(n_splits=5, groups=trip_id)`,
then evaluated once, untouched, on the 10 held-out student-test trips (including a full HMM/Viterbi pass):*

| Model | Family | CV Macro F1 | CV Top-1 | Student-Test Top-1 | Student-Test F1 | Route Continuity | Log Loss | Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | Linear | 0.7444 | 70.77% | 82.06% | 0.7594 | 92.87% | 0.5087 | 1.09s |
| Distance-Weighted k-NN | Instance-based | 0.7526 | 69.50% | 80.68% | 0.7568 | 91.01% | 2.5580 | 0.66s |
| Random Forest | Ensemble | 0.7856 | 73.00% | 84.75% | 0.7869 | 92.31% | 0.4153 | 49.83s |
| **HistGradientBoosting** | Boosting | 0.7908 | 73.09% | **84.88%** | **0.7910** | **92.32%** | 0.4117 | 5.68s |
| MLP | Neural Network | **0.7952** ⭐ | 72.70% | 84.16% | 0.7935 | 92.15% | 0.4238 | 105.84s |

⭐ MLP has the single highest CV Macro F1, but **HistGradientBoosting was selected**
for tuning and deployment: it beats MLP on every held-out student-test metric
while training ~18.6x faster, and its CV-vs-held-out gap is the smallest of all
five models (strongest generalisation evidence). See the report for the full
per-model rationale.

### B. Bayesian Hyperparameter Optimization (Optuna)
*Tuning HistGradientBoosting over 30 trials with TPESampler, `GroupKFold(n_splits=3)`; untuned and tuned are both scored on the same held-out student-test partition:*
* **CV Macro F1**: 0.7908 -> 0.7970 (+0.006).
* **Held-out Student-Test Top-1 Accuracy**: 84.88% -> 84.99% (+0.11 points) -- a small, honest gain; the default configuration was already near this feature set's ceiling. (An earlier draft reported "73.09% -> 84.99%" by mixing a cross-validation number with a student-test number; that was wrong and has been corrected.)
* **Best Hyperparameters**: `learning_rate=0.0894, max_depth=13, max_leaf_nodes=84, l2_regularization=0.0232, min_samples_leaf=116`.

### C. Probability Calibration (experiment, not adopted)
*Isotonic calibration fit with GroupKFold-by-trip splits (`model_training/calibrate_model.py`, `results/calibration_comparison.csv`):*
* Student-test log loss 0.4719 -> 0.4539 and Brier 0.1411 -> 0.1399, but Top-1, pointwise accuracy and route continuity do not improve, and the model grows to ~88MB. The tuned, uncalibrated model stays in production.

### D. HMM & Viterbi Sequential Route Reconstruction
*Tested across all 10 held-out student-test trips with the tuned model, HMM `sigma=25, beta=50` (chosen by a 4x5 grid, `results/hmm_param_sweep.csv`; the surface is flat, so these are a minor lever):*
* **Mean pointwise accuracy**: 86.32% (classifier-only) -> **88.86%** (HMM); improved on all 10 trips.
* **Mean Route Continuity %**: 76.82% (classifier-only) -> 90.44% (HMM/Viterbi alone) -> **99.01%** (after deterministic connectivity repair; per-trip range 96.97%-100%).
* **Connectivity repair**: after Viterbi decoding, consecutive predicted arcs that are not graph-adjacent are bridged with the real shortest-path arcs, only when that path lies within the same 400m radius the transition model already uses. 1,639 gaps were bridged and 64 genuine network gaps were left untouched. Repair never changes the per-GPS-point assignment, so Top-1 / pointwise accuracy / Macro F1 are unaffected by it; the HMM alone did not reach the 95% continuity target, and the repair stage is what does.

### E. Dataset Partition Policy
`Student folder/` is gitignored and never committed. `train` (80 trips) is used for training; `student_test` (10 trips) for all development-stage comparison, tuning-table scoring and HMM parameter selection; `evaluator_test` (10 trips) is **final-evaluation only** -- it is never used for training, feature selection, hyperparameter or threshold choice, and no evaluator_test accuracy has been used for any decision. The API serves `evaluator_test` trips when that folder is present on Laptop 1 (falling back to `student_test` otherwise), and ground-truth `.route` files are read server-side only.

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
*(For a true two-laptop demo, bind to `0.0.0.0` as above and give Laptop 2 your
LAN IP -- full walkthrough in `deployment/setup_network.md`.)*

### 3. Running on Laptop 2 (Evaluator UI — Uttam)
```bash
streamlit run ui/app.py
```
*(Enter Laptop 1's LAN IP address e.g. `http://192.168.1.x:8000` in the sidebar and select any test trip).*

### 4. Reproducing the full pipeline from scratch
`results/train_features.csv` is not committed (over GitHub's 100MB limit) --
regenerate it and every downstream artifact with:
```bash
# 1. Candidate generation + 12-feature extraction (train: 80 trips, test: 10 trips)
python3 -c "from feature_engineering.extract_features import build_dataset; \
  build_dataset('Student folder/train', 'results/train_features.csv'); \
  build_dataset('Student folder/student_test', 'results/test_features.csv')"

# 2. 5-model benchmark (CV + held-out student-test + per-model HMM route continuity)
python3 model_training/train_5_models.py

# 3. Bayesian hyperparameter optimisation (30 trials, HistGradientBoosting)
python3 hyperparameter_optimisation/tune_optuna.py --family histgb --trials 30 --splits 3

# 4. HMM/Viterbi evaluation on the student-test partition
python3 hmm_viterbi/tune_hmm_params.py   # sigma/beta sweep (student_test only)
python3 model_training/calibrate_model.py   # calibration experiment
python3 hmm_viterbi/viterbi.py --base_dir "Student folder/student_test" \
  --model_path results/best_tuned_model.joblib --n_trips 10

# 5. Diagnostic plots + production Folium maps
python3 model_evaluation/generate_diagnostic_plots.py
python3 visualisations/generate_production_artifacts.py

# 6. Technical report PDF (reads results/*.csv and results/*.json directly)
python3 report/generate_report_pdf.py
```

---

## 6. Academic Integrity & Data Hygiene
* **Zero Cross-Trip Leakage**: All validation is conducted under `GroupKFold` strictly grouped by `trip_id`.
* **Private Ground Truth**: Reference `.route` files remain strictly server-side on Laptop 1; the evaluator client receives only predicted routes and calculated performance scorecards.
* **Independent Implementation**: All algorithms, network models, feature formulations, and API endpoints are independently developed.
