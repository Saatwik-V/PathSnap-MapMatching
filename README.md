# PathSnap: Map-Matching & Route Reconstruction

**UE24CS352A Machine Learning Hackathon, PES University**

PathSnap turns noisy 1 Hz GPS observations into the road-network route a vehicle actually drove. A classifier scores
candidate road arcs, a Hidden Markov Model decodes the whole trip with the Viterbi algorithm, and a deterministic
repair step guarantees a connected path. The system runs as a **two-laptop deployment**: a FastAPI model server on
Laptop 1 and a Streamlit + Folium evaluator interface on Laptop 2.

| | |
|---|---|
| **Technical report** | [`report/PathSnap_Technical_Report.pdf`](report/PathSnap_Technical_Report.pdf) (source: [`report/TECHNICAL_REPORT.md`](report/TECHNICAL_REPORT.md)) |
| **Champion model** | Optuna-tuned HistGradientBoosting, 12 features, `results/best_tuned_model.joblib` |
| **Held-out Top-1 / continuity** | student-test 88.9% / 99.0% &nbsp;·&nbsp; evaluator-test dry run 86.3% / 99.1% |
| **Latency** | about 0.9 s per trip end to end |

---

## 1. How it works

```
 .track .nodes .arcs                                                    predicted route
        |                                                                      ^
        v                                                                      |
 KD-tree candidates ──> 12 features ──> classifier ──> HMM + Viterbi ──> connectivity repair
 (10 nearest arcs         per (point,      P(arc is        emission x          bridges gaps with real
  per GPS point)          candidate)       the true arc)   transition          shortest-path arcs
```

1. **Candidate generation** (`data_processing/candidate_gen.py`): a `cKDTree` over road-arc midpoints returns the 10 nearest arcs per GPS point.
2. **Features** (`feature_engineering/extract_features.py`): 12 geometric and kinematic features (perpendicular distance, projection ratio, heading error and cosine, arc length, endpoint distances, speed, time gap, candidate rank, and two derived terms). Training and serving share this one function.
3. **Classifier** (`model_training/`, `hyperparameter_optimisation/`): five model families compared under `GroupKFold` by trip; the best trade-off model is tuned with Optuna.
4. **HMM + Viterbi** (`hmm_viterbi/viterbi.py`): hidden states are candidate arcs; emission = classifier probability x Gaussian distance term (sigma = 25 m); transition compares graph shortest-path distance with great-circle displacement (beta = 50 m).
5. **Connectivity repair**: consecutive predicted arcs that are not graph-adjacent are bridged with the real shortest-path arcs, only within the same 400 m radius the transition model uses. It never changes the per-point assignment, so accuracy is unaffected.
6. **Serving** (`api/`, `ui/`): FastAPI computes everything and scores it against the reference route *server-side*; Streamlit shows the scorecards and an interactive Folium map.

---

## 2. Quick start

**Requirements:** Python 3 with `pip`. Developed and tested on Python 3.13.7 (`requirements-lock.txt` lists the exact versions used).

```bash
git clone https://github.com/Saatwik-V/PathSnap-MapMatching.git
cd PathSnap-MapMatching
python3 -m venv venv
source venv/bin/activate            # Windows: .\venv\Scripts\activate
pip install -r requirements.txt     # or requirements-lock.txt for the exact tested versions
```

**Dataset.** `Student folder/` (course data) is not in git. Place it at the repository root so that
`Student folder/train/`, `Student folder/student_test/` and `Student folder/evaluator_test/` exist, or point the API at
another location with the environment variable `PATHSNAP_DATA_DIR`. Each trip folder holds `.track`, `.nodes`, `.arcs`
and (for train and student_test) `.route` files.

### Laptop 1: model server and API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Binding to `0.0.0.0` (not `127.0.0.1`) is what lets Laptop 2 reach it. Keep this terminal visible: every request prints a
log banner with the trip, data sizes, server-side scores and latency. Interactive API docs: `http://localhost:8000/docs`.

### Laptop 2: evaluator interface
```bash
streamlit run ui/app.py
```
Open `http://localhost:8501`, enter `http://<LAPTOP_1_IP>:8000` in the sidebar, pick a trip, and press
**Snap Route & Evaluate**. Laptop 2 needs the trip data too (the UI draws the road network and GPS locally): copy
`Student folder/evaluator_test/` over, **without the `.route` files**, which are never needed on Laptop 2.

### Verify the connection (from Laptop 2)
```bash
curl http://<LAPTOP_1_IP>:8000/health          # Windows PowerShell: use curl.exe
curl http://<LAPTOP_1_IP>:8000/trips
python deployment/rehearsal_all_trips.py --api http://<LAPTOP_1_IP>:8000   # all 10 trips, prints scorecards
```
Full network walkthrough (hotspot, firewall, fallbacks): [`deployment/setup_network.md`](deployment/setup_network.md).
Live-demo playbook: [`docs/LIVE_DEMO_RUN_GUIDE.md`](docs/LIVE_DEMO_RUN_GUIDE.md).

### API reference
| Endpoint | Purpose |
|---|---|
| `GET /health` | connectivity check |
| `GET /trips` | the ten reserved evaluation trips (discovered from disk; `evaluator_test` first, `student_test` as fallback) |
| `POST /predict/{trip_id}` | runs the full pipeline; returns metrics, predicted route and raw GPS (plus the reference route for `student_test` trips only) |

---

## 3. Results

Every number comes from `results/*.csv|json` and is reproducible with the commands in section 5.

**Five-model benchmark** (12 features; trained on 80 trips, 2,084,170 candidate rows; CV is `GroupKFold(5)` by trip; test columns are the 10 held-out student-test trips):

| Model | Family | CV Macro F1 | Test Top-1 | Test Macro F1 | Test log loss | Train time |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | Linear & Probabilistic | 0.7444 | 82.06% | 0.7594 | 0.5399 | 1.1 s |
| k-NN (distance-weighted) | Instance-based | 0.7526 | 80.68% | 0.7568 | 2.9169 | 0.68 s |
| Random Forest | Ensemble (Bagging) | 0.7856 | 84.75% | 0.7869 | 0.464 | 43.84 s |
| **HistGradientBoosting** (selected) | Boosting | 0.7908 | 84.88% | 0.791 | 0.4673 | 5.68 s |
| MLP | Neural Network | 0.7952 | 84.16% | 0.7935 | 0.4861 | 107.83 s |

All five land within about 0.04 test Macro F1 of each other. HistGradientBoosting was selected because it is
essentially tied on held-out quality while training about 19x faster than the MLP, which makes a real 30-trial
Bayesian search practical.

**Bayesian tuning** (Optuna TPE, 30 trials, `GroupKFold(3)`), both rows scored on the same held-out partition:

| | CV Macro F1 | Test Top-1 | Test Macro F1 | Test log loss |
|---|:---:|:---:|:---:|:---:|
| Untuned | 0.7908 | 84.88% | 0.791 | 0.4673 |
| Tuned | 0.797 | 84.99% | 0.7937 | 0.4719 |

Tuning is a small, real gain. Isotonic probability calibration was also tried (better log loss, no accuracy gain, ~60x larger
model) and not adopted (`results/calibration_comparison.csv`).

**HMM / Viterbi on the 10 student-test trips** (sigma = 25, beta = 50):

| Stage | Mean pointwise accuracy | Mean route continuity |
|---|:---:|:---:|
| Classifier only | 86.32% | 76.82% |
| + HMM / Viterbi | 88.86% | 90.44% |
| + connectivity repair | 88.86% | **99.01%** |

The HMM alone does not reach a 95% continuity target; the repair stage does, and it is reported separately so the two are not conflated.

### Reading the metrics honestly
* **Top-1** is the share of GPS points whose predicted arc lies on the reference route. **Continuity** only says the predicted path is connected and drivable, *not* that it is correct.
* **Known limitation:** the main error source is direction ambiguity. The two directions of a street are near-identical arcs, so the decoder sometimes picks the reverse one. Such a path stays connected (high continuity) while Top-1 drops, as on evaluator trip `00000015` (62.4% Top-1, 100% continuity). See report section 5 and `results/direction_error_analysis.csv`.
* Per-trip variance is large (evaluator dry run: 62.4% to 95.2% Top-1), so read the 10-trip mean together with that spread.

---

## 4. Data partitions and integrity

| Partition | Trips | Use |
|---|:---:|---|
| `train` | 80 | EDA, features, cross-validation, tuning |
| `student_test` | 10 | development-stage comparison, tuned-vs-untuned scoring, HMM parameter selection |
| `evaluator_test` | 10 | **final evaluation only**; never used for training, feature selection, hyperparameter or threshold choice, or model selection |

* **Ground truth stays server-side.** `.route` files are read only by the API. For `evaluator_test` trips the API returns no reference geometry and the UI never reads or draws one.
* **No cross-trip leakage.** All validation is `GroupKFold` grouped by `trip_id`.
* **No hardcoded predictions.** Every response is computed at request time; trips are discovered from disk, not listed in code.

---

## 5. Reproducing the pipeline

`results/train_features.csv` (about 208 MB) is not committed (GitHub's 100 MB limit); step 1 regenerates it.

```bash
# 1. Candidate generation + 12-feature extraction
python3 -c "from feature_engineering.extract_features import build_dataset; \
  build_dataset('Student folder/train', 'results/train_features.csv'); \
  build_dataset('Student folder/student_test', 'results/test_features.csv')"

# 2. Five-model benchmark (CV, held-out test, per-model HMM continuity)
python3 model_training/train_5_models.py

# 3. Bayesian hyperparameter optimisation, then the calibration experiment
python3 hyperparameter_optimisation/tune_optuna.py --family histgb --trials 30 --splits 3
python3 model_training/calibrate_model.py

# 4. HMM: parameter sweep and student-test evaluation
python3 hmm_viterbi/tune_hmm_params.py
python3 hmm_viterbi/viterbi.py --base_dir "Student folder/student_test" \
  --model_path results/best_tuned_model.joblib --n_trips 10 --output results/hmm_vs_classifier_studenttest.csv

# 5. EDA, diagnostics and maps
python3 data_processing/eda.py
python3 model_evaluation/generate_diagnostic_plots.py
python3 model_evaluation/analyze_direction_errors.py
python3 visualisations/generate_eda_figures.py
python3 visualisations/generate_production_artifacts.py

# 6. Technical report PDF (needs pandoc, Google Chrome and `pip install playwright`)
python3 report/build_report.py
```

---

## 6. Repository structure

```
api/                          FastAPI server: main.py (endpoints, logging), utils.py (data discovery, server-side scoring)
data_processing/              candidate_gen.py (KD-tree candidates, labels), eda.py (dataset audit and EDA plots)
feature_engineering/          extract_features.py (the 12 features; shared by training and serving)
model_training/               train_5_models.py (five-model benchmark), calibrate_model.py (calibration experiment)
hyperparameter_optimisation/  tune_optuna.py (30-trial Bayesian search)
hmm_viterbi/                  viterbi.py (graph, HMM, Viterbi, repair), tune_hmm_params.py (sigma/beta sweep)
model_evaluation/             generate_diagnostic_plots.py, analyze_direction_errors.py
visualisations/               folium_map.py (4-layer map), generate_production_artifacts.py, generate_eda_figures.py
ui/                           app.py (Streamlit evaluator interface, Laptop 2)
deployment/                   setup_network.md, rehearsal_all_trips.py
report/                       TECHNICAL_REPORT.md, build_report.py, PathSnap_Technical_Report.pdf, figures/
results/                      CSV/JSON results, trained models (best_tuned_model.joblib, best_model.joblib), maps/, plots/
docs/                         team role plans and demo guides
requirements.txt              flexible dependency ranges        requirements-lock.txt   exact tested versions
```

---

## 7. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Laptop 2 shows "Cannot reach Laptop 1 API" | API not running, bound to `127.0.0.1`, wrong IP (it can change after reconnecting), or firewall. Restart with `--host 0.0.0.0`, re-check the IP, allow incoming connections. |
| `ping` between laptops fails | Wi-Fi client isolation. Move both to a phone hotspot. |
| `curl` prompts in Windows PowerShell | `curl` is an alias of `Invoke-WebRequest`; use `curl.exe`. |
| UI shows scorecards but no map | the trip data is missing on Laptop 2; copy `Student folder/evaluator_test/` (without `.route`). |
| `/trips` returns 503 | no dataset found; place `Student folder/` at the repo root or set `PATHSNAP_DATA_DIR`. |
| Port 8000 already in use | `kill -9 $(lsof -t -i:8000)` (macOS/Linux) and restart. |

---

## 8. Team

| Member | Role |
|---|---|
| Saatwik | ML architecture, model comparison and tuning, HMM/Viterbi decoder |
| Uttam | Streamlit evaluator interface, Folium visualisation |
| Vikas Jadar | Dataset audit, candidate generation, feature pipeline, EDA |
| Vikas Aihole | FastAPI backend, two-laptop deployment, report coordination |

All algorithms, features, models and results in this repository were produced by the team; every reported metric is
generated by the scripts above and can be regenerated from the committed code.
