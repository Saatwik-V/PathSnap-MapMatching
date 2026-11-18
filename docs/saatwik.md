# Saatwik — Chief ML Architect & Sequential Decoding Lead

## Mission
Own the mathematical/algorithmic core of PathSnap: train and compare the five
required model families, select and Bayesian-tune the champion model, and build
the HMM + Viterbi decoder that turns per-point candidate predictions into a
single continuous, road-network-consistent route.

Your output is consumed directly by Vikas Aihole's FastAPI `/predict/{trip_id}`
endpoint, so the final artifacts (model + preprocessing + HMM config) must be
loadable in a clean Python session with no notebook-only state.

## Folders you own
- `model_training/` — one script/notebook per model family.
- `model_evaluation/` — the group-aware comparison harness and metrics tables.
- `hyperparameter_optimisation/` — Optuna search + tuned artifact.
- `hmm_viterbi/` — emission/transition modelling + Viterbi decoder + route continuity scoring.
- `results/` — `best_model.joblib`, metrics CSVs, comparison tables (shared with Vikas Aihole for the report and with the API for loading).

## Inputs you depend on
- Vikas Jadar's `results/train_features.csv` and `results/test_features.csv` (candidate rows with 6 geometric features + binary label `y`, plus `trip_id` and `arc_id` columns — confirm the exact schema with him before you start, you need `trip_id` for `GroupKFold` and `arc_id` to reconstruct sequences).
- The raw `.arcs` / `.nodes` files directly, for building the road-network graph used by the HMM transition model.

## End-to-end task list

### 1. Model Training Suite (`model_training/`)
Train all five models on Vikas Jadar's feature table. Each script should:
load `train_features.csv` → fit on `(perp_dist, proj_ratio, heading_error,
arc_length, dist_to_start, dist_to_end)` → predict probability that a candidate
arc is the true one → save the fitted model + a record of its config (as JSON
or in a shared results log).

| # | Family | Model | Config |
|---|--------|-------|--------|
| 1 | Linear | Logistic Regression | with `StandardScaler` preprocessing |
| 2 | Instance-based | Distance-Weighted k-NN | `k=5`, `weights='distance'` |
| 3 | Ensemble | Random Forest | `n_estimators=100` |
| 4 | Boosting | LightGBM or XGBoost | `n_estimators=100`, `max_depth=6` |
| 5 | Neural Network | MLPClassifier | `hidden_layer_sizes=(64, 32)` |

Record for every model: training time, inference time, and the config used —
the report needs this per the Deliverables doc.

### 2. Group-Aware Evaluation Harness (`model_evaluation/`)
- Use `GroupKFold(n_splits=5)` with `groups=trip_id` — **never** a plain
  k-fold, since consecutive candidate rows from the same trip are correlated
  and a plain split leaks information across folds (this is the exact
  leakage rule called out in the Main Guidelines / Dataset doc: no observations
  from the same trip may cross partitions, and that principle extends to CV folds).
- For every one of the 5 models, compute and tabulate:
  - CV Macro F1-score
  - Observation-level Top-1 Accuracy (does the highest-scoring candidate match the reference arc?)
  - Candidate Miss Rate (cases where the true arc wasn't even in the candidate set — this is really a diagnostic on Vikas Jadar's candidate generation, report it per-model anyway since it caps achievable accuracy)
  - Log Loss (skip/mark N/A for models without a natural probability output, e.g. raw SVM; explain why in the table)
- Output a single consolidated comparison table (CSV + printed) — this is the
  exact table required in Deliverables.pdf §3.1. Also produce the plots:
  CV Macro F1 comparison across models, Top-1 Accuracy comparison, Log Loss
  comparison. Save plots to `results/plots/`.
- Pick the final model with written justification (not highest raw accuracy
  alone) — note generalisation stability across folds, inference speed, and
  compatibility with the sequential decoder (it must output per-candidate
  probabilities for the HMM emission model).

### 3. Bayesian Hyperparameter Optimisation (`hyperparameter_optimisation/`)
- Use **Optuna** on the selected model (e.g. LightGBM).
- Search space (justify ranges in the report): `learning_rate`, `num_leaves`,
  `max_depth`, `min_child_samples`. Aim for ~30 trials.
- Optimisation objective: CV Macro F1 (or another metric you justify) under
  the same `GroupKFold` scheme — **never** touch evaluator-testing data for
  this, you don't have it anyway, but also don't let the student-test set leak
  into the search objective itself (use it only afterward for a held-out check).
- Log every trial (params + objective value) — keep the full Optuna study
  object or export trials to CSV, the report needs "how many trials" and
  "best values obtained."
- Refit the final model with best params, evaluate on student-testing trips,
  and compare tuned vs untuned on: CV Macro F1, Top-1 Accuracy, Macro F1,
  Log Loss, Inference Time.
- Save the tuned model to `results/best_model.joblib`, plus whatever
  preprocessing object (e.g. the `StandardScaler`, if your final model needs
  one) the API will need to reproduce inference exactly.

### 4. HMM + Viterbi Route Reconstruction (`hmm_viterbi/`)
This is the piece that fixes "disconnected" predictions — e.g. two consecutive
GPS points independently classified onto arcs that don't connect in the road
graph.

- **Hidden states**: the candidate arc_ids at each GPS timestep (the per-point
  candidate set from Vikas Jadar's candidate generation).
- **Emission probability**: combine the trained model's predicted score with
  spatial plausibility:
  `P(O_t | a_i) ∝ ŷ_{t,i} · exp(-d_i² / (2σ²))`
  where `ŷ_{t,i}` is your model's predicted probability for candidate `i` at
  time `t`, and `d_i` is the perpendicular distance from the GPS point to arc
  `i`. Tune `σ` empirically (document what value you picked and why).
- **Transition probability**: derived from shortest-path distance between
  candidate arcs in the road-network graph (build this graph from `.nodes` +
  `.arcs` — e.g. with `networkx`). Arcs with no feasible path, or an
  implausibly long path given the elapsed time between observations, should get
  very low/zero transition probability. This is also where you handle GPS
  gaps: if the time delta between two observations is large, relax the
  transition distance threshold accordingly rather than penalizing a long but
  physically plausible jump.
- **Viterbi decoding**: implement the standard forward dynamic program over
  per-timestep candidate sets, picking the arc sequence with max joint
  emission × transition probability. Handle: missing candidates at a timestep
  (skip/interpolate), ties, and the case where **no** candidate exists at a
  given timestep.
- **Route Continuity %**: for the final decoded sequence, compute the
  percentage of consecutive predicted arcs that are directly connected in the
  road graph (i.e. share a node per `.arcs`). Report this for both
  classifier-only predictions (no HMM) and classifier+HMM predictions — the
  report explicitly requires this comparison.
- Expose a single clean function, e.g. `reconstruct_route(trip_id, candidates_df, model) -> list[arc_id]`,
  that the API layer can call directly.

## What you need from teammates
- **From Vikas Jadar**: exact schema of `train_features.csv` / `test_features.csv` (column names, whether `trip_id`/`timestamp`/`arc_id` are included), and confirmation of units (meters vs degrees) for `perp_dist`/`arc_length`.
- **From Vikas Aihole**: how he wants to call your `reconstruct_route` function from the API (import path, expected return format — likely a list of arc_ids plus per-point metrics).

## What you hand off
- `results/best_model.joblib` (+ any preprocessing artifact), to Vikas Aihole for the API.
- A callable route-reconstruction function/module in `hmm_viterbi/`, to Vikas Aihole.
- The model comparison table, plots, and Optuna results, to Vikas Aihole for the report.

## Pre-lunch checkpoint (~12:45 PM)
Baseline training script running on the candidate dataset, showing initial
Macro F1 results across **2–3** models (e.g. Logistic Regression vs Random
Forest) — doesn't need to be the full 5-model suite or HMM yet, just prove the
pipeline runs end-to-end on real candidate features.
