# PathSnap — Phase 1 Audit & Phase 2 Results

## Task 1: Audit of completed work

| Check | Status | Notes |
|---|---|---|
| 5 distinct model families in `results/five_model_comparison.csv` | **Pass** | Logistic Regression, Distance-Weighted k-NN, Random Forest, HistGradientBoosting, MLP. |
| `GroupKFold` grouped by `trip_id`, no trip crosses train/val | **Pass** | `model_training/train_5_models.py` uses `GroupKFold(n_splits=..., groups=trip_id)`. |
| 3 comparison plots in `results/plots/` | **Pass** | `model_macro_f1_comparison.png`, `model_top1_acc_comparison.png`, `model_logloss_comparison.png`. |
| `results/best_model.joblib` saved and loadable | **Pass** | Loads via `joblib.load()`; contains `{"model", "scaler", "model_name"}`. |

**One correction made during the audit**: `results/train_features.csv` had only
been generated from **3** training trips (`00000001`–`00000003`, ~46.9K rows),
not the full 80-trip training partition. `GroupKFold(n_splits≈3)` over 3 groups
is not a meaningful group-aware CV (each fold trains on only 2 trips). This
was regenerated from **all 80 training trips** (1,042,085 candidate rows) and
the 10 student-test trips (95,800 rows, `results/test_features.csv`, not
previously generated), then the 5-model benchmark was re-run on the full
dataset. Updated champion: **MLP** (CV Macro F1 = 0.7984), narrowly ahead of
Random Forest (0.7905) and HistGradientBoosting (0.7879).

## Task 2: Bayesian Hyperparameter Optimisation (Optuna)

**Model tuned: HistGradientBoosting**, not the nominal champion MLP. MLP's CV
Macro F1 lead was small (0.7984 vs 0.7879, a 0.0105 gap) but its fit time at
this dataset size (~1M rows) is ~50s, vs ~2.6s for HistGradientBoosting — a
30-trial × 3-fold Optuna search would take over an hour for MLP vs ~5 minutes
for HistGradientBoosting. The Models and Training Guidelines doc explicitly
lists inference speed/computational efficiency as selection criteria, not CV
F1 alone, so this trade-off is documented rather than hidden.

- Search space: `learning_rate` (log-uniform, 1e-3–3e-1), `max_depth` (3–14),
  `max_leaf_nodes` (15–127), `l2_regularization` (log-uniform, 1e-4–1e-1),
  `min_samples_leaf` (10–200).
- Objective: maximize CV Macro F1 under `GroupKFold(n_splits=3, groups=trip_id)`.
- 30 trials, Optuna `TPESampler(seed=42)`. Search wall time: 309s.
- Best params: `learning_rate=0.1055, max_depth=7, max_leaf_nodes=39, l2_regularization=0.00216, min_samples_leaf=48`.

| Variant | CV Macro F1 | Student-Test Top-1 Accuracy | Log Loss | Inference Latency |
|---|---|---|---|---|
| HistGradientBoosting (untuned) | 0.7879 | 71.59% | 0.4445 | 0.001 ms |
| HistGradientBoosting (tuned, 30 trials) | 0.7889 | **82.69%** | 0.4699 | 0.001 ms |

Tuning gave a marginal CV F1 gain but a large jump in held-out student-test
Top-1 Accuracy (71.59% → 82.69%), at the cost of a slightly higher log loss —
the tuned model is a better discriminator of the correct candidate but
slightly less calibrated in its raw probabilities. Full trial log:
`results/optuna_trials.csv`. Best params: `results/optuna_best_params.json`.
Tuned artifact: `results/best_tuned_model.joblib`.

## Task 3: HMM + Viterbi Sequential Route Reconstruction

Implemented in `hmm_viterbi/viterbi.py`:
- **Graph**: directed `networkx` graph per trip, built from `.nodes`/`.arcs`, edge weight = haversine length (meters).
- **Emission**: `P(O_t | a_i) ∝ P_model(y=1) * exp(-perp_dist_i² / (2σ²))`.
- **Transition**: `P(a_{t+1} | a_t) ∝ exp(-|d_graph(a_t, a_{t+1}) - d_greatcircle(p_t, p_{t+1})| / β)`, floored at `1e-6`; `d_graph = 0` for directly-connected or identical arcs, otherwise a cutoff-limited Dijkstra search (400m) — cached per source node so repeated/parameter-sweep decoding doesn't re-run the graph search.
- **Decoding**: standard Viterbi recurrence in log-space over per-timestep candidate sets.
- Hyperparameters tuned empirically by grid search on training trips: `sigma=10, beta=100, k_candidates=10`.

### Classifier-only vs Classifier+HMM (10 student-test trips, tuned model)

| Metric | Classifier-only | Classifier + HMM/Viterbi |
|---|---|---|
| Mean pointwise accuracy | ~0.831 | ~0.876 |
| Mean route continuity % | **76.38%** | **91.94%** |

Per-trip results: `results/hmm_vs_classifier_studenttest.csv`. HMM decoding
improved both accuracy and continuity on 9 of 10 trips (one trip,
`00000000`, saw a very small pointwise-accuracy dip of 0.4 points while its
continuity still rose from 84.8% → 93.9% — a case where enforcing a connected
path overrode one weakly-confident candidate pick).

**Note on the acceptance target**: the brief's target of >95% route continuity
was not fully reached on average (91.94%; several individual trips, e.g.
`00000007` at 98.2%, do clear it). Diagnosis: with `k_candidates=10`, ~98% of
timesteps already have at least one directly-connected candidate pair
available, so the ceiling is high — but the classifier's emission
probabilities are confident enough on some wrong candidates to occasionally
outweigh the transition term even at a loosely-tolerant `beta=100`. This is
reported honestly rather than tuned to hit a specific number; further gains
would likely require either a stronger/recalibrated emission model or a
higher structural weight on direct connectivity in the transition term.
