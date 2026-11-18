# PathSnap — Team Work Breakdown

Master index for the team's division of labor. Each teammate has a dedicated,
in-depth file; this file is the shared map of how the pieces connect, the
hard constraints everyone must respect, and the two checkpoints (pre-lunch,
final demo).

## Team & files
| Teammate | Role | File |
|---|---|---|
| Saatwik | Chief ML Architect & Sequential Decoding Lead | [saatwik.md](./saatwik.md) |
| Uttam | Frontend UI & Interactive Visualization Lead | [uttam.md](./uttam.md) |
| Vikas Jadar | Data Pipeline & Feature Extraction Specialist | [vikas_jadar.md](./vikas_jadar.md) |
| Vikas Aihole | FastAPI Backend & Deployment/Report Lead | [vikas_aihole.md](./vikas_aihole.md) |

## How the pipeline connects end-to-end

```
Raw .track/.nodes/.arcs/.route files
            │
            ▼
   Vikas Jadar (data_processing/, feature_engineering/)
   EDA → candidate generation → 6 geometric features → labelled dataset
            │
            ▼  results/train_features.csv, results/test_features.csv
   Saatwik (model_training/, model_evaluation/, hyperparameter_optimisation/, hmm_viterbi/)
   5 models → group-aware comparison → Bayesian tuning → HMM/Viterbi decoder
            │
            ▼  results/best_model.joblib + reconstruct_route() function
   Vikas Aihole (api/)
   FastAPI loads model + HMM + Vikas Jadar's feature code → serves /predict/{trip_id}
            │
            ▼  JSON: predicted route, Top-1 Accuracy, Macro F1, Route Continuity %, latency
   Uttam (ui/, visualisations/)
   Streamlit UI (Laptop 2) → calls API (Laptop 1) → renders Folium map + scorecards
```

Vikas Aihole also sits at the side as report/deployment coordinator, pulling
artifacts from all three other roles into the final PDF and the two-laptop
network setup.

## Repo folder ownership
- `data_processing/`, `feature_engineering/` — Vikas Jadar
- `model_training/`, `model_evaluation/`, `hyperparameter_optimisation/`, `hmm_viterbi/` — Saatwik
- `api/`, `deployment/`, `report/` — Vikas Aihole
- `ui/`, `visualisations/` — Uttam
- `results/` — shared output location; subfolders/files are written by whoever produces them (features by Vikas Jadar, model artifacts/plots by Saatwik, EDA plots also by Vikas Jadar), consumed by the others.
- `requirements.txt`, root `README.md` — Vikas Aihole maintains, everyone adds their own deps.

## Hard constraints (apply to everyone, not just one role)
- **No evaluator-testing data exists on your side at all** (it's held by the
  evaluators) — but the same leakage discipline applies to the data you do
  have: never let student-test trips influence training, and use
  `GroupKFold`/group-aware splits everywhere so no single trip's points cross
  a train/validation boundary.
- **Minimum 5 trained models from genuinely different families** before
  final selection — this is graded explicitly and is non-negotiable.
- **No hardcoded predictions or bypassing the API** anywhere in the final
  repo — it will be checked. Mock responses are fine only at the pre-lunch
  checkpoint.
- **Ground truth stays server-side**, inside Vikas Aihole's API, and is never
  exposed to the model's feature pipeline or to the UI as an input.
- **Independent work** — don't share code, model weights, or configs with
  other teams; submissions are compared across teams for similarity.
- Every team member must be able to explain every stage (data, models, HPO,
  HMM, API, UI) in a viva — don't let roles become silos; do a short
  cross-team walkthrough before submission.

## Pre-lunch checkpoint (~12:45 PM) — combined view
- [ ] GitHub repo initialized, public, clean commit history.
- [ ] Vikas Jadar: 2 EDA plots (trajectory lengths + sampling gaps) + candidate/feature extraction working on 5–10 trips.
- [ ] Saatwik: baseline training script on the candidate dataset, Macro F1 across 2–3 models (e.g. Logistic Regression vs Random Forest).
- [ ] Uttam: Streamlit UI showing the trip list + an initial Folium map for trip `00000001`.
- [ ] Vikas Aihole: FastAPI server running locally, `GET /trips` and `POST /predict/00000001` returning mock predictions.

## Final deliverables (per Deliverables.pdf)
1. **Public GitHub repository** — all source code (data processing, features,
   training, evaluation, HPO, HMM/Viterbi, API, UI, deployment config,
   visualisation code, logs/results), organized folders, root README with
   install/run/deploy instructions, no secrets.
2. **One consolidated technical report PDF** (Vikas Aihole coordinates) —
   EDA, 5-model comparison + per-model Q&A, model visualisations, Bayesian
   HPO section, HMM/Viterbi section with classifier-only vs classifier+HMM
   comparison, Folium geographical visualisation, API/UI screenshots.

## Final evaluation flow (what actually happens on the day)
1. API runs on Laptop 1, UI runs on Laptop 2, both on the same network.
2. Evaluator opens the UI, picks one of the 10 reserved trips.
3. UI sends the trip ID to the API; API runs the full pipeline (candidate gen
   → model → HMM/Viterbi) and computes accuracy against its private `.route`.
4. UI displays the predicted route (Folium map) and metrics.
5. Evaluator can repeat for any of the other 9 trips — the system must work
   for all ten, not just a rehearsed one, and the API terminal logs must be
   visible throughout.
