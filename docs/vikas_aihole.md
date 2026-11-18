# Vikas Aihole — FastAPI Backend & Deployment/Report Lead

## Mission
You build the server that actually runs on **Laptop 1** during the live demo,
wire up the two-laptop network, and are the coordinator who assembles
everyone's work into the single mandatory technical report PDF. You are the
integration point: the API is where Saatwik's model+HMM, Vikas Jadar's
candidate pipeline, and the evaluator ground truth all come together — and
this is also the piece that must never leak ground truth to the UI or model.

## Folders you own
- `api/main.py`
- `deployment/setup_network.md`
- `report/` (the final PDF and its source material)

## Inputs you depend on
- Saatwik's `results/best_model.joblib` + preprocessing artifact + his
  `hmm_viterbi` route-reconstruction function.
- Vikas Jadar's candidate-generation / feature-extraction code (the API needs
  to run the *same* pipeline at inference time that produced the training
  features — any mismatch between training-time and serving-time feature
  computation silently breaks everything).
- The reserved evaluation trips' `.route` files, kept server-side only, for
  computing accuracy metrics after the fact.

## End-to-end task list

### 1. FastAPI Backend Service (`api/main.py`)
Run with: `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`

Required endpoints:
- **`GET /trips`** — returns the list of available test trip IDs (the ten
  reserved evaluation trips, e.g. `["00000000", "00000007", ...]`). Decide
  with Uttam the exact JSON shape, e.g. `{"trips": [...]}`.
- **`POST /predict/{trip_id}`** — the core pipeline call:
  1. Load the trip's `.track`/`.nodes`/`.arcs` (reuse Vikas Jadar's
     candidate-generation + feature-extraction code — don't reimplement it).
  2. Run Saatwik's trained model to score candidates, then his HMM/Viterbi
     decoder to get the final predicted arc sequence.
  3. Compare the predicted sequence against the **server-side** `.route` file
     for that trip to compute Top-1 Accuracy and Route Continuity %.
  4. Measure and include inference latency.
  5. Return clean JSON to the UI: predicted route (as coordinates or arc_ids
     the UI can resolve to a polyline), Top-1 Accuracy, Macro F1, Route
     Continuity %, Inference Latency, and whatever raw/reference-route
     geometry Uttam's map needs (agree with him whether the API returns full
     coordinate layers or just IDs).
  6. **Ground truth must never appear anywhere upstream of this endpoint** —
     it's computed here, server-side, and only the resulting *metrics* go out,
     never the raw reference route used as an "answer key" the UI could
     display as if it were an input.
- **`GET /health`** — trivial health check, used by Vikas Aihole himself for
  the network test and by Uttam's UI to show "API connected" status.
- **Logging**: print a clean, readable log line to the terminal on every
  request (trip ID, endpoint, latency, result summary). This is explicitly
  required — evaluators will watch Laptop 1's terminal live during the demo
  as proof the pipeline is really running and not hardcoded.

### 2. Two-Laptop Network Setup (`deployment/setup_network.md`)
Write this as a short runnable checklist, test it yourself before the demo:
1. Connect both laptops to the same Wi-Fi network or a phone hotspot.
2. On Laptop 1, find its IP: `ipconfig` (Windows) or `ifconfig` / `ip a`
   (macOS/Linux) — note it's the LAN IP, not `127.0.0.1`.
3. Start the API on Laptop 1 bound to `0.0.0.0` (not `127.0.0.1`, or Laptop 2
   can't reach it).
4. From Laptop 2, verify with `curl http://<laptop1_ip>:8000/health`.
5. Note any firewall prompts (macOS/Windows will ask to allow incoming
   connections the first time `uvicorn` binds — approve it ahead of the demo,
   not during it).
6. Document the fallback plan if Wi-Fi is unreliable on the day (e.g. hotspot
   from one of the laptops, or direct Ethernet/USB tethering if available).

### 3. Technical Report Coordinator (`report/`)
You own assembling the **one** mandatory consolidated PDF (never submit
per-model reports). Follow the exact structure from `Deliverables.pdf`:

1. **Dataset Understanding & EDA** — pull in Vikas Jadar's EDA plots, data
   quality notes, and partition-labelled analysis (clearly state which
   partition each analysis used).
2. **Model Experimentation & Comparison** — Saatwik's 5-model comparison
   table (exact columns: Model, Algorithm Family, Key Configuration, CV Macro
   F1, Top-1 Accuracy, Route Continuity, Student-Test Macro F1, Log Loss,
   Inference Time) plus a separate subsection per model answering the
   compulsory questions (why suitable, strengths/limitations, hyperparameters,
   loss/objective, student-test performance, over/underfitting evidence).
3. **Model Visualisations** — CV Macro F1 comparison plot, Top-1
   Accuracy/Macro F1 comparison plot, Log Loss comparison, confusion matrices
   (optional), and model-comparison justification plots. All must come from
   real experiment output — never fabricate a training curve for a model that
   doesn't produce epoch-wise history (e.g. Random Forest, k-NN).
4. **Bayesian Hyperparameter Optimisation** — search space, ranges, objective,
   number of trials, best values, tuned-vs-untuned comparison table, from
   Saatwik's Optuna logs.
5. **HMM & Viterbi Route Reconstruction** — hidden-state definition, emission
   and transition formulas actually used, how gaps/missing candidates/
   disconnected segments were handled, and the classifier-only vs
   classifier+HMM comparison with real numbers.
6. **Geographical Visualisation (Folium)** — at least one real exported/
   screenshotted map showing raw GPS, road network, reference route, and
   predicted route together, from Uttam's actual map component, with the
   generation code committed to the repo.
7. **API, UI & Deployment** — screenshots of the working Streamlit UI and of
   successful API responses on Laptop 1's terminal (the actual CMD/log
   output, not a mockup).

Also own repo hygiene for submission:
- Public GitHub repo, organized into the folders already created
  (`data_processing/`, `feature_engineering/`, `model_training/`,
  `model_evaluation/`, `hyperparameter_optimisation/`, `hmm_viterbi/`, `api/`,
  `ui/`, `deployment/`, `visualisations/`, `report/`, `results/`).
- A root `README.md` with installation, execution, and deployment
  instructions for the whole system (how to run EDA, train models, start the
  API, start the UI, run the two-laptop demo).
- No secrets/API keys committed; `requirements.txt` kept up to date as
  teammates add dependencies.
- Double-check nothing hardcodes evaluation answers anywhere in the repo —
  this is explicitly checked.

## What you need from teammates
- Saatwik: final model artifact + loading code + the HMM module's call
  signature.
- Vikas Jadar: the exact same feature-extraction/candidate-generation code
  used for training, so the API can call it identically at inference time.
- Uttam: agreed JSON contract for `/trips` and `/predict/{trip_id}`, and his
  UI/map screenshots for the report.
- Collect everyone's plots, tables, and written answers continuously (not at
  the very end) since the report must be verifiable against the actual repo.

## What you hand off
- A live, logged API on Laptop 1 that the whole team can test against.
- The network setup doc, verified working on both laptops before the demo.
- The final assembled PDF report.

## Pre-lunch checkpoint (~12:45 PM)
FastAPI server running locally, `GET /trips` and `POST /predict/00000001`
both returning **mock** predictions (hardcoded or trivially computed stand-ins
are fine at this checkpoint only — they must be replaced with the real
pipeline well before final evaluation, since hardcoded outputs are
explicitly prohibited in the final submission).
