# Uttam — Frontend UI & Interactive Visualization Lead

## Mission
Build the evaluator-facing portal that runs on **Laptop 2** during the final
demo. The evaluator will sit in front of your screen, pick one of the 10
reserved evaluation trips from a dropdown, click a button, and your UI must
call Laptop 1's API and render the result as a clean interactive Folium map
plus scorecards. This is the part the faculty actually touches — it needs to
be robust (handle a slow/failed API call gracefully) as much as it needs to
look good.

## Folders you own
- `ui/app.py` — the Streamlit app.
- `visualisations/folium_map.py` — the reusable Folium map-building component.

## Inputs you depend on
- Vikas Aihole's FastAPI contract: `GET /trips`, `POST /predict/{trip_id}`,
  `GET /health`. Agree on the exact JSON response shape with him early —
  you need it to know what fields to pull into your scorecards and map layers.
- The raw trip files (`.track`, `.route`, `.nodes`, `.arcs`) for whichever
  trip is selected, to draw the "ground truth" and "raw GPS" layers — either
  the API returns these coordinates in its response, or you read them
  directly from the local `Student folder/` copy on Laptop 2. Decide this with
  Vikas Aihole (simplest: have the API return everything the map needs, so the
  UI stays a thin client and the single source of truth is the server).

## End-to-end task list

### 1. Streamlit Evaluator UI (`ui/app.py`)
Run with: `streamlit run ui/app.py`

Required elements:
- **API address input box** — text field for Laptop 1's IP, e.g.
  `http://192.168.x.x:8000`, defaulting to something sensible for local
  testing (`http://localhost:8000`). Store it in `st.session_state` so it
  persists across reruns.
- **Trip dropdown** — populate from `GET /trips` (the 10 test trip IDs, e.g.
  `00000000`, `00000007`, ...). Call this on app load / on a "connect" action,
  and handle the case where the API isn't reachable yet (show a clear error,
  don't crash).
- **"Snap Route & Evaluate" button** — on click, `POST` the selected trip ID
  to `/predict/{trip_id}` on Laptop 1, show a spinner while waiting, and
  handle timeouts/errors with a visible message (the demo will be live in
  front of evaluators — a silent hang is the worst failure mode here).
- **Metrics scorecards** — once the response comes back, display:
  Top-1 Accuracy, Macro F1, Route Continuity %, Inference Latency. Use
  `st.metric()` for a clean scorecard look, laid out in columns.
- Render the map (see below) beneath the scorecards.

### 2. Interactive Folium Map Component (`visualisations/folium_map.py`)
Write this as a standalone function, e.g.
`build_map(raw_gps_points, candidate_arcs, reference_route, predicted_route) -> folium.Map`,
so it's testable outside Streamlit and reusable in the technical report
notebook too (Vikas Aihole will want this code committed and referenced in the
report per the Deliverables doc — "include the map-generation code in the
repository").

Embed it in Streamlit via `streamlit-folium` (`st_folium(m)`).

Required layers (as separate `folium.FeatureGroup`s so they can be toggled):
- **Layer 1 — Blue dots**: raw GPS points (`.track`), each with a popup
  showing its timestamp.
- **Layer 2 — Grey lines**: the relevant bounding road-network arcs (don't
  render the *entire* national graph — just the arcs near this trip's bounding
  box, or you'll get an unreadable/slow map).
- **Layer 3 — Green bold line**: the ground-truth reference route
  (from `.route`, resolved to arc geometry via `.nodes`/`.arcs`).
- **Layer 4 — Red dashed line**: PathSnap's reconstructed route (from the API
  response).
- Add `folium.LayerControl()` so the evaluator can toggle layers to visually
  compare how closely the red dashed line snapped onto the green line.
- Set map center/zoom from the trip's bounding box so it opens already
  framed on the trip — don't make the evaluator manually zoom in.
- Add a legend (Folium doesn't have this built in — use a small HTML/CSS
  overlay via `folium.Element` or `MacroElement`, or at minimum clear labels in
  the layer control names) — this is explicitly required in the Deliverables doc.

### 3. Robustness for the live demo
- The guidelines require the API logs to be visibly shown to the evaluator on
  Laptop 1's terminal, and require the system to work for *all ten* reserved
  trips, not just one — so test your UI against every one of the 10 trip IDs
  before the demo, not just the one you built it against.
- Never hardcode a response or bypass the API call — the policy docs
  explicitly forbid this and repositories will be checked.

## What you need from teammates
- **From Vikas Aihole**: the exact JSON schema of `/predict/{trip_id}`'s
  response (field names for accuracy/F1/continuity/latency, and whatever
  coordinate data you need for the map layers), plus confirmation of the
  `/trips` response shape.
- Decide early whether raw GPS/route/network geometry for the map comes from
  the API response or from local files on Laptop 2 — don't build both.

## What you hand off
- A working Streamlit app + Folium component, demoed against the live API
  before the final evaluation.
- Screenshots of the working UI (all states: trip selected, metrics shown, map
  rendered) for Vikas Aihole's report section 7.

## Pre-lunch checkpoint (~12:45 PM)
Working Streamlit UI that displays the trip list (even against a mocked
`/trips` response from Vikas Aihole) and renders an initial Folium map for
trip `00000001` — the map doesn't need the predicted-route layer yet, just
prove raw GPS + road network + reference route render correctly.
