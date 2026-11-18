# Vikas Jadar — Data Pipeline & Feature Extraction Specialist

## Mission
You are the foundation of the whole project: everyone else's work depends on
the quality of the feature tables you produce. You turn raw `.track` /
`.nodes` / `.arcs` / `.route` files into a clean, labelled candidate dataset
that Saatwik can train five models on directly.

## Folders you own
- `data_processing/eda.py`
- `data_processing/candidate_gen.py`
- `feature_engineering/extract_features.py`
- `results/` (writing `train_features.csv`, `test_features.csv`, and EDA plots to `results/plots/`)

## Dataset facts to work from (confirmed by inspecting the actual files)
- `Student folder/train/` has 80 trip folders, `Student folder/student_test/`
  has 10. Each trip folder `NNNNNNNN/` contains `NNNNNNNN.track`,
  `.nodes`, `.arcs`, `.route`.
- `.track`: whitespace-separated, **no header**, 3 columns = `lon, lat, timestamp(s)`. ~1 Hz, ~97% of steps exactly 1s apart (verify this per-trip, don't assume).
- `.nodes`: whitespace-separated, no header, 2 columns = `lon, lat`. Row index (0-based) is the node ID.
- `.arcs`: whitespace-separated, no header, 2 integer columns = `from_node_id, to_node_id`. Row index (0-based) is the **arc_id** — this is the ID you need everywhere else (candidate labels, route sequences, HMM states).
- `.route`: one integer per line = the ground-truth **sequence of arc_ids** traversed, in chronological order. Confirmed by inspection: this is a sequence, not a per-GPS-point label — you have to figure out which arc in that sequence corresponds to which GPS timestep (there is no explicit timestamp alignment given).
- **Critical warning from the Dataset Exploration doc**: do not assume a row in a GPS file maps directly to a row in a road-network file. Verify column order (lon vs lat — double check against the dataset doc's stated order, since swapped lon/lat is a classic silent bug), and don't assume all trips have identical structure.

## End-to-end task list

### 1. Exploratory Data Analysis (`data_processing/eda.py`)
Run against `Student folder/train/` (and separately note any differences in
`student_test/`, without mixing the partitions).

Compute and print/save:
- Number of trips, number of GPS points per trip (distribution).
- Timestamp gaps: mean Δt (should be ~1.0s), max gap per trip, how many steps
  deviate from exactly 1s.
- Coordinate bounding box per trip and overall (Notice: trips span multiple global regions including Germany lat≈49.8/lon≈8.6, Russia lat≈56.2/lon≈55.0, and USA lat≈34.7/lon≈-86.6. Calculate meter projections dynamically per trip using its local mean latitude, never hardcode a single region!).
- Missing/invalid values (NaNs, out-of-range coordinates, zero-length records) — report "zero found" only if you actually checked, not by assumption.
- Road network size per trip: node count, arc count, isolated nodes /
  disconnected components (use `networkx` to check connectivity).
- `.route` length vs `.track` length per trip — note the ratio, since this
  tells you how many GPS points roughly correspond to each route arc.

Required plots (save to `results/plots/`):
1. Distribution of sampling intervals (Δt) across all trips.
2. Trajectory length histogram (points per trip).
3. Coordinate scatter plot(s) — raw GPS points, to visually sanity-check
   geographic plausibility.

Keep a short written log (can live at the top of the script or in a
`results/data_quality_notes.md`) of every data-quality issue and assumption
you make — the report explicitly requires this ("a short record of
data-quality issues, assumptions and decisions") and you'll need it again
for the report's EDA section.

### 2. Candidate Generation & Ground-Truth Matcher (`data_processing/candidate_gen.py`)
For every GPS point in every trip's `.track`:
1. Compute the distance (in **meters** — use a geodesic calculation or project
   to a local metric CRS; do not use raw lon/lat Euclidean distance, per the
   Dataset Exploration doc's explicit warning) from the GPS point to every
   nearby road segment (arc), where an arc's geometry is the line between its
   two `.nodes` endpoints.
2. Narrow this to the **nearest 5–10 candidate arcs** per GPS point (e.g. via
   a spatial index — `scipy.spatial.cKDTree` or similar on segment midpoints,
   then refine with true point-to-segment distance — don't brute-force every
   arc against every point for performance).
3. Label each candidate: `y=1` if that arc appears in the ground-truth
   `.route` within the appropriate time window for that point, else `y=0`.
   You'll need to decide/document how you map a `.route` arc_id sequence to a
   GPS timestep — e.g. by nearest-point-on-route-polyline or by assuming
   roughly even time distribution along the route — **document this
   assumption explicitly**, since the Dataset doc warns the supplied route is
   reference information, not infallible ground truth.
4. Track and report the **Candidate Miss Rate** input for Saatwik: how often
   the true reference arc isn't even among your generated candidates — this
   caps the ceiling of every downstream model, so get this as low as
   reasonably possible before handing off.

### 3. Feature Extraction (`feature_engineering/extract_features.py`)
For every (GPS point, candidate arc) pair, compute 6 geometric features:
- `perp_dist` — orthogonal distance from the GPS point to the road segment
  line, in meters.
- `proj_ratio` — where the GPS point's projection falls along the segment,
  normalized to `t ∈ [0, 1]` (0 = at `from_node`, 1 = at `to_node`).
- `heading_error` — absolute difference between the vehicle's instantaneous
  movement heading (derived from consecutive GPS points) and the segment's
  direction, in degrees, `0°–180°`.
- `arc_length` — length of the candidate road segment, in meters.
- `dist_to_start` / `dist_to_end` — distance from the GPS point to each of the
  segment's endpoints, in meters.

Output: `results/train_features.csv` (from the 80 training trips) and
`results/test_features.csv` (from the 10 student-test trips) — **never mix
trips across these files**, and keep a `trip_id` column plus `arc_id` and the
label `y` in both, since Saatwik needs `trip_id` for `GroupKFold` and `arc_id`
to reconstruct sequences for the HMM stage. Confirm this exact column list
with Saatwik before he starts building on it.

## What you need from teammates
Nothing blocking — you're first in the pipeline. Get your candidate+feature
schema finalized and communicated to Saatwik early, since any late column
rename breaks his training scripts.

## What you hand off
- `results/train_features.csv`, `results/test_features.csv` → Saatwik.
- EDA plots + data-quality notes → Vikas Aihole, for the report's EDA section.
- Your candidate-generation logic (nearest-arc search) is also reusable by
  Uttam if the map needs to show "bounding road-network arcs" near a trip.

## Pre-lunch checkpoint (~12:45 PM)
2 EDA plots (trajectory lengths + sampling time gaps) committed, and a working
candidate + feature extraction pipeline run on **5–10 trips** (not necessarily
all 80 yet) to prove the geometry/labelling logic is correct before scaling up.
