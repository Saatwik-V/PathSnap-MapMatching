# PathSnap — Final Review Viva & Grill Defense Guide
## Comprehensive Master Manual for Defending Phase 2 & Final Review
**Course:** UE24CS352A — Machine Learning | **Department:** CSE, PES University  
**Team:** Saatwik (Lead / ML & HMM), Uttam (Frontend UI), Vikas Jadar (Data / EDA), Vikas Aihole (Backend / Report)

---

## 1. Zero Hardcoding: The Truth Behind Our Numbers

> **Examiner Grill:** *"Did you hardcode these metrics or answers anywhere in your code or API response?"*

### The Ironclad Defense:
**"Absolutely not. Every single number is computed dynamically at runtime from raw trajectory coordinates and topological graph traversal."**

- **Live Dynamic Execution Proof:**
  1. When Laptop 2 makes a request `POST /predict/{trip_id}`, Laptop 1 reads the raw files (`.track`, `.nodes`, `.arcs`, `.route`) directly from disk.
  2. A `cKDTree` dynamically indexes road arc midpoints in metric space and retrieves the $k=10$ nearest candidate road segments per observation in $\mathcal{O}(\log M)$ time.
  3. The 12-feature vectorized pipeline (`feature_engineering/extract_features.py`) calculates perpendicular distances, projection ratios, heading cosines, velocities, and traversal times dynamically.
  4. The trained model (`results/best_tuned_model.joblib`) performs dynamic vector matrix / tree bin evaluations via `predict_proba(X)`.
  5. The Viterbi recurrence algorithm runs dynamic programming in log-space over the directed graph $G = (V, E)$, calculating shortest Dijkstra paths in real time.
  6. Server-side ground-truth evaluation dynamically compares predicted arc IDs against the private server-side `.route` file.
  7. Latency is measured using Python's hardware clock `time.perf_counter()`.
- **Zero Leakage:** The ground-truth `.route` file is read only on Laptop 1, server-side. For evaluator-partition trips the API returns **no reference-route geometry at all** (only metrics, the predicted route and raw GPS), and Laptop 2 holds no `.route` files for them, so the UI cannot draw or see the answer. (Reference geometry is returned only for `student_test` trips, used for the report's maps.)

---

## 2. The Route Continuity Breakthrough: How It Shot Up to 99.1%

> **Examiner Grill:** *"Why was raw classifier route continuity only ~76.8%, and how did you get it up to 90.4% and then 99.1%? Explain the exact mathematics."*

### Step 1: Why the Raw Classifier Fails (Continuity = 76.82%)
- A standard supervised classifier predicts the most likely road segment for each GPS point independently:
  $$\hat{a}_t = \arg\max_{a_i \in \text{Cands}(t)} P(y=1 \mid \mathbf{x}_{t,i})$$
- Because vehicle GPS exhibits satellite multipath noise, shadow drift, and jitter, the nearest spatial segment frequently flickers between parallel streets or cross-cutting alleys. Two consecutive timesteps ($t$ and $t+1$) are classified onto road segments that **do not physically connect in the graph**, resulting in a jagged, teleporting trajectory with an average continuity of only **76.82%**.

### Step 2: The Hidden Markov Model & Viterbi Decoding (Continuity = 90.44%)
- We formulate map-matching as a sequential state-space model:
  - **Hidden States:** The $k=10$ candidate road segments $a_{t,i}$ at each timestep $t$.
  - **Observations:** The sequence of noisy GPS points $\mathbf{p}_1, \dots, \mathbf{p}_T$.
  - **Emission Probability $P(O_t \mid a_i)$:** Combines the machine learning classifier's predicted probability with orthogonal Gaussian spatial distance:
    $$P(O_t \mid a_i) \propto P_{\text{model}}(y=1 \mid \mathbf{x}_{t,i}) \cdot \exp\left(-\frac{\text{perp\_dist}_i^2}{2\sigma^2}\right)$$
    with $\sigma = 25\text{ m}$.
  - **Transition Probability $P(a_{t+1} \mid a_t)$:** Penalizes transitions whose graph shortest-path distance $d_G$ deviates from the great-circle Euclidean GPS displacement $d_{\text{gc}}$:
    $$P(a_{t+1} \mid a_t) \propto \exp\left(-\frac{|d_G(a_t, a_{t+1}) - d_{\text{gc}}(\mathbf{p}_t, \mathbf{p}_{t+1})|}{\beta}\right)$$
    with $\beta = 50\text{ m}$. If two arcs cannot connect within 400m, $d_G = \infty$ and $P = 10^{-6}$.
  - **Viterbi Recurrence:** Solves for the globally optimal trajectory sequence in log-space:
    $$V_t(j) = \max_i \Big(V_{t-1}(i) + \log P(a_j \mid a_i)\Big) + \log P(O_t \mid a_j)$$
- **Impact:** Eliminates independent flickers and boosts route continuity to **90.44%**.

### Step 3: Deterministic Shortest-Path Topological Repair (Continuity = 99.01%)
- **Why gaps still remained:** In dense urban intersections or complex cloverleafs, if a vehicle cruises through a junction, an observation's emission score can occasionally outweigh the transition penalty, leaving a 1-arc topological disconnect.
- **The Shortest-Path Repair:** For any two consecutive decoded arcs $A$ and $B$ where $v_A \neq u_B$, the algorithm queries the pre-computed Dijkstra distance cache within a bounded 400m radius (`GRAPH_SEARCH_CUTOFF_M = 400.0`). If a valid path exists, it splices the intermediate connecting graph arcs into the trajectory. If no path exists within 400m (e.g. GPS tunnel dropout), it is left untouched as a genuine network break.
- **Result:** Route continuity shoots up to **99.01% on student-test** and **99.1% on evaluator-test**, with multiple trips reaching **100.0% perfect continuity**.

---

## 3. The Decoupling Architecture: Why Pointwise Accuracy Did Not Collapse

> **Examiner Grill:** *"If you spliced extra road arcs into the route to get 99% continuity, didn't you mess up your pointwise accuracy?"*

### The Brilliant Architectural Answer:
**"No, because we strictly decoupled pointwise classification evaluation from sequence path reconstruction."**

- **Pointwise Accuracy Array (`point_pred_arcs`):** Contains exactly $T$ elements (one prediction per 1 Hz GPS observation). It evaluates how accurately the classifier and HMM assign each GPS ping to a reference road arc.
- **Repaired Route Sequence (`seq_pred_arcs`):** Contains the continuous, traversable route including spliced bridge arcs. It is used strictly for **Route Continuity %** and drawing the red dashed polyline on the Folium map.
- If we had injected bridge arcs into the pointwise array, length mismatch would have desynchronized the 1 Hz timestamps and cratered accuracy. By decoupling them:
  - **Pointwise Top-1 Accuracy stayed high**: **88.86%** on student-test and **86.3%** on evaluator-test.
  - **Route Continuity hit the target**: **99.1%**.

---

## 4. Defending the Models: Why MLP Won CV, but HistGradientBoosting Was Deployed

> **Examiner Grill:** *"Your 5-model table shows MLP Classifier had the highest CV Macro F1 (0.7952 vs 0.7908). Why did you tune and deploy HistGradientBoosting into the HMM instead of MLP?"*

### The 4-Pillar Engineering Justification:

| Evaluation Dimension | Multi-Layer Perceptron (MLP) | HistGradientBoosting | Winner & Practical Rationale |
|---|---|---|:---:|
| **Cross-Validation Macro F1 (5-fold, untuned)** | **0.7952** | 0.7908 | **MLP** (+0.0044, marginal) |
| **Held-Out Student-Test Top-1** | 84.16% | **84.88%** (tuned: 84.99%) | **HistGB** (+0.72 pt) |
| **Held-Out Student-Test Macro F1** | **0.7935** | 0.7910 (tuned: 0.7937) | **MLP by 0.0025 untuned; tied once HistGB is tuned** |
| **Training time (2.08M rows)** | ~108 s | **~6 s** | **HistGB ~19x faster** |
| **30-trial Bayesian search** | ~1.5 h (estimate: 30 trials x 3 folds x ~60-70 s; **not run**) | **~14 min (835 s, measured)** | **HistGB makes the search practical** |
| **Inference latency per candidate row** | < 0.001 ms | ~0.001 ms | **Tie** (both negligible next to the ~1 s per-trip pipeline latency, dominated by candidate generation and graph search) |
| **Preprocessing** | Requires `StandardScaler` | None needed | **HistGB has one fewer artifact to keep in sync at serving time** |

1. **Honest summary:** the two models are essentially tied on held-out quality. HistGB is ahead on student-test Top-1; MLP is marginally ahead on student-test Macro F1 and CV Macro F1. We did **not** run the MLP on the evaluator partition, so we make no evaluator-test comparison between them.
2. **Computational feasibility (the deciding factor):** the Models and Training Guidelines name generalisation, inference speed and compatibility with the sequential decoder as selection criteria, not CV F1 alone. An MLP fit takes ~108 s on 2.08 million rows, so a 30-trial, 3-fold Optuna search would take on the order of 1.5 hours (our estimate, not measured). HistGradientBoosting fits in ~6 s, so the real search took ~14 minutes.
3. **Result of tuning:** modest. Tuned HistGB reached CV Macro F1 0.797 (3-fold Optuna objective), student-test Top-1 84.99% (+0.11 pt over untuned) and student-test Macro F1 0.7937. Say this plainly; do not claim a large tuning gain.

---

## 5. The 12 Features: Physical and Mathematical Rationale

> **Examiner Grill:** *"Why did you expand from 6 features to 12 features? Explain what physical phenomena each new feature captures."*

| Feature Name | Category | Mathematical Formulation | Physical Phenomena & Map-Matching Purpose |
|---|---|---|---|
| **1. `perp_dist`** | Geometric | $\|\mathbf{p} - \text{proj}(\mathbf{p})\|$ (m) | Orthogonal Euclidean proximity to road segment. |
| **2. `proj_ratio`** | Geometric | $t = \frac{(\mathbf{p} - \mathbf{u}) \cdot (\mathbf{v} - \mathbf{u})}{\|\mathbf{v} - \mathbf{u}\|^2} \in [0, 1]$ | Detects whether the vehicle is between junction nodes or beyond segment endpoints. |
| **3. `heading_error`** | Kinematic | $\|\theta_{\text{trajectory}} - \theta_{\text{arc}}\| \in [0^\circ, 180^\circ]$ | Detects parallel vs cross-cutting streets. |
| **4. `heading_cos`** *(New)* | Non-linear Kinematic | $\cos(\text{radians}(\text{heading\_error})) \in [-1, +1]$ | **Eliminates decision tree step-cliffs.** Smoothly maps collinear travel to $+1$, orthogonal turns to $0$, and opposing wrong-way lanes to $-1$. |
| **5. `arc_length`** | Geometric | $\|\mathbf{v} - \mathbf{u}\|$ (m) | Road segment physical length. |
| **6. `dist_to_start`** | Geometric | $\|\mathbf{p} - \mathbf{u}\|$ (m) | Proximity to starting junction node $u$. |
| **7. `dist_to_end`** | Geometric | $\|\mathbf{p} - \mathbf{v}\|$ (m) | Proximity to ending junction node $v$. |
| **8. `speed_mps`** *(New)* | Kinematic | $\frac{\sqrt{\Delta x^2 + \Delta y^2}}{\max(\Delta t, 0.1\text{s})}$ (m/s) | **Mode & Class Discrimination.** Prevents 80 km/h vehicles from snapping to parallel 5 km/h pedestrian alleys or service lanes. |
| **9. `delta_t`** *(New)* | Temporal | $t_i - t_{i-1}$ (seconds) | **GPS Dropout Detection.** Identifies tunnels and sensor packet loss ($\Delta t > 1\text{s}$), allowing the model to widen candidate uncertainty. |
| **10. `candidate_rank`** *(New)* | Topological | $k \in \{0, \dots, 9\}$ | Provides an explicit spatial proximity prior to tree splitters. |
| **11. `dist_diff_endpoints`** *(New)* | Geometric | $\|\text{dist\_to\_start} - \text{dist\_to\_end}\|$ | Midpoint vs intersection indicator (near 0 when cruising in block center). |
| **12. `traversal_time_est`** *(New)* | Spatio-temporal | $\frac{\text{arc\_length}}{\max(\text{speed\_mps}, 0.5\text{ m/s})}$ | **Kinematic Feasibility.** Evaluates whether vehicle velocity physically permits traversing this arc in the elapsed timeframe. |

---

## 6. Defending Evaluator-Test Scores & The Outlier Trips

> **Examiner Grill:** *"Why did Trip `00000015` only get 62.4% Top-1 accuracy while its route continuity is 100%? And why did Trip `00000023` get 73.8%?"*

Every number below is reproducible: `results/evaluator_predemo_run.json` (scores) and `results/direction_error_analysis.csv` (diagnostics, produced by `model_evaluation/analyze_direction_errors.py`).

### Overall scorecard context
- **10 evaluator trips, mean:** 86.3% Top-1, 99.1% route continuity, ~1,012 ms latency. (Student-test mean Top-1 was 88.9%, so the system generalises.)
- **7 of 10 trips are above 88% Top-1** (`00000073` 95.2%, `00000067` 95.0%, `00000049` 93.9%, `00000017` 91.8%, `00000059` 91.8%, `00000094` 89.3%, `00000087` 88.2%).

### The key idea: continuity is not accuracy
**Route Continuity** only asks whether consecutive predicted arcs connect in the road graph (a drivable path). **Top-1** asks whether each GPS point's predicted arc is actually on the reference route. A path can be perfectly connected and still run along the wrong arcs. Never present continuity as a proxy for correctness.

### Defending Trip `00000015` (62.4% Top-1, 100.0% continuity)
1. **Largest trip:** 6,854 GPS points (about 1.9 hours at 1 Hz) on a 136-arc reference route, so one systematic mistake costs a lot.
2. **What the numbers say:** reference-arc recall is **99.26%** (135 of 136 route arcs appear in our predicted set) but predicted-set precision is only **36.3%**: we predict many arcs that are not on the route.
3. **Measured cause:** **80% of the 2,580 wrong points are the opposite-direction twin of a route arc.** The decoder found the right street but travelled it the wrong way. Ignoring direction, Top-1 would be **92.5%**. The errors are **short bursts, not a sustained wrong-way drive**: 1,355 separate runs with a median length of 2 points (longest 75; only 16% of wrong points sit in runs of 5 or more). The decoder flips point by point between an arc and its reverse twin. The twin shares both endpoint nodes with the arc, so the path stays connected, which is why continuity stays at 100%.
4. **Why (our hypothesis, not a tested result):** the two directions of a street overlap almost exactly, so the classifier scores them alike, and in our HMM transition term moving to the reverse twin has graph distance 0, exactly like staying on the same arc, so both are penalised identically and nothing discourages the flip. (That this is the cause is our hypothesis; that the flips happen is measured.)
5. **It is systematic, not a one-off:** the weakest student-test trip (`00000077`) has the same signature (76.3% strict Top-1, 91.6% ignoring direction, 65% of errors are reverse twins).
6. **What we would do next:** add a heading / direction-of-travel consistency term to the HMM transition. We did not, because it was identified after the decoder was frozen and would require re-running the benchmark, tuning and report.
7. **Do not say** that the reference route "skipped connector links" or that connectivity repair caused the errors: the evidence does not support either. Repair never changes the per-point assignment (see Section 3).

### Defending Trip `00000023` (73.8% Top-1, 99.3% continuity)
1. **Short trip:** 458 GPS points, the shortest of the 10 evaluator trips (not of the whole dataset: a training trip has 361 points).
2. **Different failure from `00000015`:** only 2.5% of its errors are direction flips. Instead, **18.3% of its points have none of the 10 nearest candidate arcs on the reference route** (the other evaluator trips are at 1% to 4.5%). Where the true arc is not among the candidates, no decoder can be right.
3. **Likely reasons (consistent with the data, not proven):** GPS far from the reference route's arcs, through noise or drift, or an imperfect reference route (the Dataset Exploration PDF explicitly warns the reference is not error-free).
4. **Sustained, not flickering:** unlike `00000015`, its errors come in longer runs (median 4 points; 83% of wrong points are in runs of 5 or more), which matches stretches where the GPS is far from the reference route.
5. **Start-up effect, partly:** the error rate is 51.7% in the first 60 points versus 22.4% afterwards, which is consistent with initial GPS drift, but most of the no-candidate points occur later in the trip, so drift at the start explains only part of the gap.

---

## 7. Crucial ML Hygiene Defenses

### Q: Why didn't you use standard K-Fold Cross-Validation?
**A:** "Standard K-Fold randomly shuffles observations, which severely violates the independent and identically distributed (i.i.d.) assumption due to **trajectory temporal and spatial autocorrelation**. Point $t+1$ is physically adjacent to point $t$. Randomly splitting points would train on point $t$ and test on point $t+1$, resulting in massive data leakage and artificially inflated scores. We enforced **`GroupKFold(n_splits=5, groups=trip_id)`**, guaranteeing that no trip's observations ever crossed train and validation partitions."

### Q: Why didn't you use a single UTM projection for metric distances?
**A:** "The PathSnap dataset covers trajectories across multiple global regions (Germany lat ~49.8°, Russia lat ~56.2°, USA lat ~34.7°, Japan lat ~35.6°). A single fixed UTM zone or global Cartesian plane introduces massive geographic distortion outside its 6° longitudinal strip. Instead, we implemented **local metric equirectangular tangent plane projections** dynamically centered at each trip's centroid $(\lambda_0, \phi_0)$:
$$\Delta x = (\lambda - \lambda_0) \cdot \frac{\pi}{180} \cdot R_{\text{earth}} \cdot \cos(\phi_0), \quad \Delta y = (\phi - \phi_0) \cdot \frac{\pi}{180} \cdot R_{\text{earth}}$$
guaranteeing sub-meter Euclidean accuracy (<0.1% distortion) globally."

### Q: Why did you evaluate and reject Probability Calibration (Isotonic Regression)?
**A:** "We trained `CalibratedClassifierCV(method='isotonic')` to evaluate Platt scaling and isotonic regression. Calibration improved log loss (0.4719 to 0.4539) and Brier score (0.1411 to 0.1399), but **no accuracy metric improved**: student-test Top-1 went from 84.99% to 84.94%, Macro F1 from 0.7937 to 0.7898, and HMM pointwise accuracy from 88.86% to 88.81%. It also grew the saved model from about **1.4 MB to about 88 MB** (roughly 60x), because `CalibratedClassifierCV` keeps one calibrated model per fold. We rejected a 60x size penalty for no accuracy gain."

---

## 8. Known Limitations (say these before the examiner finds them)

1. **Direction ambiguity** is our largest identified error source (Section 6). Not fixed; the fix is a direction-aware HMM transition.
2. **Top-1 is set membership**: the predicted arc must be anywhere on the reference route's arc set; it ignores visit order and counts the reverse twin of a route arc as wrong.
3. **Continuity is boosted by the repair step by construction.** Always quote it with Top-1, and quote the HMM-only value (90.4%) too; the HMM alone does not reach 95%.
4. **Small tuning and calibration gains.** All five model families land within about 0.04 student-test Macro F1 of each other, which points at the feature set, not the algorithm, as the ceiling.
5. **Per-trip variance is large** (evaluator Top-1 from 62.4% to 95.2%), so the 10-trip mean should always be read with that spread.
