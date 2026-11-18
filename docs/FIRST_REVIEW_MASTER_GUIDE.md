# PathSnap: End-to-End System Guide & First Review Master Manual

**UE24CS352A: Machine Learning Hackathon | PES University**  
*Comprehensive System Reference, Theoretical Concepts, Code Walkthrough, and Viva Preparation Guide for Saatwik, Uttam, Vikas Jadar, and Vikas Aihole.*

---

## Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Theoretical Concepts & Mathematical Formulations](#3-theoretical-concepts--mathematical-formulations)
   * [3.1 Coordinate Systems & Dynamic Metric Projections](#31-coordinate-systems--dynamic-metric-projections)
   * [3.2 Spatial Indexing & Candidate Generation](#32-spatial-indexing--candidate-generation)
   * [3.3 The 6 Geometric & Kinematic Features](#33-the-6-geometric--kinematic-features)
   * [3.4 Data Leakage Prevention & GroupKFold](#34-data-leakage-prevention--groupkfold)
   * [3.5 The 5 Evaluated Machine Learning Families](#35-the-5-evaluated-machine-learning-families)
   * [3.6 Hidden Markov Models (HMM) & Viterbi Sequential Decoding](#36-hidden-markov-models-hmm--viterbi-sequential-decoding)
   * [3.7 Evaluation Metrics Decoded](#37-evaluation-metrics-decoded)
4. [Codebase Walkthrough: What Each Script Does](#4-codebase-walkthrough-what-each-script-does)
5. [Actual Experimental Results (Pre-Lunch Checkpoint)](#5-actual-experimental-results-pre-lunch-checkpoint)
6. [First Phase Review Presentation Script (3 Minutes)](#6-first-phase-review-presentation-script-3-minutes)
7. [Comprehensive Viva Q&A by Team Role](#7-comprehensive-viva-qa-by-team-role)

---

## 1. Executive Summary & Problem Statement

### The Real-World Challenge
GPS tracking devices produce noisy location measurements due to:
* **Urban Canyons**: Tall buildings obstruct line-of-sight satellite signals.
* **Multipath Interference**: Radio signals bounce off concrete, glass, and asphalt before reaching the receiver.
* **Low Sampling Frequency (Nominally 1 Hz)**: Gaps between observations allow multiple plausible paths.
* **Signal Dropouts**: Tunnels, underpasses, or sensor latency can cause gaps spanning up to 15+ minutes.

Consequently, a raw GPS trajectory appears to float across rooftops, zigzag onto parallel one-way streets, or make physically impossible teleports.

### The Objective: PathSnap
To build a distributed, real-time Machine Learning and Sequential Route Reconstruction system that:
1. Filters noisy GPS observations.
2. Identifies candidate road segments on a physical road network graph.
3. Scores candidate segments using a supervised ML classifier.
4. Enforces physical topological continuity using a **Hidden Markov Model (HMM) with Viterbi decoding**.
5. Serves predictions in real time via a **two-laptop distributed deployment** (Laptop 1: Model Server API; Laptop 2: Evaluator UI).

---

## 2. End-to-End System Architecture

```
+-----------------------------------------------------------------------------------+
|                                 DATA & GRAPH ENGINE                               |
|  - Raw .track (lon, lat, time)   - .nodes (vertices)   - .arcs (directed edges)   |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                           SPATIAL CANDIDATE GENERATION                            |
|  - Dynamic Local Metric Projection (degrees -> meters per trip)                   |
|  - cKDTree on road arc midpoints (k=5 nearest within radius R=60m)                |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                               FEATURE ENGINEERING                                 |
|  [perp_dist, proj_ratio, heading_error, arc_length, dist_to_start, dist_to_end]   |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                         SUPERVISED CLASSIFICATION (5 MODELS)                      |
|  - Linear, k-NN, Random Forest, HistGradientBoosting, MLP Neural Network          |
|  - Enforcing GroupKFold(groups=trip_id) to eliminate cross-trip data leakage      |
+-----------------------------------------------------------------------------------+
                                         │  Candidate Probabilities P(y=1)
                                         ▼
+-----------------------------------------------------------------------------------+
|                           HMM & VITERBI SEQUENTIAL DECODER                        |
|  - Emission Probability: Classifier Confidence * Spatial Gaussian Likelihood      |
|  - Transition Probability: Road Network Shortest Path vs Great Circle Distance    |
|  - Viterbi Dynamic Program: Reconstructs maximum likelihood connected path        |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                           TWO-LAPTOP DISTRIBUTED DEMO                             |
|  [LAPTOP 1: Saatwik]  FastAPI Backend (Uvicorn 0.0.0.0:8000)                      |
|                       Hosts Model + HMM + Server-side Ground Truth (.route)       |
|                                         ▲                                         |
|                       HTTP REST LAN     │  Selected Trip ID                       |
|                       Wi-Fi / Hotspot   │  Predicted Arcs + Metrics               |
|                                         ▼                                         |
|  [LAPTOP 2: Uttam]    Streamlit Evaluator UI + Interactive Folium Map             |
|                       Multi-layer toggle (GPS dots, True route, Snapped route)    |
+-----------------------------------------------------------------------------------+
```

---

## 3. Theoretical Concepts & Mathematical Formulations

### 3.1 Coordinate Systems & Dynamic Metric Projections
* **The Pitfall**: Longitude and latitude are spherical angular coordinates measured in degrees. Euclidean distance calculated in degrees ($\sqrt{\Delta \text{lon}^2 + \Delta \text{lat}^2}$) causes severe geographic distortion. Furthermore, 1 degree of longitude shrinks significantly towards the poles:
  $$\Delta x = \Delta \text{lon} \cdot \left(\frac{\pi}{180} R \cos(\phi)\right), \quad \Delta y = \Delta \text{lat} \cdot \left(\frac{\pi}{180} R\right)$$
* **Our Solution**: The dataset contains trips across global locations (latitudes $34^\circ$ to $56^\circ$, including the USA, Germany, and Russia). Rather than using a fixed UTM zone or distorted degrees, we compute a **Dynamic Local Metric Projection** centered on the mean latitude $\bar{\phi}$ of each trip:
  $$\text{lat\_m} = 111,139.0\text{ m/deg}, \quad \text{lon\_m} = 111,139.0 \cdot \cos(\bar{\phi})\text{ m/deg}$$
  Every spatial calculation (distances, projection points, and lengths) is evaluated strictly in **meters**.

### 3.2 Spatial Indexing & Candidate Generation
* A road network subgraph can contain over **165,000 arcs** per trip. Brute-force distance calculations for every GPS point would take minutes per trip.
* **Our Approach**: We build a spatial index using `scipy.spatial.cKDTree` over the midpoints of all directed road segments:
  $$\vec{M}_a = \frac{\vec{u}_a + \vec{v}_a}{2}$$
* For every GPS observation $p_t$, the KD-Tree queries the nearest $k=5$ arc midpoints in $\mathcal{O}(\log N)$ time, providing candidate road segments within an empirical buffer ($R \approx 60\text{m}$).

### 3.3 The 6 Geometric & Kinematic Features
For every candidate arc $a = (u, v)$ paired with GPS observation $p_t = (x_p, y_p)$, we extract 6 discriminatory features:

1. **`perp_dist` (Perpendicular Distance in meters)**:
   Vector from node $u$ to GPS point $p$: $\vec{w} = \vec{p} - \vec{u}$.  
   Segment vector: $\vec{s} = \vec{v} - \vec{u}$.  
   Projection scalar: $t = \frac{\vec{w} \cdot \vec{s}}{\|\vec{s}\|^2}$. Clamped to $t^* = \max(0, \min(1, t))$.  
   Closest point on segment: $\vec{c} = \vec{u} + t^* \vec{s}$.  
   $$\text{perp\_dist} = \|\vec{p} - \vec{c}\|$$
2. **`proj_ratio` (Along-Segment Projection Factor)**:
   The clamped ratio $t^* \in [0.0, 1.0]$. Indicates whether the car is near the start ($0.0$), middle ($0.5$), or end ($1.0$) of the segment.
3. **`heading_error` (Angular Difference in degrees $\in [0^\circ, 180^\circ]$)**:
   GPS motion vector from consecutive points: $\vec{v}_{\text{gps}} = \vec{p}_t - \vec{p}_{t-1}$.  
   Heading error between vehicle movement and road segment orientation:
   $$\theta_{\text{err}} = \arccos\left(\frac{\vec{v}_{\text{gps}} \cdot \vec{s}}{\|\vec{v}_{\text{gps}}\| \|\vec{s}\|}\right)$$
   Penalizes road segments pointing in the opposite direction (e.g., dual carriageways or wrong-way driving).
4. **`arc_length` (Segment Length in meters)**:
   $L = \|\vec{v} - \vec{u}\|$. Distinguishes short intersection connectors from long arterial avenues.
5. **`dist_to_start` ($d_u$)**: Euclidean distance in meters from $p_t$ to the start vertex $u$.
6. **`dist_to_end` ($d_v$)**: Euclidean distance in meters from $p_t$ to the end vertex $v$.

### 3.4 Data Leakage Prevention & GroupKFold
* **Why Standard K-Fold is Invalid**: GPS trajectories are high-frequency time series (1 Hz). Consecutive points within the same trip share nearly identical noise profiles, road segments, and headings. Randomly shuffling points into standard K-Fold causes points from the same vehicle trip to land in both training and validation sets, artificially inflating validation accuracy.
* **The Solution**: We strictly enforce `GroupKFold(n_splits=5)` grouped by `trip_id`. An entire trip belongs exclusively to either the training split or the validation split. Zero observations from the same trip ever cross fold boundaries.

### 3.5 The 5 Evaluated Machine Learning Families
Per official guidelines, we implemented models across 5 distinct algorithm paradigms:
1. **Linear & Probabilistic (Logistic Regression)**:
   * Uses `StandardScaler` and `class_weight='balanced'`.
   * Maps linear combination of features to probabilities via the sigmoid function:
     $$P(y=1 \mid \vec{x}) = \frac{1}{1 + e^{-\vec{w}^T \vec{x} - b}}$$
   * *Strengths*: Highly interpretable, extremely fast ($0.000\text{ms}$ inference latency).
   * *Limitations*: Assumes linear decision boundaries; cannot capture non-linear spatial interactions.
2. **Instance-Based (Distance-Weighted $k$-NN)**:
   * $k=5$ neighbors with KD-Tree search and weights proportional to inverse distance:
     $$w_i = \frac{1}{d(\vec{x}, \vec{x}_i) + \epsilon}$$
   * *Strengths*: Non-parametric, requires no assumptions on feature distributions.
   * *Limitations*: Slow inference latency ($\mathcal{O}(N)$ in dataset size); sensitive to unscaled feature noise.
3. **Ensemble / Bagging (Random Forest Classifier)**:
   * 100 decision trees, `max_depth=12`, parallelized across CPU cores.
   * Aggregates predictions across decorrelated trees to reduce variance.
   * *Strengths*: Handles non-linear feature interactions, robust to outliers, zero feature scaling needed.
   * *Limitations*: Moderately large model artifact size (~8 MB).
4. **Boosting (Histogram Gradient Boosting Classifier)**:
   * Bins continuous features into 256 integer bins for fast sequential tree building.
   * Sequentially minimizes binomial log-loss by fitting trees to pseudo-residuals:
     $$r_{im} = -\left[\frac{\partial L(y_i, f(\vec{x}_i))}{\partial f(\vec{x}_i)}\right]_{f=f_{m-1}}$$
   * *Strengths*: State-of-the-art tabular accuracy, naturally handles scale differences, fast training.
5. **Neural Network (Multi-Layer Perceptron - MLP)**:
   * Architecture: Input (6) $\to$ Dense (64, ReLU) $\to$ Dense (32, ReLU) $\to$ Output (1, Sigmoid).
   * Trained using the Adam optimizer with early stopping and L2 weight penalty ($\alpha=10^{-4}$).
   * *Strengths*: Learns smooth, complex non-linear representations across geometric spatial dimensions.
   * *Limitations*: Requires strict feature normalization (`StandardScaler`); slightly higher training epoch time.

### 3.6 Hidden Markov Models (HMM) & Viterbi Sequential Decoding
Point-wise classification alone evaluates each observation independently. If a GPS reading jumps due to a multipath bounce, the classifier will snap the vehicle onto a parallel street, breaking physical topology. HMM solves this by modeling sequential transitions:

* **Hidden States ($S_t$)**: The true candidate road segment $a_{t, i}$ traversed at time $t$.
* **Emission Probability $P(O_t \mid S_t = a_{t, i})$**: The likelihood that GPS observation $O_t$ was generated given the vehicle was on road segment $a_{t, i}$. We formulate this by blending the classifier's predicted probability $\hat{y}_{t, i}$ with a calibrated spatial Gaussian:
  $$P(O_t \mid a_{t, i}) \propto \hat{y}_{t, i} \cdot \exp\left(-\frac{\text{perp\_dist}(p_t, a_{t, i})^2}{2\sigma_z^2}\right)$$
* **Transition Probability $P(S_{t+1} = a_{t+1, j} \mid S_t = a_{t, i})$**: The physical plausibility of transitioning from arc $a_i$ to arc $a_j$.
  Using the road network directed graph ($\mathcal{G}$ built from `.nodes` and `.arcs`), we compute the shortest routing distance $d_{\text{graph}}(a_i, a_j)$ and compare it against the great-circle spatial displacement $d_{\text{gps}}(p_t, p_{t+1})$:
  $$P(a_{t+1, j} \mid a_{t, i}) \propto \frac{1}{\beta} \exp\left(-\frac{|d_{\text{graph}}(a_i, a_j) - d_{\text{gps}}(p_t, p_{t+1})|}{\beta}\right)$$
  * If $a_i == a_j$ (staying on same segment): $d_{\text{graph}} = 0$.
  * If $a_i$ directly connects to $a_j$ ($v_i == u_j$): high transition probability.
  * If disconnected in the road network: probability floors to $\epsilon = 10^{-6}$.
* **Viterbi Decoding Algorithm**: Evaluated in log-space to prevent underflow:
  $$V_{t, j} = \max_{i} \left( V_{t-1, i} + \ln P(a_{t, j} \mid a_{t-1, i}) \right) + \ln P(O_t \mid a_{t, j})$$
  Backtracking from $t=T$ yields the global, mathematically optimal, connected sequence of road segments!

### 3.7 Evaluation Metrics Decoded
1. **Observation-Level Top-1 Accuracy**: For every GPS point, we select the candidate arc with the highest predicted score. The metric is the percentage of observations where this top candidate matches the reference ground-truth route:
   $$\text{Top-1 Accuracy} = \frac{1}{N} \sum_{t=1}^N \mathbb{I}\left(\arg\max_{i} \hat{y}_{t, i} \in \text{Route}\right)$$
2. **Macro F1-Score**: The unweighted mean of F1-scores across binary classes ($y=0$ non-route vs $y=1$ true route):
   $$\text{Macro F1} = \frac{F1_{\text{class 0}} + F1_{\text{class 1}}}{2}$$
3. **Log Loss (Cross-Entropy)**: Evaluates the quality and calibration of predicted probabilities:
   $$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$
4. **Candidate Miss Rate**: The percentage of GPS observations where the true road segment was not present in the KD-Tree candidate set (sets the theoretical ceiling for classifier accuracy).
5. **Route Continuity %**: The percentage of consecutive predicted segments in the reconstructed route that are physically connected in the road network:
   $$\text{Route Continuity} = \frac{\sum_{t=1}^{T-1} \mathbb{I}\left(\text{arcs}[S_t].v == \text{arcs}[S_{t+1}].u \text{ or } S_t == S_{t+1}\right)}{T - 1} \times 100\%$$

---

## 4. Codebase Walkthrough: What Each Script Does

| File Path | Lead Owner | What It Executes | Key Outputs Produced |
| :--- | :--- | :--- | :--- |
| **`data_processing/eda.py`** | Vikas Jadar | Scans all 80 training trips (208,417 observations). Computes trajectory lengths, sampling interval distribution ($\Delta t$), checks missing values (0), and analyzes road subgraph node/arc scales. | `results/plots/sampling_intervals.png`<br>`results/plots/trajectory_lengths.png` |
| **`data_processing/candidate_gen.py`** | Vikas Jadar & Saatwik | Builds a spatial `cKDTree` on road arc midpoints using dynamic metric projections. Queries $k=5$ candidate arcs within $60\text{m}$, computes the 6 geometric features, and assigns ground-truth labels from `.route`. | `results/train_features.csv` (46,895 candidate observations) |
| **`model_training/train_5_models.py`** | Saatwik | Trains and benchmarks the 5 required model families (Linear, $k$-NN, Random Forest, Boosting, MLP) under 3-fold `GroupKFold(groups=trip_id)`. Computes Top-1 accuracy, Macro F1, Log Loss, and Latency. | `results/five_model_comparison.csv`<br>`results/best_model.joblib`<br>3 comparison plots in `results/plots/` |
| **`api/main.py`** | Vikas Aihole | FastAPI server running on **Laptop 1** (`0.0.0.0:8000`). Exposes `GET /health`, `GET /trips` (lists 10 evaluation trips), and `POST /predict/{trip_id}`. Computes metrics against server-side ground truth and logs latency live in the terminal. | Terminal CMD logs + REST API JSON responses |
| **`ui/app.py`** | Uttam | Streamlit evaluator interface running on **Laptop 2**. Allows the evaluator to pick any test trip, calls Laptop 1's API, displays 4 metric scorecards, and renders the interactive Folium map. | Live browser GUI on `http://localhost:8501` |
| **`visualisations/folium_map.py`** | Uttam | Reusable Folium component generating a 4-layer interactive map: Blue circle markers (GPS), Grey lines (road network), Green bold line (ground truth), and Red dashed line (PathSnap prediction). | Leaflet.js interactive map with layer toggles |

---

## 5. Actual Experimental Results (Pre-Lunch Checkpoint)

Our benchmark experiment evaluated all 5 algorithm families on 46,895 candidate rows with strict 3-fold `GroupKFold` cross-validation:

```
                       Model       Algorithm Family       CV Macro F1   Top-1 Accuracy   Log Loss   Latency
         Logistic Regression Linear & Probabilistic            0.6602           57.50%     0.6341  0.000 ms
      Distance-Weighted k-NN         Instance-based            0.6376           51.24%     3.4963  0.003 ms
               Random Forest     Ensemble (Bagging)            0.6810           58.69%     0.6761  0.002 ms
 Histogram Gradient Boosting               Boosting            0.6746           57.43%     0.6861  0.001 ms
Multi-Layer Perceptron (MLP)         Neural Network            0.6869           60.91%     0.5993  0.000 ms
```

### Key Technical Observations for the Report
1. **The Champion Model**: The **Multi-Layer Perceptron (MLP)** achieved the highest performance across all three primary evaluation dimensions: **CV Macro F1 of 0.6869**, observation-level **Top-1 accuracy of 60.91%**, and the lowest **Log Loss of 0.5993**.
2. **Why MLP Outperformed**: Road snapping depends on complex non-linear combinations of angular heading deviation and perpendicular distance (e.g., perpendicular distance matters much more when heading error is near $0^\circ$). The MLP's hidden layers $(64, 32)$ learned these non-linear cross-feature relationships more smoothly than linear models.
3. **Why k-NN Lagged**: Distance-weighted $k$-NN suffered from high Log Loss (3.4963) due to overconfident, uncalibrated nearest-neighbor probability estimates, and lower Top-1 accuracy (51.24%) due to sensitivity to varying local road density.
4. **Ensembles and Boosting**: Random Forest (0.6810) and HistGradientBoosting (0.6746) showed strong generalization stability across folds, making them robust fallback candidates.

---

## 6. First Phase Review Presentation Script (3 Minutes)

When the evaluators walk to your table, execute this precise speaking order:

```
[Saatwik] ──(30s)──> [Vikas Jadar] ──(45s)──> [Saatwik] ──(45s)──> [Uttam] ──(45s)──> [Vikas Aihole] ──(15s)
```

### 1. Welcome & Architecture (Saatwik - 30s)
> *"Good morning, Professors. We are Team PathSnap. We have built an end-to-end distributed system that reconstructs true vehicle paths from noisy GPS observations on a physical road network. We strictly follow the two-laptop architecture: Laptop 1 (mine) hosts the Model Server & FastAPI backend with server-side ground truth, while Laptop 2 (Uttam's) hosts the Evaluator UI."*

### 2. EDA & Feature Engineering (Vikas Jadar - 45s)
*Show `results/plots/sampling_intervals.png` and `trajectory_lengths.png`.*
> *"For exploratory data analysis, we inspected all 80 training trips (208,417 observations). We found that 96.5% of points follow a 1.0-second interval, but detected dropouts spanning several minutes. We also discovered that trips span diverse global regions (latitudes 34° to 56°), so we implemented dynamic metric projections per trip instead of distorted Euclidean degrees. Using a spatial KD-Tree on road arc midpoints, we extracted 6 geometric features: perpendicular distance, along-segment projection ratio, heading error, segment length, and endpoint distances."*

### 3. Five-Model Benchmark & Leakage Prevention (Saatwik - 45s)
*Show `results/five_model_comparison.csv` and `results/plots/model_macro_f1_comparison.png`.*
> *"To ensure zero data leakage, we enforced `GroupKFold` cross-validation grouped strictly by `trip_id`, ensuring observations from the same trip never cross fold boundaries. Ahead of the 12 PM deadline, we have already benchmarked all five required algorithm families: Linear, Instance-based, Random Forest, Gradient Boosting, and an MLP Neural Network. Our MLP achieved the highest Macro F1 of 0.6869 and 60.91% Top-1 accuracy with the lowest Log Loss of 0.5993. All comparison plots and the saved model artifact are pushed to our repository."*

### 4. Evaluator Portal & Interactive Folium Map (Uttam - 45s)
*Show Laptop 2 running `streamlit run ui/app.py`.*
> *"On Laptop 2, we built the Evaluator Portal using Streamlit. Evaluators can select any of the 10 reserved evaluation trips from this dropdown. Clicking 'Snap Route & Evaluate' sends a REST request to Laptop 1 over our local network and renders this interactive Folium map. We have toggleable layer controls: blue dots for raw GPS, green for the true road, and red for PathSnap's predicted path, alongside Top-1 Accuracy, Macro F1, and Route Continuity scorecards."*

### 5. Live Server Logs & Next Phase (Vikas Aihole - 15s)
*Point to Laptop 1 terminal running `uvicorn api.main:app`.*
> *"Here on Laptop 1, our FastAPI server processes the inference request, computes accuracy server-side against our private ground-truth `.route` file, and logs every request live with latency. In the afternoon phase, we are running Bayesian hyperparameter optimization with Optuna and implementing our HMM Viterbi decoder to achieve over 95% route continuity."*

---

## 7. Comprehensive Viva Q&A by Team Role

### Questions for Vikas Jadar (Data Pipeline & Features)
* **Q1: Why can't you calculate distance directly using $\sqrt{(\text{lon}_1 - \text{lon}_2)^2 + (\text{lat}_1 - \text{lat}_2)^2}$?**  
  *Answer*: *"Degrees are angular units, not linear distance. One degree of latitude is roughly 111 km, but one degree of longitude shrinks with the cosine of latitude, becoming much smaller as you move away from the equator. Calculating Euclidean distance in degrees distorts geometry and produces incorrect perpendicular offsets."*
* **Q2: How did you handle candidate generation without slow $\mathcal{O}(N \times M)$ comparisons?**  
  *Answer*: *"We built a spatial KD-Tree on the midpoints of the directed road segments using `scipy.spatial.cKDTree`. This allows querying candidate segments within a 60-meter radius in $\mathcal{O}(\log M)$ time."*
* **Q3: What is the `proj_ratio` feature and why is it useful?**  
  *Answer*: *"It represents the normalized scalar projection $t \in [0, 1]$ of the GPS point along the road segment vector. It tells the classifier whether the observation is near the start vertex ($0$), center ($0.5$), or end vertex ($1$), which helps resolve transitions at road intersections."*

### Questions for Saatwik (ML Models, Leakage & HMM)
* **Q1: How did you prevent data leakage during model training and evaluation?**  
  *Answer*: *"We used `GroupKFold` grouped strictly by `trip_id`. In trajectory data, consecutive 1 Hz points from the same trip are heavily autocorrelated. Standard K-Fold would randomly split points from the same vehicle into train and validation sets, causing severe data leakage. Our approach guarantees that entire trips are held out."*
* **Q2: Why did you evaluate 5 different algorithm families instead of just tuning one?**  
  *Answer*: *"The official guidelines require experimental evidence across genuinely different mathematical paradigms—linear, instance-based, bagging ensembles, gradient boosting, and neural networks. This ensures we select a model based on empirical trade-offs between generalization, log-loss calibration, and inference latency."*
* **Q3: Why is point-wise classification alone insufficient for map-matching?**  
  *Answer*: *"Point-wise classifiers treat each GPS observation independently. A sudden multipath bounce can cause the classifier to pick a segment on a disconnected parallel street, resulting in a physically impossible path where the vehicle teleports. An HMM with Viterbi decoding enforces topological connectivity by considering transition probabilities along the road network graph."*

### Questions for Uttam (UI & Visualizations)
* **Q1: How does your UI communicate with the model server?**  
  *Answer*: *"Streamlit acts as a client on Laptop 2. When an evaluator clicks 'Evaluate', Python's `requests` library sends an HTTP POST request over the local area network to Laptop 1's FastAPI server (`http://<ip>:8000/predict/{trip_id}`). The server returns a JSON payload containing the predicted arc sequence and metrics, which Streamlit renders into scorecards and an interactive Leaflet map."*
* **Q2: Why use Folium with layer controls?**  
  *Answer*: *"Folium compiles Leaflet.js maps directly in Python. By adding separate `folium.FeatureGroup` layers for raw GPS, ground truth, and predictions with `folium.LayerControl`, evaluators can visually toggle layers on and off to verify how accurately our prediction snaps to the ground truth."*

### Questions for Vikas Aihole (FastAPI, Deployment & Report)
* **Q1: Why must ground-truth labels remain strictly server-side?**  
  *Answer*: *"In a real-world production deployment (like Uber or delivery routing), the client device never knows the true answer beforehand. Furthermore, the hackathon rules strictly prohibit hardcoding predictions or leaking ground truth to the UI. The API processes the raw trajectory blindly and computes evaluation metrics server-side."*
* **Q2: How is the two-laptop network configured?**  
  *Answer*: *"Both laptops connect to the same local subnet (preferably via a mobile hotspot to avoid campus Wi-Fi client isolation). FastAPI binds to `--host 0.0.0.0` on port 8000 to accept external LAN traffic, and CORS middleware is configured to permit cross-origin requests."*
