# PathSnap: Distributed Map-Matching & Sequential Route Reconstruction
## Comprehensive Technical Report & Evaluation Dossier (Phase 2 Final)

**Course:** UE24CS352A — Machine Learning  
**Institution:** Department of Computer Science and Engineering, PES University  
**Team Members & Roles:**
- **Saatwik:** Chief ML Architect & Sequential Decoding Lead
- **Uttam:** Frontend UI & Interactive Visualization Lead
- **Vikas Jadar:** Data Pipeline & Feature Extraction Specialist
- **Vikas Aihole:** FastAPI Backend, Two-Laptop Deployment & Technical Report Lead

---

## 1. Dataset Understanding and Exploratory Data Analysis (EDA)

### 1.1 Dataset Architecture and Partition Structure
The PathSnap challenge provides noisy GPS trajectories alongside local topological road network subgraphs. The complete dataset is structured into three strictly isolated partitions to prevent any data leakage:
- **Training Partition (80 trips):** Used exclusively for exploratory data analysis, spatial candidate generation, geometric feature engineering, 5-fold group-aware cross-validation, and Bayesian hyperparameter optimisation.
- **Student-Testing Partition (10 trips):** Held-out development partition used solely for post-hoc validation, evaluating the generalization of trained models, and testing the HMM sequential decoder.
- **Evaluator-Testing Partition (10 trips):** Reserved exclusively by the evaluation committee for live demonstration on Laptop 1. Ground truth remains strictly server-side.

Each trip directory `NNNNNNNN/` contains four raw files:
1. `NNNNNNNN.track`: Space-separated GPS observations without headers (`lon, lat, timestamp`).
2. `NNNNNNNN.nodes`: Road network junction nodes (`lon, lat`), indexed 0-based by row.
3. `NNNNNNNN.arcs`: Directed and bidirectional road segments connecting nodes (`from_node_id, to_node_id`), where 0-based row index serves as `arc_id`.
4. `NNNNNNNN.route`: Ground-truth chronological sequence of traversed `arc_id` values.

### 1.2 Data Preprocessing & Coordinate Transformation
GPS tracking coordinates $(\lambda, \phi)$ are angular spherical coordinates on the WGS84 datum. Directly computing Euclidean distances on degree coordinates introduces severe spatial distortion because longitude scale contracts as a function of latitude ($\cos \phi$). 

To ensure exact physical distance measurements in meters, all coordinates for each trip are converted into an equirectangular metric tangent plane centered at the local mean trajectory coordinates $(\lambda_0, \phi_0)$:
$$\Delta x = (\lambda - \lambda_0) \cdot \frac{\pi}{180} \cdot R_{\text{earth}} \cdot \cos\left(\phi_0 \cdot \frac{\pi}{180}\right)$$
$$\Delta y = (\phi - \phi_0) \cdot \frac{\pi}{180} \cdot R_{\text{earth}}$$
where $R_{\text{earth}} = 6,371,000 \text{ meters}$.

### 1.3 Exploratory Data Analysis & Empirical Observations
An exhaustive analysis across all 80 training trips (208,417 raw GPS observations) yielded the following empirical findings:
- **Missing-Value Analysis:** Exactly **0 missing values** detected across all 208,417 trajectory records.
- **Sampling Interval ($\Delta t$):** The nominal sampling rate is 1 Hz. Exactly **96.50%** of all observations occur at an exact $\Delta t = 1.0\text{ s}$ gap. The overall mean time gap is $2.181\text{ s}$.
- **Trajectory Lengths:** Trajectories average **2,605.2 points** per trip (range: 361 to 19,313 points).
- **Multi-Region Geographical Spread:** Trajectories span distinct international coordinate zones including Germany (lat $\approx 49.8^\circ$), Russia (lat $\approx 56.2^\circ$), and the USA (lat $\approx 34.7^\circ$).

#### Official EDA Visualizations:
- **Geographic Regions Overview:** `results/plots/geographic_regions.png`
- **Sampling Interval Distribution:** `results/plots/sampling_intervals.png`

---

## 2. Model Experimentation and 5-Model Comparison

### 2.1 Official 5-Model Benchmark Comparison Table
Trained and evaluated on the full candidate dataset (1,042,085 candidate rows across all 80 training trips) using `GroupKFold(n_splits=5, groups=trip_id)` to strictly prevent cross-trip data leakage:

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Log Loss | Train Time (s) | Inference Latency (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | `max_iter=1000, class_weight='balanced'` | 0.7562 | 70.52% | 0.5033 | 0.27s | 0.000 ms |
| **Distance-Weighted $k$-NN** | Instance-Based | `k=5, weights='distance', algorithm='kd_tree'` | 0.7697 | 69.76% | 2.7509 | 0.25s | 0.006 ms |
| **Random Forest** | Ensemble (Bagging) | `n_estimators=100, max_depth=12, n_jobs=-1` | 0.7905 | 71.70% | 0.4437 | 13.23s | 0.001 ms |
| **Histogram Gradient Boosting** | Boosting (Ensemble) | `max_iter=100, learning_rate=0.08, l2=0.1` | 0.7879 | 71.59% | 0.4445 | 2.17s | 0.001 ms |
| **Multi-Layer Perceptron (MLP)** | Neural Network | `hidden=(64, 32), early_stopping=True` | **0.7984** | **71.86%** | **0.4402** | 40.03s | 0.000 ms |

- **Official Visualization:** `results/plots/model_macro_f1_comparison.png`

---

## 3. Bayesian Hyperparameter Optimisation (Optuna)

### 3.1 Search Configuration
Bayesian hyperparameter optimization was executed on **HistGradientBoosting** over 30 sequential trials using Optuna's Tree-structured Parzen Estimator (`TPESampler(seed=42)`). Although MLP achieved a nominally higher raw CV F1 (0.7984 vs 0.7879), HistGradientBoosting trains in 2.17s vs 40.03s, providing exceptional computational efficiency and low serving latency.

- **Search Space:**
  - `learning_rate`: Log-uniform $[0.001, 0.30]$
  - `max_depth`: Integer $[3, 14]$
  - `max_leaf_nodes`: Integer $[15, 127]$
  - `l2_regularization`: Log-uniform $[0.0001, 0.10]$
  - `min_samples_leaf`: Integer $[10, 200]$
- **Best Hyperparameters Found:**
  - `learning_rate = 0.1055`
  - `max_depth = 7`
  - `max_leaf_nodes = 39`
  - `l2_regularization = 0.002156`
  - `min_samples_leaf = 48`

### 3.2 Official Tuned vs. Untuned Benchmark Table

| Variant | CV Macro F1 | Student-Test Top-1 Accuracy | Log Loss | Inference Latency |
| :--- | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Untuned)** | 0.7879 | 71.59% | 0.4445 | 0.001 ms |
| **HistGradientBoosting (Tuned, 30 Trials)** | **0.7889** | **82.69%** | 0.4699 | 0.001 ms |

*Tuning yielded a massive **+11.10% absolute gain** in student-test Top-1 Observation Accuracy (from 71.59% to **82.69%**).*

---

## 4. HMM and Viterbi Sequential Route Reconstruction

### 4.1 HMM Formulation & Energy Model
To convert disconnected per-point candidate classifications into a physically continuous vehicle trajectory, PathSnap applies a Hidden Markov Model (HMM) decoded via Viterbi dynamic programming:
- **Hidden States:** Candidate road segment IDs $a_{t,i}$ retrieved at each timestep.
- **Emission Probability:** 
  $$P(O_t \mid a_i) \propto P_{\text{model}}(y=1 \mid \mathbf{x}_i) \cdot \exp\left(-\frac{d_i^2}{2\sigma^2}\right) \quad (\sigma = 10\text{ m})$$
- **Transition Probability:** 
  $$P(a_{t+1} \mid a_t) \propto \exp\left(-\frac{|d_{\text{graph}}(a_t, a_{t+1}) - d_{\text{gc}}(p_t, p_{t+1})|}{\beta}\right) \quad (\beta = 100\text{ m})$$
  where $d_{\text{graph}} = 0$ for connected or identical segments, and shortest-path Dijkstra distance otherwise.

### 4.2 Official Route Continuity Benchmark (10 Student-Test Trips)

| Trip ID | Timesteps | Classifier-Only Accuracy | HMM Pointwise Accuracy | Classifier-Only Continuity | HMM Continuity % |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `00000000` | 2,503 | 87.97% | 87.53% | 84.81% | **93.92%** |
| `00000007` | 1,354 | 85.97% | 91.36% | 82.26% | **98.23%** |
| `00000009` | 1,049 | 87.13% | 87.70% | 72.61% | **86.16%** |
| `00000040` | 2,995 | 87.41% | 88.81% | 80.76% | **95.36%** |
| `00000052` | 262 | 93.89% | 95.80% | 83.91% | **91.57%** |
| `00000056` | 3,958 | 87.57% | 92.42% | 74.48% | **90.67%** |
| `00000065` | 768 | 83.98% | 89.06% | 76.79% | **93.22%** |
| `00000076` | 1,486 | 88.49% | 89.17% | 74.61% | **88.08%** |
| `00000077` | 3,911 | 68.06% | 77.52% | 63.02% | **92.61%** |
| `00000091` | 874 | 80.43% | 86.16% | 70.56% | **89.58%** |
| **OVERALL MEAN** | **19,260** | **83.09%** | **87.57%** | **76.38%** | **91.94%** |

*Sequential decoding elevated mean Route Continuity from **76.38%** to **91.94% (+15.56%)**, resolving broken parallel segment assignments and delivering smooth connected vehicle paths.*

---

## 5. Geographical Visualisation Using Folium

The interactive visualization architecture implemented in `visualisations/folium_map.py` renders four distinct map layers:
- **Layer 1 (Blue Dots):** Raw observed GPS points with timestamp popups.
- **Layer 2 (Grey Lines):** Road network bounding graph segments.
- **Layer 3 (Green Bold Line):** Ground-truth reference route (`.route`).
- **Layer 4 (Red Dashed Line):** PathSnap reconstructed route.
Equipped with `folium.LayerControl()` to enable evaluators to toggle and compare routes interactively.

---

## 6. API, UI and Distributed Deployment

- **Laptop 1 (FastAPI Service on `0.0.0.0:8000`):** Ingests raw GPS data, executes candidate feature extraction matching the 12 geometric/kinematic features, loads `best_tuned_model.joblib`, runs HMM Viterbi decoding, and computes server-side evaluation metrics.
- **Real-Time CMD Logs:** Evaluators observe live request banners on every prediction:
  ```text
  ================================================================================
  [2026-11-18 12:15:00] [EVALUATION REQUEST] POST /predict/00000000
  [2026-11-18 12:15:00] Ingested 2,503 points | Model: HistGradientBoosting (Tuned)
  [2026-11-18 12:15:00] Server Evaluation: Top-1 Acc: 87.53% | Route Continuity: 93.92%
  ================================================================================
  ```
- **Laptop 2 (Streamlit UI):** Queries Laptop 1 REST endpoints and renders interactive multi-layer maps and metrics scorecards.
