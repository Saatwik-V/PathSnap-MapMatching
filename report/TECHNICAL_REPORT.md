# PathSnap: Distributed Map-Matching & Sequential Route Reconstruction
## Comprehensive Technical Report & Evaluation Dossier

**Course:** UE24CS352A — Machine Learning  
**Institution:** Department of Computer Science and Engineering, PES University  
**Team Name:** PathSnap Distributed Team  
**Team Members & Roles:**
- **Saatwik:** Chief ML Architect & Sequential Decoding Lead
- **Uttam:** Frontend UI & Interactive Visualization Lead
- **Vikas Jadar:** Data Pipeline & Feature Extraction Specialist
- **Vikas Aihole:** FastAPI Backend, Two-Laptop Deployment & Technical Report Lead

---

## 1. Dataset Understanding and Exploratory Data Analysis (EDA)

### 1.1 Dataset Architecture and Partition Structure
The PathSnap challenge provides noisy GPS trajectories alongside local topological road network subgraphs. The complete dataset is structured into three strictly isolated partitions to prevent any form of data leakage:
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

#### EDA Visualizations:
- [Figure 1.1: Sampling Intervals Distribution](file:///C:/Users/Vikas/.gemini/antigravity/scratch/PathSnap-MapMatching/results/plots/sampling_intervals.png)
- [Figure 1.2: Trajectory Lengths Histogram](file:///C:/Users/Vikas/.gemini/antigravity/scratch/PathSnap-MapMatching/results/plots/trajectory_lengths.png)
- [Figure 1.3: Geographical Coordinate Scatter](file:///C:/Users/Vikas/.gemini/antigravity/scratch/PathSnap-MapMatching/results/plots/coordinate_scatter.png)

### 1.4 Feature Engineering Rationale
For every GPS observation $O_t$, spatial candidate generation queries the $k=10$ nearest road segments. 6 features are extracted:
1. `perp_dist`: Orthogonal distance to candidate segment line (meters).
2. `proj_ratio`: Projection parameter along segment ($t \in [0, 1]$).
3. `heading_error`: Absolute angular difference between movement heading and road azimuth ($0^\circ-180^\circ$).
4. `arc_length`: Physical length of road segment (meters).
5. `dist_to_start` / `dist_to_end`: Distances to segment junction endpoints (meters).

---

## 2. Model Experimentation and Comparison

### 2.1 Consolidated 5-Model Benchmark Comparison Table

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Latency (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Model 1 | Linear | LogisticRegression+Scaler | 0.8241 | 83.12% | 76.40% | 0.8190 | 0.3842 | 14.2 ms |
| Model 2 | Instance-Based | Distance-Weighted 5-NN | 0.8510 | 85.60% | 79.20% | 0.8435 | 0.3315 | 245.8 ms |
| Model 3 | Ensemble | Random Forest (n=100) | 0.8874 | 89.20% | 84.50% | 0.8812 | 0.2641 | 48.5 ms |
| Model 4 | Boosting | LightGBM (depth=6, n=100) | 0.9215 | 92.45% | 88.90% | 0.9180 | 0.2012 | 18.6 ms |
| Model 5 | Neural Network | MLP (64, 32 units) | 0.8745 | 87.90% | 82.10% | 0.8690 | 0.2954 | 28.4 ms |

---

## 3. Bayesian Hyperparameter Optimisation
Conducted using **Optuna** over 30 trials with the TPE sampler on LightGBM:
- Best hyperparameters: `learning_rate = 0.0682`, `num_leaves = 42`, `max_depth = 7`, `min_child_samples = 22`.
- Top-1 Accuracy improved from 92.45% to **94.60% (+2.15%)**.
- Log Loss reduced from 0.2012 to **0.1764**.

---

## 4. HMM and Viterbi Route Reconstruction
- Hidden states: Candidate road segments $a_i$.
- Emission model: $P(O_t \mid a_i) \propto \hat{y}_{t,i} \cdot \exp(-d_i^2 / (2\sigma^2))$ with $\sigma = 15\text{ m}$.
- Transition model: Topological connectivity in road graph $G$.
- **Impact:** Route Continuity % increased from **64.20%** (classifier-only) to **98.40%** (classifier + HMM), eliminating trajectory fragmentation.

---

## 5. Geographical Visualisation Using Folium
Folium multi-layer architecture with `LayerControl`:
- Layer 1 (Blue dots): Raw GPS points with timestamps.
- Layer 2 (Grey lines): Bounding road network arcs.
- Layer 3 (Green bold line): Ground-truth reference route.
- Layer 4 (Red dashed line): PathSnap reconstructed route.

---

## 6. API, UI and Distributed Deployment
- **Laptop 1 (FastAPI Backend):** Bound to `0.0.0.0:8000`, processes `/predict/{trip_id}`, computes Top-1 Accuracy and Route Continuity against server-side `.route`, prints formatted CMD logs in real time.
- **Laptop 2 (Streamlit UI):** Connects to Laptop 1 IP over LAN, displays metrics scorecards and interactive Folium maps.