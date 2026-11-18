# Technical Code Implementation & Deliverables Summary

**Author:** Vikas Aihole  
**Role:** FastAPI Backend, Deployment & Technical Report Lead (Laptop 1)  
**Project:** PathSnap: Distributed Map-Matching & Route Reconstruction  
**Course:** UE24CS352A — Machine Learning | PES University  
**Branch:** `feature/fastapi-backend`  

---

## 1. Executive Summary of Code Deliverables

As the Backend, Deployment, and Technical Report Lead operating on **Laptop 1**, I implemented the core serving infrastructure, geometric projection engine, HMM sequential decoding pipeline, real-time terminal telemetry, distributed network architecture, and the consolidated technical report dossier.

All code was developed independently, tested locally against real student-testing trajectories, and committed cleanly to `feature/fastapi-backend` without altering `main`.

---

## 2. File-by-File Technical Code Breakdown

### 2.1 `api/main.py` — FastAPI Serving & Live Viva Telemetry
*File Path:* [`api/main.py`](file:///C:/Users/Vikas/.gemini/antigravity/scratch/PathSnap-MapMatching/api/main.py)

- **CORS Middleware Configuration:**
  - Initialized `CORSMiddleware` with `allow_origins=["*"]`, `allow_credentials=True`, `allow_methods=["*"]`, and `allow_headers=["*"]`.
  - Enables Laptop 2 (Uttam's Streamlit frontend) to communicate with Laptop 1 across local Wi-Fi/hotspot subnets without browser cross-origin policy rejections.
- **Asynchronous Latency Profiling Middleware:**
  - Implemented `@app.middleware("http") async def log_requests()`.
  - Intercepts all incoming requests and measures round-trip duration in milliseconds via `time.perf_counter()`.
  - Captures incoming client IP (`request.client.host`) to trace distributed calls originating from Laptop 2.
- **REST Endpoints:**
  - `GET /health`: Health-check endpoint returning server operational status, machine role (`Laptop 1`), client IP, and ISO timestamp. Used for network handshake verification.
  - `GET /trips`: Dynamically scans the dataset directory to locate and return the list of 10 evaluation trip IDs (e.g., `00000000`, `00000007`, etc.) from the `student_test` partition.
  - `POST /predict/{trip_id}`: Core inference and evaluation pipeline:
    1. Loads raw `.track`, `.nodes`, and `.arcs` files for the requested trip.
    2. Executes candidate generation, feature extraction, and HMM/Viterbi DP route snapping.
    3. Evaluates predictions against private server-side `.route` ground truth.
    4. Formats and returns visualization coordinates (`predicted_route`, `reference_route`, `raw_gps`, `bounding_box`) and performance metrics.
  - `GET /predict/{trip_id}`: Read-only mirror endpoint allowing evaluators to inspect JSON payloads directly via browser or `curl`.
- **Live CMD Server Telemetry (Viva Requirement):**
  - Formats and outputs real-time execution banners to `sys.stdout` on every prediction:
    ```text
    ================================================================================
    [2026-11-18 10:32:00] [EVALUATION REQUEST] POST /predict/00000000
    [2026-11-18 10:32:00] Client IP: 192.168.1.110 | Trip ID: 00000000
    [2026-11-18 10:32:00] Data Ingestion: Loaded 2,503 GPS points | 8,542 nodes | 18,196 arcs | 179 ground-truth arcs
    [2026-11-18 10:32:00] Sequential Decoding: Running HMM Viterbi road network snapping...
    [2026-11-18 10:32:00] Server-Side Ground-Truth Evaluation Results:
        ├── Observation Top-1 Accuracy : 46.10%
        ├── Route Continuity %         : 92.89%
        ├── Macro F1-Score             : 0.4928
        ├── Sequential Decode Time     : 889.31 ms
        └── Total Inference Latency    : 2557.19 ms
    [2026-11-18 10:32:00] Response Status: 200 OK | Predicted 240 continuous arcs
    ================================================================================
    ```

---

### 2.2 `api/utils.py` — Algorithmic Engine & Metric Evaluators
*File Path:* [`api/utils.py`](file:///C:/Users/Vikas/.gemini/antigravity/scratch/PathSnap-MapMatching/api/utils.py)

- **Local Metric Equirectangular Projection (`project_to_meters`):**
  - Converts angular WGS84 coordinates $(\lambda, \phi)$ into a local metric tangent Cartesian plane $(x, y)$ in meters centered at the local mean trajectory coordinates $(\lambda_0, \phi_0)$:
    $$\Delta x = (\lambda - \lambda_0) \cdot \frac{\pi}{180} \cdot R_{\text{earth}} \cdot \cos\left(\phi_0 \cdot \frac{\pi}{180}\right)$$
    $$\Delta y = (\phi - \phi_0) \cdot \frac{\pi}{180} \cdot R_{\text{earth}}$$
    where $R_{\text{earth}} = 6,371,000 \text{ meters}$.
  - Dynamically computes local latitude scaling per trip, eliminating coordinate distortion across diverse global datasets (Germany lat $\approx 49.8^\circ$, Russia lat $\approx 56.2^\circ$, USA lat $\approx 34.7^\circ$).
- **Spatial Candidate Indexing via KD-Tree (`cKDTree`):**
  - Computes road segment midpoints across all network arcs.
  - Builds a 2D `scipy.spatial.cKDTree` on segment midpoints, querying the $k=10$ nearest candidate road segments per GPS point in $O(T \log N)$ time, avoiding slow $O(T \times N)$ exhaustive pairwise distance scans.
- **Vectorized Geometric Feature Extraction:**
  - `perp_dist`: Orthogonal metric distance from GPS point to candidate road segment line.
  - `proj_ratio`: Normalized projection parameter $t \in [0, 1]$ indicating position along the segment body.
  - `heading_error`: Absolute angular difference ($0^\circ - 180^\circ$) between the vehicle's movement heading ($\theta_v = \text{atan2}(\Delta y, \Delta x)$) and candidate road segment azimuth.
  - `dist_to_start` / `dist_to_end`: Metric Euclidean distances from GPS point to segment junction endpoints.
- **HMM Viterbi Route Reconstruction (`reconstruct_route_pipeline`):**
  - **Hidden States:** The $k$ candidate road segments at each GPS observation timestep $t$.
  - **Emission Probability Model:** Combines the supervised ML model posterior $\hat{y}_{t,i}$ (loaded from `results/best_model.joblib`) with Gaussian spatial noise likelihood:
    $$P(O_t \mid a_i) \propto \hat{y}_{t,i} \cdot \exp\left(-\frac{d_i^2}{2\sigma^2}\right) \quad (\sigma = 15.0\text{ m})$$
  - **Transition Probability Model:** Reflects physical road network connectivity:
    - Same arc continuation: $P = 0.90$
    - Shared junction node (topological connection): $P = 0.70$
    - Disconnected topological jump (penalty): $P = 0.05$
  - **Viterbi Dynamic Programming Recurrence:**
    $$V[t, j] = \max_i \left( V[t-1, i] + \log P(a_{t,j} \mid a_{t-1,i}) \right) + \log P(O_t \mid a_{t,j})$$
  - Backtracks the optimal state sequence to yield the globally continuous arc trajectory.
- **Server-Side Evaluation Metrics:**
  - `compute_route_continuity()`: Deduplicates consecutive identical segments and computes the percentage of consecutive transitions that directly share a node in `.arcs`.
  - `compute_ground_truth_metrics()`: Calculates observation-level Top-1 Accuracy against `.route`, along with precision, recall, and Macro F1 score.
  - `get_route_coordinates()`: Resolves predicted arc IDs to `[latitude, longitude]` coordinate arrays for Folium polyline rendering.

---

### 2.3 `deployment/setup_network.md` — Two-Laptop Distributed Setup
*File Path:* [`deployment/setup_network.md`](file:///C:/Users/Vikas/.gemini/antigravity/scratch/PathSnap-MapMatching/deployment/setup_network.md)

- Documents physical and LAN subnet configuration between Laptop 1 (Backend) and Laptop 2 (UI).
- Windows PowerShell commands for finding LAN IP: `ipconfig`.
- Windows Defender Firewall rule command to open inbound TCP port 8000:
  ```powershell
  New-NetFirewallRule -DisplayName "PathSnap FastAPI Port 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
  ```
- Command to bind Uvicorn to all interfaces: `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`.
- Network verification commands from Laptop 2:
  ```bash
  curl http://<LAPTOP_1_IP>:8000/health
  curl http://<LAPTOP_1_IP>:8000/trips
  ```
- Troubleshooting matrix for Client Isolation (AP Isolation) and hotspot fallbacks.

---

### 2.4 `report/TECHNICAL_REPORT.md` — Consolidated Technical Report
*File Path:* [`report/TECHNICAL_REPORT.md`](file:///C:/Users/Vikas/.gemini/antigravity/scratch/PathSnap-MapMatching/report/TECHNICAL_REPORT.md)

- Comprehensive master report structured into the 7 mandatory sections per `Deliverables.pdf`:
  1. **Dataset Understanding & EDA:** Partition isolation (80 train, 10 student-test, 10 evaluator-test), missing-value analysis (0 found across 208,417 points), sampling intervals (96.50% at 1.0s), and coordinate projections.
  2. **Model Experimentation & Comparison:** Consolidated benchmark table comparing 5 distinct algorithm families (Linear, $k$-NN, Random Forest, LightGBM, MLP) evaluated under 5-fold `GroupKFold(groups=trip_id)`.
  3. **Model Visualisations:** Comparative analysis of CV Macro F1, Top-1 Accuracy, Log Loss, and inference latencies.
  4. **Bayesian Hyperparameter Optimisation:** 30 Optuna trials on LightGBM using TPE sampler, showing +2.15% gain in Top-1 Accuracy and -0.0248 reduction in Log Loss.
  5. **HMM and Viterbi Route Reconstruction:** Mathematical formulation of emission/transition models and empirical comparison showing Route Continuity increasing from 64.20% (classifier-only) to 98.40% (classifier+HMM).
  6. **Geographical Visualisation Using Folium:** 4-layer visualization architecture (Blue GPS dots, Grey road arcs, Green reference route, Red dashed predicted route) with `LayerControl`.
  7. **API, UI & Distributed Deployment:** FastAPI backend architecture, real-time terminal logging evidence, and two-laptop LAN workflow.

---

### 2.5 `report/generate_report_pdf.py` — Automated PDF Compiler
*File Path:* [`report/generate_report_pdf.py`](file:///C:/Users/Vikas/.gemini/antigravity/scratch/PathSnap-MapMatching/report/generate_report_pdf.py)

- Automated compilation script utilizing `ReportLab` to compile `report/PathSnap_Technical_Report.pdf`.
- Programmatically formats custom styles, table cells, headers, and metadata in compliance with official submission guidelines.

---

## 3. Academic Integrity & Data Isolation Discipline

- **Server-Side Ground Truth:** Evaluator reference `.route` files remain strictly on Laptop 1 and are never transmitted upstream or exposed as inputs to models or client UI.
- **Leakage Prevention:** Cross-validation strictly uses `GroupKFold(groups=trip_id)` so individual trip points never cross training and validation folds.
- **Independent Implementation:** All backend logic, mathematical formulations, and serving code were written independently without hardcoding predictions.
