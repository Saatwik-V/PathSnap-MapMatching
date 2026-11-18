# PathSnap — Live Demo Run & Presentation Guide
## Step-by-Step Command Playbook and Team Choreography for Final Evaluation
**Course:** UE24CS352A — Machine Learning | **Department:** CSE, PES University  
**Team:** Saatwik (Lead / ML & HMM), Uttam (Frontend UI), Vikas Jadar (Data / EDA), Vikas Aihole (Backend / Report)

---

## 1. System Architecture & Role Responsibilities

```
+-------------------------------------------------------+          +-------------------------------------------------------+
|                       LAPTOP 1                        |          |                       LAPTOP 2                        |
|             Backend Server & Model Engine             |          |              Client Evaluator Portal UI               |
|            (Saatwik & Vikas Aihole)                   |  HTTP    |                       (Uttam)                         |
|                                                       |  REST    |                                                       |
|  - Host: 0.0.0.0:8000                                 |  (LAN)   |  - Host: localhost:8501                               |
|  - Ingests raw .track, .nodes, .arcs                  | <======> |  - Dropdown with 10 Evaluator Trips                   |
|  - Extracts 12 kinematic & geometric features         |          |  - "Snap Route & Evaluate" trigger button             |
|  - Optuna-Tuned Champion Model (HistGradientBoosting) |          |  - Modern Dark CSS Metric Cards (with fill bars)      |
|  - NetworkX HMM & Viterbi Decoder (with gap repair)   |          |  - 4-Layer Interactive Folium Map                     |
|  - Evaluates against private server-side .route       |          |  - Floating bottom-left legend & layer toggle controls|
|  - Prints real-time CMD request logs with latency     |          |  - "Download Interactive HTML Map" offline export     |
+-------------------------------------------------------+          +-------------------------------------------------------+
```

### Team Member Assignments:
- **Saatwik (Laptop 1):** Manages the FastAPI server, monitors live terminal logs, and explains the ML models, feature engineering, and HMM Viterbi decoding.
- **Vikas Aihole (Laptop 1):** Manages LAN hotspot networking, verifies request payloads, and presents `report/PathSnap_Technical_Report.pdf`.
- **Uttam (Laptop 2):** Drives the Streamlit evaluator UI, demonstrates the 4-layer Folium map, toggles layers, and downloads offline HTML maps.
- **Vikas Jadar (Floater / Specialist):** Defends the dataset topology, multi-region coordinates, missing value analysis, and GPS dropout handling.

---

## 2. Pre-Demo Network Setup (Do This 10 Minutes Before Presentation)

### Step 1: Connect to the Same Network
- Connect both Laptop 1 and Laptop 2 to the **same Wi-Fi network or a phone hotspot**.
- *Tip:* A personal phone hotspot (e.g. "PathSnap-Demo") is strongly recommended over campus Wi-Fi to avoid university firewall blocks or IP isolation.

### Step 2: Find Laptop 1's LAN IP Address
- **On Laptop 1 Terminal:**
  - *macOS/Linux:*
    ```bash
    ifconfig | grep "inet " | grep -v 127.0.0.1
    ```
    *(e.g., `10.30.202.122` or `192.168.43.150`)*
  - *Windows:*
    ```cmd
    ipconfig
    ```
    *(Look for IPv4 Address under Wireless LAN)*

### Step 3: Test Connectivity from Laptop 2
- **On Laptop 2 Terminal (Uttam):**
  ```bash
  curl http://<LAPTOP_1_IP>:8000/health
  ```
  *Expected Output:*
  ```json
  {"status":"healthy","service":"PathSnap FastAPI Backend","machine":"Laptop 1", ...}
  ```

---

## 3. Exact Commands to Launch the System

### Laptop 1: Launch FastAPI Model Server (Saatwik & Vikas Aihole)
Open terminal in the repository root directory on Laptop 1:

```bash
# 1. Activate your Python environment
source venv/bin/activate   # or conda activate pathsnap

# 2. Start the Uvicorn server bound to all network interfaces (0.0.0.0)
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

> **IMPORTANT:** Keep this terminal window open and visible side-by-side with Laptop 2 during the evaluation! Evaluators are instructed to inspect Laptop 1's terminal logs.

---

### Laptop 2: Copy the trip data first (one time, before the demo)
`Student folder/` is **not in git**, so `git pull` does not bring it. The UI reads the road network and GPS points locally to draw the map, so without the data the scorecards appear but the map does not.
1. Copy `Student folder/evaluator_test/` (10 trip folders) from Laptop 1 to the same relative path in the repo on Laptop 2, by USB or AirDrop. Only `.track`, `.nodes` and `.arcs` are needed; **the `.route` files are not needed on Laptop 2 and should not be copied** (the ground truth stays on Laptop 1).
2. Check: `ls "Student folder/evaluator_test" | wc -l` prints `10`.

### Laptop 2: Launch Streamlit Evaluator Portal (Uttam)
Open terminal in the repository root directory on Laptop 2:

```bash
# 1. Pull latest code from main
git checkout main
git pull origin main

# 2. Launch Streamlit
streamlit run ui/app.py
```

> **Laptop 2 is a Windows machine (PowerShell):** activate the environment with `.\venv\Scripts\activate`, and test connectivity with `curl.exe http://<LAPTOP_1_IP>:8000/health` (plain `curl` is an alias of `Invoke-WebRequest` and prompts for confirmation).

Streamlit will open in the browser at `http://localhost:8501`.

---

## 4. Live Presentation Choreography (The 3-Minute Evaluation Walkthrough)

Follow this exact step-by-step sequence when the faculty panel arrives:

### Minute 1: Handover Documentation & System Architecture
1. **Hand over the Technical Report**:
   - Open or hand over [PathSnap_Technical_Report.pdf](file:///Users/saatwik/Documents/ML_hackathon/report/PathSnap_Technical_Report.pdf).
   - Point out: *"Our full technical report contains the 5-model benchmark, Bayesian tuning logs, HMM continuity gain charts, and mathematical feature justifications."*
2. **Explain the 2-Laptop Distributed Topology**:
   - Show Laptop 1: *"Laptop 1 is our isolated model server and ground-truth vault. It hosts the 12-feature model and HMM Viterbi decoder. The ground-truth `.route` files remain quarantined here."*
   - Show Laptop 2: *"Laptop 2 is the client portal that you (the evaluators) interact with. It communicates with Laptop 1 over our LAN network."*

---

### Minute 2: Trigger the Live Route Reconstruction
1. **Show the Connection Status on Laptop 2**:
   - In the sidebar under **Laptop 1 API Base URL**, show `http://<LAPTOP_1_IP>:8000`.
   - Point out the green badge: `✅ Connected to Laptop 1 API`.
2. **Select a Star Trip from the Dropdown**:
   - Select **Trip `00000073`** or **Trip `00000067`** from the dropdown.
   *(These trips score about 95% Top-1 with 99.5-100% route continuity in well under a second of latency).*
3. **Click `🚀 Snap Route & Evaluate`**:
   - Click the red button on Laptop 2.

---

### Minute 3: The Live Reveal (Laptops 1 & 2 Synchronized)
1. **Immediately Direct Evaluator's Eyes to Laptop 1 Terminal**:
   - Show the real-time server logs streaming live:
     ```text
     ================================================================================
     [2026-11-18 16:38:44] [EVALUATION REQUEST] POST /predict/00000073
     [2026-11-18 16:38:44] Client IP: 127.0.0.1 | Trip ID: 00000073
     [2026-11-18 16:38:44] Data Ingestion: Loaded 1,144 GPS points | 12,151 nodes | 25,631 arcs | 238 ground-truth arcs
     [2026-11-18 16:38:44] Sequential Decoding: Running HMM Viterbi road network snapping...
     [2026-11-18 16:38:44] Server-Side Ground-Truth Evaluation Results:
         ├── Observation Top-1 Accuracy : 95.19%
         ├── Route Continuity %         : 100.00%
         ├── Macro F1-Score             : 0.8577
         ├── Sequential Decode Time     : 368.95 ms
         └── Total Inference Latency    : 567.16 ms
     [2026-11-18 16:38:44] Response Status: 200 OK | Predicted 368 continuous arcs
     ================================================================================
     ```
   - Emphasize: *"Notice that the server executed 12-feature candidate extraction, tree scoring, and Viterbi dynamic programming across 1,144 points in about 570 milliseconds."* (The numbers above were captured from a real run of this trip; yours will differ by a few milliseconds.)

2. **Return to Laptop 2 to Inspect the Interactive Results**:
   - **Scorecards**: Point out the 4 modern dark cards:
     * **Top-1 Accuracy**: `95.2%` (with blue progress fill bar)
     * **Macro F1**: `0.858` (with purple accent; this is the route-level F1 between predicted and reference arc sets, not the candidate-classifier Macro F1 in the report)
     * **Route Continuity**: `100.0%` (with mint green fill bar)
     * **Inference Latency**: about `567 ms` (with amber accent)
   - **Status Alert**: Point out `✅ HMM Viterbi decoding complete — 368 continuous road arcs reconstructed.`
   - **Interactive Folium Map**:
     * Point out the **auto-fit bounds** framing the trajectory perfectly in the center.
     * Point out the **floating legend** in the bottom-left corner explaining the 4 layers.
     * Open **LayerControl** in top-right: toggle the blue GPS dots off to show the red path snapped onto the grey road network. **There is no green reference layer for evaluator trips, by design:** ground truth stays on Laptop 1, and accuracy is read from the scorecards and Laptop 1's log. (For `student_test` trips the green reference route is shown.)
     * Hover over a blue GPS dot to show the tooltip with observation index, time, and instant velocity in m/s.
   - **Offline Export Button**:
     * Click **"⬇️ Download Interactive HTML Map"**.
     * Show that the HTML map downloads instantly (`trip_00000073_map.html`, under 1 MB) and can be opened in any standalone web browser without Python.

---

## 5. Recommended Evaluation Trips Cheat Sheet

If the evaluators let you pick or ask to test multiple trips, use this reference:

| Trip ID | GPS Points | Top-1 Accuracy | Route Continuity | Latency | Why It's Great for Live Demo |
|---|:---:|:---:|:---:|:---:|---|
| **`00000073`** | 1,144 | **95.2%** | **100.0%** | **567 ms** | **Best overall demo trip.** Ultra-fast, perfect continuity, high accuracy. |
| **`00000067`** | 916 | **95.0%** | **99.5%** | **278 ms** | **Fastest inference.** Instant sub-300ms return, beautiful urban path. |
| **`00000049`** | 2,642 | **93.9%** | **98.4%** | **1,106 ms** | Demonstrates scaling over longer 2,600-point trajectory. |
| **`00000017`** | 1,346 | **91.8%** | **100.0%** | **1,265 ms** | Perfect 100.0% continuity on divided corridor. |
| **`00000015`** | 6,854 | **62.4%** | **100.0%** | **2,372 ms** | *Edge case:* largest trip (~1.9 h). 99.26% reference-arc recall but 36% precision, because 80% of its errors are the opposite-direction twin of a route arc. Use only with the prepared defence in `FINAL_REVIEW_GRILL_GUIDE.md` Section 6. |

---

## 6. Emergency Troubleshooting Playbook

### Scenario A: Laptop 2 shows `❌ Cannot reach Laptop 1 API`
- **Fix:**
  1. Check Wi-Fi: Did one laptop switch to campus Wi-Fi? Ensure both are on the same phone hotspot.
  2. Verify IP: Run `ifconfig` or `ipconfig` on Laptop 1. Did the IP address change? Update the text box on Laptop 2.
  3. Firewall: Check if macOS / Windows popped up an incoming connection warning on Laptop 1. Click "Allow".

### Scenario B: Port 8000 already in use on Laptop 1
- **Fix:**
  ```bash
  kill -9 $(lsof -t -i:8000)
  uvicorn api.main:app --host 0.0.0.0 --port 8000
  ```

### Scenario C: Streamlit on Laptop 2 is slow or frozen
- **Fix:** Press `R` in the Streamlit browser window to rerun, or Ctrl+C in terminal and run `streamlit run ui/app.py`.
