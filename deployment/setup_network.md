# Two-Laptop Network Setup & Distributed Deployment Guide

**Lead Author:** Vikas Aihole (FastAPI Backend & Deployment Lead)  
**System:** PathSnap Distributed Map-Matching Architecture  
**Hardware Nodes:**
- **Laptop 1 (Vikas Aihole & Saatwik):** Hosts FastAPI Model Serving, Candidate Generation, HMM/Viterbi DP Decoder, and Server-Side Ground-Truth Evaluator.
- **Laptop 2 (Uttam):** Hosts Streamlit Evaluator Interface with interactive multi-layer Folium maps.

---

## 1. Physical & Local Network Setup

Both laptops must reside on the same Local Area Network (LAN) subnet to allow direct HTTP communication without public DNS routing.

### Option A: Common Wi-Fi / Hotspot (Recommended)
1. Connect both Laptop 1 and Laptop 2 to the same dedicated Wi-Fi network or mobile phone personal hotspot (e.g., `PathSnap-Demo-Net`).
2. *Tip:* Avoid institutional guest Wi-Fi networks that enable Client Isolation (AP Isolation), which blocks peer-to-peer communication across devices.

### Option B: Direct Ethernet or USB Tethering (Zero-Latency Fallback)
1. Connect an Ethernet cable directly between Laptop 1 and Laptop 2, or enable USB tethering.
2. Assign static IP addresses (e.g., Laptop 1: `192.168.137.1`, Laptop 2: `192.168.137.2` with subnet mask `255.255.255.0`).

---

## 2. Laptop 1 Configuration (Server Node)

### Step 2.1: Find Laptop 1's LAN IP Address
Open PowerShell or Command Prompt on Laptop 1:
```powershell
ipconfig
```
Look for the active adapter (`Wireless LAN adapter Wi-Fi` or `Ethernet adapter`):
```text
IPv4 Address. . . . . . . . . . . : 192.168.1.105
Subnet Mask . . . . . . . . . . . : 255.255.255.0
Default Gateway . . . . . . . . . : 192.168.1.1
```
*Note down this IPv4 Address (e.g., `192.168.1.105`). This is `<LAPTOP_1_IP>`.*

### Step 2.2: Configure Windows Defender Firewall for Port 8000
Run PowerShell as Administrator on Laptop 1 to allow inbound connections on port 8000:
```powershell
New-NetFirewallRule -DisplayName "PathSnap FastAPI Port 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```
*(Alternatively, when Windows Firewall pops up upon starting Uvicorn, check both "Private networks" and "Public networks" and click "Allow access").*

### Step 2.3: Start the Backend Service Bound to 0.0.0.0
Do NOT bind to `127.0.0.1` (which restricts requests to localhost only). Bind to `0.0.0.0`:
```bash
# Navigate to project repository
cd PathSnap-MapMatching

# Launch Uvicorn server with live terminal logging
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
You will see:
```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 3. Laptop 2 Configuration (Evaluator UI Client Node)

### Step 3.1: Network Connectivity Verification (Ping & cURL)
Open terminal on Laptop 2 and run:
```bash
# Test network reachability
ping <LAPTOP_1_IP>

# Test FastAPI health check endpoint
curl http://<LAPTOP_1_IP>:8000/health
```
Expected output from Laptop 2:
```json
{
  "status": "healthy",
  "service": "PathSnap FastAPI Backend",
  "machine": "Laptop 1",
  "client_connected": "192.168.1.110",
  "timestamp": "2026-11-18T10:31:58.899451",
  "pre_lunch_checkpoint": "READY"
}
```

### Step 3.2: Verify Trip List
```bash
curl http://<LAPTOP_1_IP>:8000/trips
```
Expected output:
```json
{
  "trips": ["00000015", "00000017", "00000023", "00000030", "00000049", "00000059", "00000067", "00000073", "00000087", "00000094"],
  "count": 10
}
```

### Step 3.3: Launch Streamlit Evaluator Interface
On Laptop 2:
```bash
cd PathSnap-MapMatching
streamlit run ui/app.py
```
1. In the Streamlit web browser (`http://localhost:8501`), enter `http://<LAPTOP_1_IP>:8000` into the server URL input box.
2. Select any evaluation trip (e.g., `00000015`).
3. Click **"Snap Route & Evaluate"**.
4. The UI sends a `POST /predict/00000015` request to Laptop 1, retrieves the snapped geometry and ground-truth metrics, and renders the multi-layer Folium map.

---

## 4. Live Evaluator Viva Verification Checklist

During the evaluation, the faculty will observe both screens simultaneously:
- [x] **Laptop 1 Terminal:** Observe live request banners on every prediction:
  ```text
  ================================================================================
  [2026-11-18 10:32:00] [EVALUATION REQUEST] POST /predict/00000015
  [2026-11-18 10:32:00] Client IP: 192.168.1.110 | Trip ID: 00000015
  [2026-11-18 10:32:00] Data Ingestion: Loaded 2,503 GPS points | 8,542 nodes | 18,196 arcs
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
- [x] **Laptop 2 Streamlit UI:** Scorecard displays Top-1 Accuracy, Macro F1, Route Continuity %, and Inference Latency, accompanied by the interactive Folium map with togglable layers (Raw GPS, Bounding Road Arcs, Ground Truth, Reconstructed Route).
- [x] **Security / Data Integrity:** The server `.route` reference stays strictly internal to Laptop 1; only evaluation metrics and rendering polylines are returned to the client.

---

## 5. Troubleshooting & Fallback Procedures

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `curl: (7) Failed to connect to <IP> port 8000: Connection refused` | Server bound to `127.0.0.1` instead of `0.0.0.0` | Restart Uvicorn with `--host 0.0.0.0`. |
| `curl: (28) Connection timed out` | Windows Firewall blocking port 8000 on Laptop 1 | Run `New-NetFirewallRule -DisplayName "FastAPI" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow` on Laptop 1. |
| Laptop 2 cannot ping Laptop 1 | Client Isolation enabled on Wi-Fi router | Disconnect from college Wi-Fi and switch both laptops to a phone hotspot. |
| Inconsistent IP after reconnecting | DHCP reassigned a new IP address | Check `ipconfig` again on Laptop 1 and update the IP URL in the Streamlit UI. |