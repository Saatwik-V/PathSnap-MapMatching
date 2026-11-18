"""
Rehearsal / pre-demo check: calls POST /predict/{trip_id} on a running API for every trip
returned by GET /trips, prints the scorecard metrics and writes them to a JSON file.
Usage:  python deployment/rehearsal_all_trips.py --api http://localhost:8000 --out results/predemo_run.json
Only reads the API's own response; it never touches training or tuning.
"""
import argparse
import json
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--api", default="http://localhost:8000")
ap.add_argument("--out", default="results/predemo_run.json")
args = ap.parse_args()

trips = requests.get(f"{args.api}/trips", timeout=10).json()["trips"]
rows = []
for t in trips:
    r = requests.post(f"{args.api}/predict/{t}", timeout=120)
    m = r.json().get("metrics", {}) if r.status_code == 200 else {}
    rows.append({"trip_id": t, "status": r.status_code, "metrics": m})
    print(t, r.status_code, {k: m.get(k) for k in ("top1_accuracy", "macro_f1", "route_continuity", "inference_latency_ms")})
json.dump(rows, open(args.out, "w"), indent=2)
ok = [x["metrics"] for x in rows if x["status"] == 200]
if ok:
    for k in ("top1_accuracy", "macro_f1", "route_continuity", "inference_latency_ms"):
        print(f"mean {k}: {sum(m[k] for m in ok) / len(ok):.4f}")
