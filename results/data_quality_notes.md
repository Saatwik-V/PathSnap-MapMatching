# PathSnap Data Quality & Exploratory Analysis Notes

**Author:** Vikas Jadar (Data Pipeline & Feature Extraction Specialist)
**Role:** Data Foundation Lead

## 1. Executive Summary of Audited Dataset

- **Total Training Trips:** 80 independent trips in `Student folder/train/`
- **Total GPS Timesteps:** 208,417 points
- **Missing / Invalid Values:** Zero NaN coordinates, zero null values, zero empty files.
- **File Formats Confirmed:**
  - `.track`: whitespace-delimited (3 columns: `lon, lat, timestamp`). No headers.
  - `.nodes`: whitespace-delimited (2 columns: `lon, lat`). 0-indexed row index = Node ID.
  - `.arcs`: whitespace-delimited (2 columns: `from_node, to_node`). 0-indexed row index = Arc ID.
  - `.route`: 1 column sequence of integer Arc IDs representing chronological ground truth.

## 2. Temporal Characteristics & GPS Dropout Quirks

- **Nominal Sampling Rate:** ~1.000 Hz (96.51% of intervals exactly equal 1.0s).
- **Mean $\Delta t$:** 2.181 seconds; **Median $\Delta t$:** 1.000 seconds.
- **Max Observed $\Delta t$ Gap:** 153064.0 seconds (Observed in Trip `00000011`).
- **Gaps Exceeding 5.0 Seconds:** 1,216 occurrences (0.584%).
- **Severe Dropouts (> 30.0s):** 133 occurrences across 28 trips.
  - *Technical Implication:* During GPS dropouts (e.g. urban canyons, tunnels, hardware pauses), vehicle velocity cannot be assumed constant. Transition probabilities in the HMM must expand their spatial search radius dynamically as $\Delta t$ increases.

## 3. Geographic Multi-Region Characteristics

The dataset is not confined to a single city; trips originate from distinct global territories:
- **Russia / Volga-Urals:** 17 trips (21.2%)
- **Germany / Central Europe:** 9 trips (11.2%)
- **North America:** 6 trips (7.5%)
- **Poland / Eastern Europe:** 5 trips (6.2%)
- **Japan:** 3 trips (3.8%)
- **Other (43.4N, -3.8E):** 2 trips (2.5%)
- **Other (56.0N, 12.0E):** 2 trips (2.5%)
- **Other (49.5N, 35.5E):** 1 trips (1.2%)
- **Other (36.2N, 44.3E):** 1 trips (1.2%)
- **Other (43.8N, 1.8E):** 1 trips (1.2%)
- **Other (40.6N, -8.5E):** 1 trips (1.2%)
- **Other (15.4N, 106.5E):** 1 trips (1.2%)
- **Other (47.9N, -3.6E):** 1 trips (1.2%)
- **Other (40.3N, 44.4E):** 1 trips (1.2%)
- **Other (41.9N, 12.6E):** 1 trips (1.2%)
- **Other (59.6N, 30.1E):** 1 trips (1.2%)
- **Other (53.9N, 27.5E):** 1 trips (1.2%)
- **Other (60.4N, 24.4E):** 1 trips (1.2%)
- **Other (40.6N, -8.7E):** 1 trips (1.2%)
- **Other (10.5N, 124.0E):** 1 trips (1.2%)
- **Other (52.1N, 5.0E):** 1 trips (1.2%)
- **Other (48.5N, 3.0E):** 1 trips (1.2%)
- **Other (53.0N, -1.8E):** 1 trips (1.2%)
- **Other (41.7N, 12.6E):** 1 trips (1.2%)
- **Other (52.3N, 4.9E):** 1 trips (1.2%)
- **Other (45.2N, -0.5E):** 1 trips (1.2%)
- **Other (0.3N, -78.1E):** 1 trips (1.2%)
- **Other (51.0N, 2.3E):** 1 trips (1.2%)
- **Other (55.9N, 9.8E):** 1 trips (1.2%)
- **Other (44.2N, 22.5E):** 1 trips (1.2%)
- **Other (53.2N, -1.7E):** 1 trips (1.2%)
- **Other (47.5N, -52.8E):** 1 trips (1.2%)
- **Other (48.7N, -1.5E):** 1 trips (1.2%)
- **Other (-26.2N, 27.9E):** 1 trips (1.2%)
- **Other (38.7N, -9.1E):** 1 trips (1.2%)
- **Other (44.3N, 3.9E):** 1 trips (1.2%)
- **Other (-34.6N, -58.5E):** 1 trips (1.2%)
- **Other (59.9N, 30.6E):** 1 trips (1.2%)
- **Other (61.2N, 73.4E):** 1 trips (1.2%)
- **Other (60.0N, 30.3E):** 1 trips (1.2%)
- **Other (52.9N, -2.9E):** 1 trips (1.2%)
- **Other (50.7N, -1.3E):** 1 trips (1.2%)
- **Other (-29.5N, -51.4E):** 1 trips (1.2%)

*Design Decision:* A hardcoded UTM zone or national projection is invalid. All geometric calculations strictly use **dynamic local equirectangular metric projections** centered at each trip's local centroid $(\phi_0, \lambda_0)$, where 1 degree latitude $\equiv 111,320$ m and 1 degree longitude $\equiv 111,320 	imes \cos(\phi_0)$ m.

## 4. Road Network Graph Topology

- Average nodes per trip road graph: 24169 nodes (Max: 155,003)
- Average arcs per trip road graph: 48461 arcs (Max: 307,392)
- Ground-truth route connectivity: Verified at 100% continuous graph walks across consecutive route arcs.
- Average points per route arc: 6.65 GPS points/arc.

## 5. Candidate Generation & Feature Formulation Assumptions

1. **Spatial Indexing:** Because graph arc counts exceed 160,000 per trip, brute-force search is computationally infeasible ($O(N \cdot M) > 2 	imes 10^8$ ops/trip). We index all arc midpoints using `scipy.spatial.cKDTree` and retrieve the top-150 spatial neighbors before computing exact perpendicular point-to-segment distance.
2. **Ground-Truth Sequence Mapping:** Ground-truth routes provide an ordered list of arcs without explicit per-point timestamps. We map each GPS observation $P_t$ to the ground truth route using a normalized temporal window along the route sequence, selecting the segment with minimal perpendicular distance within the window.
3. **Top-K Candidates:** We retain $K=10$ nearest candidate road segments per GPS point. A candidate is assigned binary label `y = 1` if its `arc_id` matches the aligned ground-truth route arc, and `y = 0` otherwise.
4. **Zero Data Leakage:** Candidate generation and feature extraction operate purely trip-by-trip without any cross-trip aggregation or global normalization.
