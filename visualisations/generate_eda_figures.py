"""
EDA figures for the technical report (training partition only): GPS trajectory plot,
road-network visualisation, and a Folium map of one training trip.
Outputs: results/plots/eda_trip_<id>_network.png, results/maps/eda_train_trip_<id>_map.html
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualisations.folium_map import build_trip_map

from data_processing.eda import classify_region

BASE = "Student folder/train"
TRIP = "00000061"


def _load(trip):
    p = os.path.join(BASE, trip, trip)
    track = pd.read_csv(p + ".track", sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
    nodes = pd.read_csv(p + ".nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(p + ".arcs", sep=r"\s+", header=None, names=["u", "v"])
    route = pd.read_csv(p + ".route", sep=r"\s+", header=None, names=["arc_id"])["arc_id"].tolist()
    return track, nodes, arcs, route


def _segments(arc_ids, nodes, arcs):
    a = arcs.iloc[arc_ids]
    return [((nodes.lon[u], nodes.lat[u]), (nodes.lon[v], nodes.lat[v])) for u, v in zip(a.u, a.v)]


def plot_eda(trip=TRIP, out="results/plots/eda_trip_%s_network.png"):
    track, nodes, arcs, route = _load(trip)
    pad = 0.002
    x0, x1 = track.lon.min() - pad, track.lon.max() + pad
    y0, y1 = track.lat.min() - pad, track.lat.max() + pad
    inb = (nodes.lon.between(x0, x1) & nodes.lat.between(y0, y1)).values
    local = arcs[inb[arcs.u.values] & inb[arcs.v.values]].index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
    sc = axes[0].scatter(track.lon, track.lat, c=np.arange(len(track)), s=5, cmap="viridis")
    axes[0].set_title(f"(a) Raw GPS trajectory\n{len(track):,} points, coloured by time", fontsize=10)
    plt.colorbar(sc, ax=axes[0], fraction=0.046, label="observation index")

    for ax in axes[1:]:
        for (a, b) in _segments(local, nodes, arcs):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#b0b0b0", linewidth=0.5, zorder=1)
    axes[1].set_title(f"(b) Road network in the trip's bounding box\n{len(local):,} of {len(arcs):,} arcs shown", fontsize=10)

    axes[2].scatter(track.lon, track.lat, s=4, color="#1f77b4", alpha=0.5, zorder=2, label="Raw GPS")
    for i, (a, b) in enumerate(_segments(route, nodes, arcs)):
        axes[2].plot([a[0], b[0]], [a[1], b[1]], color="#2ca02c", linewidth=2.5, zorder=3,
                     label="Reference route" if i == 0 else None)
    axes[2].set_title(f"(c) GPS + network + reference route\n{len(route)} reference arcs", fontsize=10)
    axes[2].legend(loc="best", fontsize=8)

    for ax in axes:
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect(1 / np.cos(np.radians(track.lat.mean())))
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    fig.suptitle(f"Training partition, trip {trip}", fontweight="bold")
    path = out % trip
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)


NAMED_REGIONS = ["Russia / Volga-Urals", "Germany / Central Europe", "North America", "Poland / Eastern Europe", "Japan"]
REGION_COLORS = {"Russia / Volga-Urals": "#d62728", "Germany / Central Europe": "#1f77b4", "North America": "#2ca02c",
                 "Poland / Eastern Europe": "#9467bd", "Japan": "#ff7f0e", "Other locations": "#7f7f7f"}


def plot_geographic_regions(df_stats, out="results/plots/geographic_regions.png"):
    """
    df_stats needs columns: mean_lon, mean_lat, region (one row per training trip).
    Left: trips per named region, with the long tail of 1-2 trip locations collapsed into one bar.
    Right: the actual trip centroids, so the geographic spread is visible rather than listed.
    """
    df = df_stats.copy()
    df["group"] = df["region"].where(df["region"].isin(NAMED_REGIONS), "Other locations")
    n_other_sites = df.loc[df["group"] == "Other locations", "region"].nunique()
    counts = df["group"].value_counts()
    order = [g for g in NAMED_REGIONS if g in counts.index] + (["Other locations"] if "Other locations" in counts.index else [])
    labels = [f"Other locations\n({n_other_sites} distinct sites, 1-2 trips each)" if g == "Other locations" else g for g in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1.25]}, constrained_layout=True)
    y = np.arange(len(order))[::-1]
    ax1.barh(y, [counts[g] for g in order], color=[REGION_COLORS[g] for g in order], edgecolor="black", alpha=0.9)
    for yi, g in zip(y, order):
        ax1.text(counts[g] + 0.6, yi, f"{counts[g]} ({counts[g] / len(df):.0%})", va="center", fontweight="bold", fontsize=10)
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlim(0, counts.max() + 8); ax1.set_xlabel("Number of training trips")
    ax1.set_title(f"(a) Trips per region ({len(df)} training trips)", fontweight="bold", fontsize=11)

    for g in order[::-1]:
        sub = df[df["group"] == g]
        ax2.scatter(sub["mean_lon"], sub["mean_lat"], s=55 if g != "Other locations" else 40, color=REGION_COLORS[g],
                    edgecolor="black", linewidth=0.5, alpha=0.85, label=f"{g} ({len(sub)})", zorder=3)
    ax2.set_xlim(-130, 150); ax2.set_ylim(-45, 75)
    ax2.set_xlabel("Longitude (trip centroid)"); ax2.set_ylabel("Latitude (trip centroid)")
    ax2.grid(True, alpha=0.3); ax2.legend(loc="lower left", fontsize=8, framealpha=0.95)
    ax2.set_title("(b) Trip centroids: trips span several continents", fontweight="bold", fontsize=11)
    fig.suptitle("Geographic dispersion of the training trips: why each trip needs its own local projection", fontweight="bold")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


def compute_centroids(base=BASE):
    rows = []
    for t in sorted(os.listdir(base)):
        tr = pd.read_csv(os.path.join(base, t, f"{t}.track"), sep=r"\s+", header=None, names=["lon", "lat", "ts"])
        lon, lat = float(tr.lon.mean()), float(tr.lat.mean())
        rows.append({"trip_id": t, "mean_lon": lon, "mean_lat": lat, "region": classify_region(lon, lat)})
    return pd.DataFrame(rows)


def folium_eda(trip=TRIP):
    track, nodes, arcs, route = _load(trip)
    m = build_trip_map(track, nodes, arcs, route, [], trip)
    path = f"results/maps/eda_train_trip_{trip}_map.html"
    m.save(path)
    print("Saved:", path)


if __name__ == "__main__":
    plot_geographic_regions(compute_centroids())
    plot_eda()
    folium_eda()
