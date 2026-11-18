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


def folium_eda(trip=TRIP):
    track, nodes, arcs, route = _load(trip)
    m = build_trip_map(track, nodes, arcs, route, [], trip)
    path = f"results/maps/eda_train_trip_{trip}_map.html"
    m.save(path)
    print("Saved:", path)


if __name__ == "__main__":
    plot_eda()
    folium_eda()
