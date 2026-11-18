"""Render split summaries and a development GPS/road/reference map."""

import argparse
import csv
from pathlib import Path

import folium
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from pathsnap.data.development import DEVELOPMENT_SPLITS
from pathsnap.data.loading import load_trip
from pathsnap.data.reference_alignment import read_reference_route, reference_break_positions

GPS_COLOR = "#1D62BA"
ROAD_COLOR = "#9AA5B3"
REF_COLOR = "#148258"
SPLIT_COLORS = {"train": "#1D62BA", "student_test": "#C4602D"}


def _nearby_arcs(trip):
    low = trip.observation_xy.min(axis=0) - 200
    high = trip.observation_xy.max(axis=0) + 200
    segments = trip.node_xy[trip.arc_nodes]
    intersects_box = ((segments.max(axis=1) >= low) & (segments.min(axis=1) <= high)).all(axis=1)
    return np.flatnonzero(intersects_box), low, high


def _plot_trip(trip, route, split, road_ids, low, high, output):
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    ax.add_collection(LineCollection(trip.node_xy[trip.arc_nodes[road_ids]], colors=ROAD_COLOR,
                                     linewidths=0.45, alpha=0.6))
    ax.add_collection(LineCollection(trip.node_xy[trip.arc_nodes[route]], colors=REF_COLOR, linewidths=1.8))
    x, y = trip.observation_xy.T
    ax.scatter(x, y, s=5, color=GPS_COLOR, alpha=0.65, linewidths=0)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal",
           xlabel="Local east (m)", ylabel="Local north (m)",
           title=f"GPS and road geometry  ·  {split}/{trip.trip_id}")
    fig.legend(handles=[
        Line2D([], [], color=ROAD_COLOR, label="Road network"),
        Line2D([], [], marker="o", color=GPS_COLOR, linestyle="none", label="GPS observations"),
        Line2D([], [], color=REF_COLOR, label="Supplied reference"),
    ], loc="outside lower center", ncols=3)
    fig.savefig(output / "gps_and_road_geometry.png", dpi=175)
    plt.close(fig)


def _folium_map(trip, route, split, road_ids, output):
    center = trip.track[:, :2].mean(axis=0)[::-1].tolist()
    map_view = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=13, control_scale=True)
    roads = folium.FeatureGroup(name="Road network").add_to(map_view)
    gps = folium.FeatureGroup(name="GPS observations").add_to(map_view)
    reference = folium.FeatureGroup(name="Supplied reference route").add_to(map_view)
    folium.PolyLine(trip.node_lonlat[trip.arc_nodes[road_ids]][:, :, ::-1].tolist(),
                    color=ROAD_COLOR, weight=1, opacity=0.55).add_to(roads)
    for obs_idx, (lon, lat, time_s) in enumerate(trip.track):
        folium.CircleMarker(
            [lat, lon], radius=2, stroke=False, fill=True, fill_color=GPS_COLOR,
            fill_opacity=0.8, tooltip=f"Observation {obs_idx} · {time_s:g} s",
        ).add_to(gps)
    for part in np.split(route, reference_break_positions(trip, route)):
        endpoints = trip.arc_nodes[part]
        nodes = np.r_[endpoints[0, 0], endpoints[:, 1]]
        folium.PolyLine(trip.node_lonlat[nodes][:, ::-1].tolist(),
                        color=REF_COLOR, weight=4, opacity=0.85).add_to(reference)
    latlon = trip.track[:, :2][:, ::-1]
    map_view.fit_bounds([latlon.min(axis=0).tolist(), latlon.max(axis=0).tolist()])
    folium.LayerControl(collapsed=False).add_to(map_view)
    legend = f"""
    <div style="position:fixed;bottom:30px;left:12px;z-index:1000;background:white;
                border:1px solid #cdd6df;border-radius:6px;padding:10px 14px;
                font:13px/1.6 system-ui;color:#182333;">
      <strong>PathSnap · {split}/{trip.trip_id}</strong><br>
      <span style="color:{GPS_COLOR}">●</span> GPS observations<br>
      <span style="color:{ROAD_COLOR}">━</span> Road network<br>
      <span style="color:{REF_COLOR}">━</span> Supplied reference
    </div>
    """
    map_view.get_root().html.add_child(folium.Element(legend))
    map_view.save(output / "development_map.html")


def _plot_summaries(inventories, output):
    rows = []
    for inventory in inventories:
        with inventory.open(encoding="utf-8", newline="") as stream:
            rows.extend(row for row in csv.DictReader(stream) if row["valid"] == "True")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    for split, color in SPLIT_COLORS.items():
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            continue
        label = f"{split} ({len(subset)} trips)"
        sizes = sorted(int(row["gps_rows"]) for row in subset)
        axes[0].step(sizes, 100 * np.arange(1, len(sizes) + 1) / len(sizes),
                     where="post", color=color, marker=".", markersize=3, label=label)
        axes[1].scatter([int(row["nodes"]) for row in subset], [int(row["arcs"]) for row in subset],
                        color=color, s=22, alpha=0.75, label=label)
        axes[2].scatter([float(row["max_gap_s"]) for row in subset],
                        [int(row["gaps_gt_1s"]) for row in subset],
                        color=color, s=22, alpha=0.75, label=label)
    axes[0].set(title="Trajectory size", xlabel="GPS observations per trip", ylabel="Cumulative trips (%)")
    axes[1].set(title="Graph size", xlabel="Nodes per trip", ylabel="Directed arcs per trip",
                xscale="log", yscale="log")
    axes[2].set(title="Irregular GPS sampling", xlabel="Longest interval per trip (s)",
                ylabel="Intervals above 1 s", xscale="symlog", yscale="symlog")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.savefig(output / "development_split_eda.png", dpi=175)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=DEVELOPMENT_SPLITS, default="train")
    parser.add_argument("--trip-id", default="00000048")
    parser.add_argument("--inventory", type=Path, action="append", default=[])
    args = parser.parse_args()
    if Path(args.trip_id).name != args.trip_id or args.trip_id in {".", ".."}:
        parser.error("--trip-id must be a trip folder name")
    trip_dir = args.dataset / args.split / args.trip_id
    trip = load_trip(trip_dir)
    route = read_reference_route(trip, trip_dir)
    road_ids, low, high = _nearby_arcs(trip)
    args.output.mkdir(parents=True, exist_ok=True)
    _plot_trip(trip, route, args.split, road_ids, low, high, args.output)
    _folium_map(trip, route, args.split, road_ids, args.output)
    if args.inventory:
        _plot_summaries(args.inventory, args.output)


if __name__ == "__main__":
    main()
