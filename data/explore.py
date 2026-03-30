"""
explore.py
Dataset exploration and visualisation for nuScenes mini.
Produces class distribution charts, visibility analysis,
and per-scene object density — all saved to outputs/week1/.
"""

import os
import matplotlib.pyplot as plt
print("Matplotlib works")
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from scene_parser import (
    load_config,
    parse_scenes,
    parse_samples,
    parse_annotations,
    get_class_distribution,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "week1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette (consistent across all plots)
PALETTE = {
    "car": "#378ADD",
    "pedestrian": "#D85A30",
    "bicycle": "#1D9E75",
    "motorcycle": "#BA7517",
    "bus": "#534AB7",
    "truck": "#993556",
    "traffic_cone": "#5F5E5A",
    "barrier": "#888780",
}


def plot_class_distribution(ann_df: pd.DataFrame):
    """Bar chart of annotation counts per class."""
    dist = get_class_distribution(ann_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PALETTE.get(c, "#888780") for c in dist["class"]]
    bars = ax.bar(dist["class"], dist["count"], color=colors, width=0.6)

    # Annotate bars with percentage
    for bar, pct in zip(bars, dist["percentage"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=9, color="#444441"
        )

    ax.set_title("Class distribution — nuScenes mini", fontsize=13, pad=12)
    ax.set_xlabel("Object class")
    ax.set_ylabel("Annotation count")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print("Saved: class_distribution.png")


def plot_visibility_analysis(ann_df: pd.DataFrame):
    """
    Visibility token in nuScenes = 1 (0-40%), 2 (40-60%),
    3 (60-80%), 4 (80-100%).
    Low visibility objects are the hard safety-critical cases.
    """
    visibility_labels = {
        "1": "0–40% visible",
        "2": "40–60% visible",
        "3": "60–80% visible",
        "4": "80–100% visible",
        "": "unknown",
    }

    ann_df["visibility_label"] = ann_df["visibility_token"].map(
        lambda x: visibility_labels.get(str(x), "unknown")
    )

    vis_dist = ann_df["visibility_label"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_vis = ["#E24B4A", "#EF9F27", "#1D9E75", "#378ADD", "#888780"]
    vis_dist.plot(kind="barh", ax=ax, color=colors_vis[:len(vis_dist)])

    ax.set_title("Object visibility distribution — safety-critical analysis", fontsize=12, pad=10)
    ax.set_xlabel("Count")
    ax.spines[["top", "right"]].set_visible(False)

    # Annotation: flag low visibility as safety risk
    ax.axvline(x=0, color="black", linewidth=0.5)
    fig.text(
        0.98, 0.02,
        "Low visibility (0-40%) = highest safety risk for perception models",
        ha="right", fontsize=8, color="#A32D2D"
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "visibility_analysis.png", dpi=150)
    plt.close()
    print("Saved: visibility_analysis.png")


def plot_object_density_per_scene(ann_df: pd.DataFrame, samples_df: pd.DataFrame):
    """
    Objects per sample — identifies dense scenes for scene mining.
    High density = urban scenarios most challenging for perception.
    """
    obj_per_sample = ann_df.groupby("sample_token").size().reset_index(name="object_count")
    merged = obj_per_sample.merge(samples_df[["token", "scene_token"]], 
                                   left_on="sample_token", right_on="token")
    density_per_scene = merged.groupby("scene_token")["object_count"].mean().reset_index()
    density_per_scene.columns = ["scene_token", "avg_objects_per_sample"]
    density_per_scene = density_per_scene.sort_values("avg_objects_per_sample", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        range(len(density_per_scene)),
        density_per_scene["avg_objects_per_sample"],
        color="#378ADD", alpha=0.8
    )

    # Mark high-density threshold (scene mining candidate)
    threshold = density_per_scene["avg_objects_per_sample"].quantile(0.75)
    ax.axhline(y=threshold, color="#E24B4A", linewidth=1.2, linestyle="--",
               label=f"Top 25% density threshold ({threshold:.1f} obj/sample)")

    ax.set_title("Average object density per scene — scene mining candidates", fontsize=12, pad=10)
    ax.set_xlabel("Scene index (sorted by density)")
    ax.set_ylabel("Avg objects per sample")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "object_density_per_scene.png", dpi=150)
    plt.close()
    print("Saved: object_density_per_scene.png")


def plot_lidar_pts_distribution(ann_df: pd.DataFrame):
    """
    LiDAR point count per annotation.
    Low point count = sparse/distant objects = harder to detect.
    ADAS safety relevance: far pedestrians may have <5 LiDAR pts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Overall distribution (log scale)
    axes[0].hist(
        ann_df["num_lidar_pts"].clip(upper=200),
        bins=40, color="#1D9E75", edgecolor="white", linewidth=0.5
    )
    axes[0].set_title("LiDAR points per annotation (clipped at 200)", fontsize=11)
    axes[0].set_xlabel("LiDAR point count")
    axes[0].set_ylabel("Frequency")
    axes[0].spines[["top", "right"]].set_visible(False)

    # Median LiDAR pts by class
    median_pts = ann_df.groupby("category")["num_lidar_pts"].median().sort_values()
    colors_pts = [PALETTE.get(c, "#888780") for c in median_pts.index]
    median_pts.plot(kind="barh", ax=axes[1], color=colors_pts)
    axes[1].set_title("Median LiDAR points by class", fontsize=11)
    axes[1].set_xlabel("Median point count")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "LiDAR density analysis — low point counts indicate safety-critical detection risk",
        fontsize=10, color="#A32D2D", y=1.01
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lidar_pts_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: lidar_pts_distribution.png")


def generate_summary_report(scenes_df, samples_df, ann_df):
    """Print and save a structured dataset summary."""
    dist = get_class_distribution(ann_df)
    summary = f"""
=== nuScenes Mini — Dataset Exploration Report ===

Dataset stats:
  Scenes:      {len(scenes_df)}
  Samples:     {len(samples_df)}
  Annotations: {len(ann_df)}
  Classes:     {ann_df['category'].nunique()}

Class distribution:
{dist.to_string(index=False)}

Safety-critical observations:
  Low visibility objects (0-40%): {(ann_df['visibility_token'] == '1').sum()} 
  Zero LiDAR point objects:       {(ann_df['num_lidar_pts'] == 0).sum()}
  Pedestrian annotations:         {(ann_df['category'] == 'pedestrian').sum()}

Scene density:
  Min objects/sample: {ann_df.groupby('sample_token').size().min()}
  Max objects/sample: {ann_df.groupby('sample_token').size().max()}
  Mean objects/sample: {ann_df.groupby('sample_token').size().mean():.1f}

BMW relevance note:
  High-density scenes and low-visibility objects are primary
  candidates for scene mining in Week 3 (closed-loop evaluation).
"""
    print(summary)
    with open(OUTPUT_DIR / "week1_summary.txt", "w") as f:
        f.write(summary)
    print("Saved: week1_summary.txt")


if __name__ == "__main__":
    cfg = load_config()
    data_root = cfg["dataset"]["root"]

    print("Loading dataset...")
    scenes_df = parse_scenes(data_root)
    samples_df = parse_samples(data_root)
    ann_df = parse_annotations(data_root)

    print("\nGenerating plots...")
    plot_class_distribution(ann_df)
    plot_visibility_analysis(ann_df)
    plot_object_density_per_scene(ann_df, samples_df)
    plot_lidar_pts_distribution(ann_df)

    generate_summary_report(scenes_df, samples_df, ann_df)
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")