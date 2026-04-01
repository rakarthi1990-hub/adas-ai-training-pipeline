"""
mine_scenarios.py
Safety-critical scene mining for ADAS model improvement.

Mines nuScenes val split for scenarios where the model is most
likely to fail — based on ODD conditions, not just confidence scores.
This mirrors production data-driven development: identify failure
conditions first, then collect targeted data.

BMW relevance: closed-loop evaluation requires knowing WHICH scenes
drive KPI degradation, not just overall mAP.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import yaml
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "data"))

from scene_parser import (
    load_config, parse_scenes, parse_samples,
    parse_annotations, get_class_distribution
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "week3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = BASE_DIR / "data" / "nuscenes"

sys.path.append(str(BASE_DIR / "data"))

# Safety thresholds — deliberately conservative (ADAS safety logic)
THRESHOLDS = {
    "low_visibility_token":    "1",   # 0–40% visible — unchanged, token-based
    "dense_scene_min_objects":  20,   # raised from 10 → top-density scenes only
    "small_object_max_area":  0.002,  # tightened: <0.2% image area (very distant)
    "pedestrian_risk_min":      3,    # raised from 2 → 3+ pedestrians
}

SAFETY_COLORS = {
    "low_visibility":   "#E24B4A",
    "dense_urban":      "#EF9F27",
    "small_objects":    "#378ADD",
    "pedestrian_risk":  "#D85A30",
}


def load_visibility_map(data_root: Path) -> dict:
    v = data_root / "v1.0-mini"
    with open(v / "visibility.json") as f:
        vis = json.load(f)
    return {v["token"]: v["level"] for v in vis}


def mine_low_visibility(ann_df: pd.DataFrame,
                        samples_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scenes with objects at 0-40% visibility.
    These are the hardest cases for camera-based perception —
    directly relevant to SOTIF (ISO/PAS 21448) unknown unsafe scenarios.
    """
    low_vis = ann_df[ann_df["visibility_token"] == "1"]
    risky_samples = low_vis["sample_token"].value_counts().reset_index()
    risky_samples.columns = ["sample_token", "low_vis_object_count"]
    risky_samples = risky_samples[risky_samples["low_vis_object_count"] >= 3].copy()
    risky_samples["mining_reason"] = "low_visibility"
    risky_samples["safety_priority"] = "critical"
    return risky_samples


def mine_dense_scenes(ann_df: pd.DataFrame) -> pd.DataFrame:
    """
    High object density frames — urban scenarios.
    Dense scenes cause overlapping detections, occlusion,
    and confidence instability (validated in Week 1 analysis).
    """
    obj_per_sample = ann_df.groupby("sample_token").size().reset_index(
        name="object_count")
    dense = obj_per_sample[
        obj_per_sample["object_count"] >= THRESHOLDS["dense_scene_min_objects"]
    ].copy()
    dense["mining_reason"] = "dense_urban"
    dense["safety_priority"] = "high"
    dense = dense.rename(columns={"object_count": "low_vis_object_count"})
    return dense


def mine_small_objects(ann_df: pd.DataFrame) -> pd.DataFrame:
    """
    Distant small objects — box area < 0.5% of image.
    Small objects (distant pedestrians, far cyclists) are the
    primary cause of missed detections at low recall.
    Week 2 finding: recall 0.21 is driven by these cases.
    """
    # Box area from label files (width * height in YOLO normalised space)
    labels_dir = Path("../datasets/nuscenes_yolo/labels")
    records = []

    for split in ["train", "val"]:
        label_path = labels_dir / split
        if not label_path.exists():
            continue
        for label_file in label_path.glob("*.txt"):
            token = label_file.stem
            with open(label_file) as f:
                lines = f.read().strip().split("\n")
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, w, h = map(float, parts)
                    area = w * h
                    if area < THRESHOLDS["small_object_max_area"]:
                        records.append({
                            "sample_token": token,
                            "box_area": area,
                        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    small_counts = df.groupby("sample_token").size().reset_index(
        name="low_vis_object_count")
    small_counts["mining_reason"] = "small_objects"
    small_counts["safety_priority"] = "high"
    return small_counts


def mine_pedestrian_risk(ann_df: pd.DataFrame) -> pd.DataFrame:
    """
    Frames with 2+ pedestrians.
    Pedestrian AP@50 = 0.058 (Week 2 result) — highest safety risk.
    Multi-pedestrian scenes are most demanding for detection recall.
    """
    ped = ann_df[ann_df["category"] == "pedestrian"]
    ped_per_sample = ped.groupby("sample_token").size().reset_index(
        name="pedestrian_count")
    risky = ped_per_sample[
        ped_per_sample["pedestrian_count"] >= THRESHOLDS["pedestrian_risk_min"]
    ].copy()
    risky["mining_reason"] = "pedestrian_risk"
    risky["safety_priority"] = "critical"
    risky = risky.rename(columns={"pedestrian_count": "low_vis_object_count"})
    return risky


def plot_mined_scenes_summary(mined_df: pd.DataFrame):
    """Bar chart of mined scenes per category with safety priority colours."""
    counts = mined_df.groupby("mining_reason").size().reset_index(
        name="scene_count")
    colors = [SAFETY_COLORS.get(r, "#888780")
              for r in counts["mining_reason"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(counts["mining_reason"], counts["scene_count"],
                  color=colors, width=0.5)

    for bar, count in zip(bars, counts["scene_count"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontsize=10)

    ax.set_title(
        "Safety-critical scene mining — mined scenario counts by ODD category",
        fontsize=12, pad=10)
    ax.set_ylabel("Scenes mined")
    ax.set_xlabel("Mining category")
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    patches = [mpatches.Patch(color=c, label=r.replace("_", " ").title())
               for r, c in SAFETY_COLORS.items()]
    ax.legend(handles=patches, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mined_scenes_summary.png", dpi=150)
    plt.close()
    print("Saved: mined_scenes_summary.png")


def plot_object_size_distribution(ann_df: pd.DataFrame):
    """
    Distribution of object box areas — shows the long tail of
    small distant objects that drive recall degradation.
    """
    labels_dir = BASE_DIR / "datasets" / "nuscenes_yolo" / "labels"
    areas = []
    class_ids = []

    for split in ["train", "val"]:
        for label_file in (labels_dir / split).glob("*.txt"):
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid, cx, cy, w, h = map(float, parts)
                        areas.append(w * h)
                        class_ids.append(int(cid))

    if not areas:
        print("No label data for size distribution plot.")
        return

    areas = np.array(areas)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Area distribution (log scale shows the small-object long tail)
    axes[0].hist(np.log10(areas + 1e-8), bins=50,
                 color="#378ADD", edgecolor="white", linewidth=0.3)
    axes[0].axvline(x=np.log10(0.005), color="#E24B4A",
                    linewidth=1.5, linestyle="--",
                    label="Small object threshold (0.5%)")
    axes[0].set_title("Object size distribution (log scale)", fontsize=11)
    axes[0].set_xlabel("log10(box area)")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Percentage of objects below each size threshold
    thresholds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    pcts = [(areas < t).mean() * 100 for t in thresholds]
    axes[1].plot([str(t) for t in thresholds], pcts,
                 "o-", color="#E24B4A", linewidth=2)
    axes[1].fill_between(range(len(thresholds)), pcts,
                         alpha=0.15, color="#E24B4A")
    axes[1].set_title("Cumulative % objects below size threshold", fontsize=11)
    axes[1].set_xlabel("Box area threshold")
    axes[1].set_ylabel("% of all objects")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Small object analysis — root cause of low recall (Week 2: recall 0.21)",
        fontsize=10, color="#A32D2D", y=1.02
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "object_size_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: object_size_distribution.png")


def generate_mining_report(mined_df: pd.DataFrame,
                           ann_df: pd.DataFrame):
    """Structured report linking mined scenes to safety implications."""
    total_samples = ann_df["sample_token"].nunique()
    mined_unique  = mined_df["sample_token"].nunique()

    report_lines = [
        "=== Week 3 — Scene Mining Report ===\n",
        f"Total samples analysed: {total_samples}",
        f"Safety-critical scenes mined: {mined_unique} "
        f"({mined_unique/total_samples*100:.1f}%)\n",
        "Breakdown by mining category:",
    ]

    for reason, group in mined_df.groupby("mining_reason"):
        n = group["sample_token"].nunique()
        pct = n / total_samples * 100
        priority = group["safety_priority"].iloc[0]
        report_lines.append(
            f"  {reason:<22} {n:>4} scenes  ({pct:5.1f}%)  "
            f"priority: {priority}"
        )

    report_lines += [
        "\nConnection to Week 2 training results:",
        "  Recall 0.21 on val split — model misses ~79% of objects.",
        "  Root causes identified by scene mining:",
        "  1. Low visibility objects → model confidence collapses",
        "  2. Small distant objects → below model detection threshold",
        "  3. Dense scenes → overlapping detections, suppressed by NMS",
        "  4. Multi-pedestrian scenes → safety-critical missed detections",
        "\nProduction implication (ISO 26262 / SOTIF):",
        "  These mined scenes define the data collection priority for",
        "  the next training iteration. Targeted augmentation of",
        "  low-visibility and small-object scenarios is expected to",
        "  improve recall from 0.21 toward 0.50+ in 2-3 iterations.",
        "\nWeek 4 action:",
        "  Safety failure chain analysis per mined category.",
        "  SOTIF mapping: unknown unsafe scenarios → known unsafe → mitigated.",
    ]

    report = "\n".join(report_lines)
    print(report)

    with open(OUTPUT_DIR / "week3_mining_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {OUTPUT_DIR / 'week3_mining_report.txt'}")


def run_mining():
    print("Loading dataset...")
    cfg = load_config(str(BASE_DIR / "config.yaml"))
    ann_df   = parse_annotations(str(DATA_ROOT))
    samples_df = parse_samples(str(DATA_ROOT))

    print("Mining safety-critical scenes...")
    results = []

    lv = mine_low_visibility(ann_df, samples_df)
    print(f"  Low visibility scenes:   {len(lv)}")
    results.append(lv)

    ds = mine_dense_scenes(ann_df)
    print(f"  Dense urban scenes:      {len(ds)}")
    results.append(ds)

    sm = mine_small_objects(ann_df)
    print(f"  Small object scenes:     {len(sm)}")
    results.append(sm)

    pr = mine_pedestrian_risk(ann_df)
    print(f"  Pedestrian risk scenes:  {len(pr)}")
    results.append(pr)

    # Combine — a scene can appear under multiple categories
    mined_df = pd.concat(
        [r for r in results if not r.empty],
        ignore_index=True
    )

    print("\nGenerating plots...")
    plot_mined_scenes_summary(mined_df)
    plot_object_size_distribution(ann_df)

    generate_mining_report(mined_df, ann_df)

    # Save mined scene list for Week 4 safety analysis
    mined_df.to_csv(OUTPUT_DIR / "mined_scenes.csv", index=False)
    print(f"\nMined scene index saved: {OUTPUT_DIR / 'mined_scenes.csv'}")

    return mined_df


if __name__ == "__main__":
    run_mining()