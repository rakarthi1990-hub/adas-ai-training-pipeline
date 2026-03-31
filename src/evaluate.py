"""
evaluate.py
Evaluates trained YOLOv8 model on nuScenes val split.
Produces per-class metrics table and plots.
Connects model performance to ADAS safety relevance.
"""

import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from ultralytics import YOLO


OUTPUT_DIR   = Path("../outputs/week2")
DATASET_YAML = Path("../datasets/nuscenes_yolo/dataset.yaml")

CLASS_NAMES = [
    "car", "pedestrian", "bicycle",
    "motorcycle", "bus", "truck",
    "traffic_cone", "barrier"
]

# Safety priority — pedestrians and cyclists are highest risk
SAFETY_PRIORITY = {
    "pedestrian": "critical",
    "bicycle":    "critical",
    "motorcycle": "high",
    "car":        "high",
    "truck":      "medium",
    "bus":        "medium",
    "traffic_cone": "low",
    "barrier":    "low",
}

PRIORITY_COLOR = {
    "critical": "#E24B4A",
    "high":     "#EF9F27",
    "medium":   "#378ADD",
    "low":      "#888780",
}


def evaluate(weights_path: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {weights_path}")
    model = YOLO(weights_path)

    print("Running validation on nuScenes val split...")
    metrics = model.val(
        data=str(DATASET_YAML),
        imgsz=320,
        batch=4,
        workers=0,
        verbose=True,
    )

    # Per-class AP from results
    per_class_ap = {}
    try:
        ap_per_class = metrics.box.ap_class_index
        aps          = metrics.box.ap50
        for idx, ap in zip(ap_per_class, aps):
            if idx < len(CLASS_NAMES):
                per_class_ap[CLASS_NAMES[idx]] = round(float(ap), 4)
    except Exception as e:
        print(f"Note: per-class AP extraction: {e}")
        per_class_ap = {c: 0.0 for c in CLASS_NAMES}

    plot_per_class_ap(per_class_ap)
    save_evaluation_report(metrics, per_class_ap, weights_path)

    return per_class_ap


def plot_per_class_ap(per_class_ap: dict):
    """
    Bar chart of AP@50 per class, coloured by safety priority.
    Red = critical safety class (must have highest AP).
    """
    classes = list(per_class_ap.keys())
    aps     = list(per_class_ap.values())
    colors  = [PRIORITY_COLOR[SAFETY_PRIORITY.get(c, "low")] for c in classes]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(classes, aps, color=colors, width=0.6, zorder=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)

    # Annotate
    for bar, ap in zip(bars, aps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{ap:.2f}",
            ha="center", va="bottom", fontsize=9
        )

    # BMW KPI target line
    ax.axhline(y=0.40, color="#534AB7", linewidth=1.2,
               linestyle="--", label="Target mAP@50: 0.40")

    # Legend for safety priority
    patches = [mpatches.Patch(color=c, label=p.capitalize())
               for p, c in PRIORITY_COLOR.items()]
    patches.append(
        plt.Line2D([0], [0], color="#534AB7", linestyle="--",
                   label="Target mAP@50")
    )
    ax.legend(handles=patches, fontsize=9, loc="upper right")

    ax.set_title(
        "Per-class AP@50 — coloured by ADAS safety priority",
        fontsize=12, pad=10
    )
    ax.set_ylabel("AP@50")
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_class_ap.png", dpi=150)
    plt.close()
    print("Saved: per_class_ap.png")


def save_evaluation_report(metrics, per_class_ap: dict, weights_path: str):
    """Structured evaluation report linking metrics to safety relevance."""
    try:
        overall = {
            "mAP50":     round(float(metrics.box.map50),  4),
            "mAP50_95":  round(float(metrics.box.map),    4),
            "precision": round(float(metrics.box.mp),     4),
            "recall":    round(float(metrics.box.mr),     4),
        }
    except Exception:
        overall = {"mAP50": 0, "mAP50_95": 0, "precision": 0, "recall": 0}

    # Safety analysis — critical classes
    critical_classes = [c for c, p in SAFETY_PRIORITY.items() if p == "critical"]
    critical_aps = {c: per_class_ap.get(c, 0) for c in critical_classes}

    report = {
        "model_weights": weights_path,
        "dataset": "nuScenes mini — val split",
        "overall_metrics": overall,
        "per_class_ap50": per_class_ap,
        "safety_analysis": {
            "critical_class_performance": critical_aps,
            "safety_note": (
                "Pedestrian and bicycle AP are the primary safety KPIs. "
                "Low recall on these classes maps directly to missed detection "
                "risk in the ADAS decision pipeline (ISO 26262 / SOTIF)."
            ),
        },
    }

    with open(OUTPUT_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Evaluation Report ===")
    print(f"  mAP@50:    {overall['mAP50']}")
    print(f"  Precision: {overall['precision']}")
    print(f"  Recall:    {overall['recall']}")
    print("\n  Critical class AP@50:")
    for c, ap in critical_aps.items():
        flag = " ← SAFETY RISK" if ap < 0.30 else ""
        print(f"    {c}: {ap}{flag}")
    print(f"\nReport saved to: {OUTPUT_DIR / 'evaluation_report.json'}")


if __name__ == "__main__":
    import sys
    # Usage: python evaluate.py path/to/best.pt
    # Default: look for latest run
    if len(sys.argv) > 1:
        weights = sys.argv[1]
    else:
        runs = sorted(Path("../runs").glob("nuscenes_yolov8n/weights/best.pt"))
        weights = str(runs[-1]) if runs else "yolov8n.pt"
        print(f"Auto-detected weights: {weights}")

    evaluate(weights)


### Step 5 — Add `pyquaternion` to requirements

# pyquaternion>=0.9.9
# pip install pyquaternion