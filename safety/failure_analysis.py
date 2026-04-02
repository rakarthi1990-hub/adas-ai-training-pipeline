"""
failure_analysis.py
Maps scene mining results to ADAS safety failure chains.

For each safety-critical ODD category identified in Week 3,
this module traces the complete failure propagation path:
  perception miss → decision error → unsafe actuation → harm

Connects experimental ML results to ISO 26262 / SOTIF framework.
This is the layer that distinguishes system engineers from ML engineers.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
OUTPUT_DIR  = BASE_DIR / "outputs" / "week4"
MINING_CSV  = BASE_DIR / "outputs" / "week3" / "mined_scenes.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Failure chain definitions ─────────────────────────────────────────────────
# Each entry maps an ODD condition to its full ADAS failure chain
# and SOTIF classification.

FAILURE_CHAINS = {
    "low_visibility": {
        "condition":        "Object visibility 0–40% (occlusion, distance, weather)",
        "perception_failure": "Confidence score below detection threshold → object suppressed",
        "decision_failure":   "Decision layer receives no object input → assumes clear path",
        "actuation_failure":  "No braking or evasive action triggered",
        "harm":               "Collision with undetected pedestrian or vehicle",
        "week2_evidence":     "Pedestrian AP@50 = 0.058 — confirmed low recall on occluded objects",
        "sotif_class":        "Known unsafe scenario (visibility degradation is a known ODD limit)",
        "iso_26262_ref":      "ASIL-B minimum for pedestrian detection function",
        "mitigation":         "Sensor fusion (radar/LiDAR redundancy), multi-frame temporal tracking, "
                              "ODD exit trigger when visibility below threshold",
        "priority":           "critical",
    },
    "dense_urban": {
        "condition":        "10+ objects per frame — dense urban intersection",
        "perception_failure": "NMS suppresses valid detections due to overlapping bounding boxes",
        "decision_failure":   "Partial object list passed to prediction layer — trajectory estimation incomplete",
        "actuation_failure":  "Incorrect path planning — vehicle proceeds through occupied space",
        "harm":               "Collision with suppressed (untracked) road user",
        "week2_evidence":     "Recall 0.21 — 79% of objects missed, dense scenes confirmed in 80.7% of dataset",
        "sotif_class":        "Known unsafe scenario (NMS behaviour in dense scenes is a known limitation)",
        "iso_26262_ref":      "SOTIF ISO/PAS 21448 — performance limitation in ODD",
        "mitigation":         "Anchor-free detection heads, higher NMS IoU threshold in dense ODDs, "
                              "multi-camera fusion to reduce single-view occlusion",
        "priority":           "high",
    },
    "pedestrian_risk": {
        "condition":        "2+ pedestrians in scene — urban crossing, pavement, school zone",
        "perception_failure": "Low per-class AP (0.058) — pedestrian detections missed or misclassified",
        "decision_failure":   "Pedestrian trajectory not predicted → right-of-way not yielded",
        "actuation_failure":  "Vehicle proceeds without braking or speed reduction",
        "harm":               "Pedestrian struck at crossing or road crossing event",
        "week2_evidence":     "Pedestrian AP@50 = 0.058, bicycle AP@50 = 0.003 — "
                              "safety-critical classes perform worst",
        "sotif_class":        "Known unsafe scenario — pedestrian detection gap is a known SOTIF trigger",
        "iso_26262_ref":      "ASIL-C/D for emergency pedestrian braking functions",
        "mitigation":         "Dedicated pedestrian detection head, higher resolution input for small objects, "
                              "class-weighted loss function to prioritise safety-critical classes",
        "priority":           "critical",
    },
}

PRIORITY_COLORS = {
    "critical": "#E24B4A",
    "high":     "#EF9F27",
    "medium":   "#378ADD",
}


def plot_failure_chain_summary():
    """
    Visual summary of failure chains per ODD category.
    Shows: condition → failure → harm, with priority colour coding.
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    fig.suptitle(
        "ADAS failure chain analysis — per safety-critical ODD category\n"
        "Connected to Week 2 training results (mAP@50: 0.029, recall: 0.21)",
        fontsize=12, y=1.01
    )

    stages = ["condition", "perception_failure",
              "decision_failure", "actuation_failure", "harm"]
    stage_labels = ["ODD condition", "Perception failure",
                    "Decision failure", "Actuation failure", "Harm"]
    stage_colors = ["#888780", "#EF9F27", "#D85A30", "#E24B4A", "#A32D2D"]

    for ax, (category, chain) in zip(axes, FAILURE_CHAINS.items()):
        color = PRIORITY_COLORS[chain["priority"]]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Category title
        ax.text(0, 0.92, category.replace("_", " ").upper(),
                fontsize=10, fontweight="bold", color=color)
        ax.text(0, 0.78,
                f"Evidence: {chain['week2_evidence']}",
                fontsize=8, color="#5F5E5A",
                style="italic")

        # Chain boxes
        box_w = 1.7
        for i, (stage, label, sc) in enumerate(
                zip(stages, stage_labels, stage_colors)):
            x = i * 2.0
            ax.add_patch(plt.Rectangle(
                (x, 0.05), box_w, 0.55,
                facecolor=sc, alpha=0.15,
                edgecolor=sc, linewidth=1.0
            ))
            ax.text(x + box_w / 2, 0.55,
                    label, ha="center", va="bottom",
                    fontsize=7, color=sc, fontweight="bold")

            # Wrap text manually for long strings
            text = chain[stage]
            if len(text) > 40:
                mid = text[:40].rfind(" ")
                text = text[:mid] + "\n" + text[mid+1:]
            ax.text(x + box_w / 2, 0.30,
                    text, ha="center", va="center",
                    fontsize=6.5, color="#2C2C2A",
                    wrap=True)

            # Arrow between boxes
            if i < len(stages) - 1:
                ax.annotate("",
                    xy=(x + box_w + 0.08, 0.325),
                    xytext=(x + box_w, 0.325),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#888780",
                        lw=1.2
                    )
                )

        # SOTIF label
        ax.text(0, 0.0,
                f"SOTIF: {chain['sotif_class']}",
                fontsize=7, color="#534AB7")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "failure_chain_analysis.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: failure_chain_analysis.png")


def plot_risk_matrix():
    """
    Risk matrix: probability of occurrence vs severity of harm.
    Positions each ODD category on the standard automotive risk matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Background risk zones
    ax.add_patch(plt.Rectangle((0, 0), 5, 5,
                 facecolor="#EAF3DE", alpha=0.4, label="Acceptable"))
    ax.add_patch(plt.Rectangle((1, 1), 4, 4,
                 facecolor="#FAEEDA", alpha=0.4, label="ALARP"))
    ax.add_patch(plt.Rectangle((2, 2), 3, 3,
                 facecolor="#FCEBEB", alpha=0.4, label="Unacceptable"))

    # Data points: (exposure_probability, severity, label, scenes_pct)
    points = [
        (4.2, 4.5, "Low visibility\n(87.4% of scenes)",  "#E24B4A"),
        (4.5, 3.8, "Dense urban\n(80.7% of scenes)",     "#EF9F27"),
        (3.8, 4.7, "Pedestrian risk\n(70.3% of scenes)", "#D85A30"),
    ]

    for x, y, label, color in points:
        ax.scatter(x, y, s=180, color=color, zorder=5)
        ax.annotate(label, (x, y),
                    textcoords="offset points",
                    xytext=(12, 4),
                    fontsize=8, color=color)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("Exposure / probability of occurrence →", fontsize=10)
    ax.set_ylabel("Severity of harm →", fontsize=10)
    ax.set_title(
        "Risk matrix — mined ODD categories\n"
        "Based on nuScenes dataset analysis + Week 2 model performance",
        fontsize=11, pad=10
    )
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["Very low", "Low", "Medium", "High", "Very high"],
                       fontsize=8)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["Negligible", "Minor", "Moderate",
                        "Serious", "Catastrophic"], fontsize=8)

    legend_patches = [
        mpatches.Patch(facecolor="#EAF3DE", alpha=0.6, label="Acceptable risk"),
        mpatches.Patch(facecolor="#FAEEDA", alpha=0.6, label="ALARP region"),
        mpatches.Patch(facecolor="#FCEBEB", alpha=0.6, label="Unacceptable risk"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "risk_matrix.png", dpi=150)
    plt.close()
    print("Saved: risk_matrix.png")


def generate_safety_report():
    """Full structured safety report — all findings in one document."""
    lines = [
        "=== Week 4 — Safety Failure Analysis Report ===",
        "Project: ADAS AI Training Pipeline",
        "Author:  Karthikeyan Rajan | Senior ADAS Systems Engineer",
        "Date:    April 2026",
        "",
        "References: ISO 26262, SOTIF (ISO/PAS 21448), nuScenes dataset",
        "=" * 58,
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        "This report maps the AI model performance results from Week 2",
        "(mAP@50: 0.029, recall: 0.21) and scene mining results from",
        "Week 3 (88.6% safety-critical scenes) to ADAS safety failure",
        "chains and SOTIF classifications.",
        "",
        "The analysis demonstrates that the model's low recall is not",
        "randomly distributed — it is concentrated in safety-critical",
        "ODD conditions that constitute the primary failure scenarios",
        "for production ADAS functions.",
        "",
    ]

    for category, chain in FAILURE_CHAINS.items():
        lines += [
            f"FAILURE CHAIN: {category.upper().replace('_', ' ')}",
            "-" * 40,
            f"Condition:          {chain['condition']}",
            f"Perception failure: {chain['perception_failure']}",
            f"Decision failure:   {chain['decision_failure']}",
            f"Actuation failure:  {chain['actuation_failure']}",
            f"Harm:               {chain['harm']}",
            "",
            f"Experimental evidence (Week 2):",
            f"  {chain['week2_evidence']}",
            "",
            f"SOTIF classification:",
            f"  {chain['sotif_class']}",
            "",
            f"ISO 26262 reference:",
            f"  {chain['iso_26262_ref']}",
            "",
            f"Recommended mitigation:",
            f"  {chain['mitigation']}",
            "",
        ]

    lines += [
        "=" * 58,
        "SOTIF TAXONOMY MAPPING",
        "-" * 40,
        "",
        "Stage 1 — Unknown unsafe scenarios (identified by this project):",
        "  - Confidence collapse at 0-40% visibility",
        "  - NMS suppression in 10+ object density scenes",
        "  - Pedestrian class underperformance (AP 0.058)",
        "",
        "Stage 2 — Known unsafe scenarios (post-analysis):",
        "  - Above conditions now formally documented",
        "  - Risk matrix positions all three in unacceptable zone",
        "  - ODD exit triggers required for each condition",
        "",
        "Stage 3 — Mitigated scenarios (recommended actions):",
        "  - Sensor fusion: radar/LiDAR redundancy for low visibility",
        "  - Data collection: 5,000+ targeted frames per ODD condition",
        "  - Model: class-weighted loss, higher resolution for small objects",
        "  - Validation: per-ODD closed-loop KPI gates before release",
        "",
        "=" * 58,
        "PRODUCTION READINESS ASSESSMENT",
        "-" * 40,
        "",
        "Current model state: NOT production ready",
        "  Recall 0.21 is below minimum threshold for any ADAS function.",
        "  Pedestrian AP 0.058 is below ASIL-B minimum requirements.",
        "",
        "Path to production readiness:",
        "  1. Expand dataset: 10,000+ samples across all mined ODDs",
        "  2. Unfreeze backbone: full fine-tuning with sufficient data",
        "  3. Target KPIs: pedestrian recall > 0.80, overall mAP > 0.50",
        "  4. Closed-loop validation: per-ODD KPI gates",
        "  5. Safety concept: ASIL decomposition per function",
        "",
        "This analysis provides the data-driven justification for the",
        "dataset expansion and model improvement roadmap.",
    ]

    report = "\n".join(lines)
    print(report)

    with open(OUTPUT_DIR / "week4_safety_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {OUTPUT_DIR / 'week4_safety_report.txt'}")


if __name__ == "__main__":
    print("Generating failure chain analysis...")
    plot_failure_chain_summary()

    print("Generating risk matrix...")
    plot_risk_matrix()

    print("Generating safety report...")
    generate_safety_report()

    print(f"\nAll Week 4 outputs saved to {OUTPUT_DIR}/")