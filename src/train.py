"""
train.py
YOLOv8 transfer learning pipeline for nuScenes ADAS object detection.

Strategy: freeze backbone (pretrained COCO weights), fine-tune
detection head only. CPU-optimised: 320px images, 15 epochs.
Rationale: mirrors production practice where backbone is pretrained
on large datasets and adapted to domain-specific data.
"""

import os
import time
import yaml
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import torch


# ── Config ────────────────────────────────────────────────────────────────────
DATASET_YAML  = Path("../datasets/nuscenes_yolo/dataset.yaml")
OUTPUT_DIR    = Path("../outputs/week2")
RUNS_DIR      = Path("../runs")

# CPU-optimised settings
IMG_SIZE      = 320      # 640 takes ~4x longer per epoch on CPU
EPOCHS        = 15       # Enough for convergence with frozen backbone
BATCH_SIZE    = 4        # Safe for 8GB RAM on Windows
WORKERS       = 0        # Must be 0 on Windows (multiprocessing limitation)
MODEL_NAME    = "yolov8n.pt"   # nano — fastest, still meaningful results

# Freeze backbone layers (0–9 = backbone in YOLOv8n)
FREEZE_LAYERS = 9


def setup_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"nuscenes_{timestamp}"
    return run_dir


def log_training_config(run_dir: Path):
    config = {
        "model": MODEL_NAME,
        "dataset": str(DATASET_YAML),
        "img_size": IMG_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "freeze_layers": FREEZE_LAYERS,
        "device": "cpu",
        "strategy": "transfer_learning_frozen_backbone",
        "rationale": (
            "Backbone pretrained on COCO (80 classes, 118k images). "
            "Fine-tuning head only on nuScenes domain data. "
            "Mirrors production practice for domain adaptation."
        ),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Training config saved.")
    return config


def train():
    print("=" * 55)
    print("ADAS AI Training Pipeline — Week 2")
    print("YOLOv8n Transfer Learning on nuScenes")
    print("=" * 55)

    run_dir = setup_output_dir()
    config  = log_training_config(run_dir)

    print(f"\nDevice:   CPU")
    print(f"Model:    {MODEL_NAME} (pretrained COCO)")
    print(f"Images:   {IMG_SIZE}px  |  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}")
    print(f"Strategy: Freeze backbone layers 0-{FREEZE_LAYERS}, train head only")
    print(f"Output:   {run_dir}\n")

    # Load pretrained YOLOv8 nano
    model = YOLO(MODEL_NAME)

    # Freeze backbone
    freeze_count = 0
    for i, (name, param) in enumerate(model.model.named_parameters()):
        layer_num = int(name.split(".")[0]) if name.split(".")[0].isdigit() else 999
        if layer_num < FREEZE_LAYERS:
            param.requires_grad = False
            freeze_count += 1
    print(f"Frozen {freeze_count} backbone parameters.")

    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({trainable/total*100:.1f}%)\n")

    # Estimate time
    print("Estimated training time on CPU: 60–120 minutes")
    print("Tip: leave running — check outputs/week2/ for results after.\n")

    start_time = time.time()

    # Train
    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device="cpu",
        project=str(RUNS_DIR),
        name="nuscenes_yolov8n",
        exist_ok=True,
        verbose=True,
        # Augmentation — conservative for small dataset
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,    # Mosaic helps with small datasets
        mixup=0.0,
        # Reduce logging overhead on CPU
        plots=True,
        save=True,
        save_period=5,
    )

    elapsed = (time.time() - start_time) / 60
    print(f"\nTraining complete in {elapsed:.1f} minutes")

    # Save final metrics summary
    save_metrics_summary(results, run_dir, elapsed)

    return results, run_dir


def save_metrics_summary(results, run_dir: Path, elapsed_min: float):
    """Extract and save key metrics to a clean JSON for Week 2 reporting."""
    try:
        metrics = results.results_dict
        summary = {
            "training_time_minutes": round(elapsed_min, 1),
            "final_metrics": {
                "mAP50":      round(metrics.get("metrics/mAP50(B)",    0), 4),
                "mAP50_95":   round(metrics.get("metrics/mAP50-95(B)", 0), 4),
                "precision":  round(metrics.get("metrics/precision(B)",0), 4),
                "recall":     round(metrics.get("metrics/recall(B)",   0), 4),
            },
            "model":   "yolov8n — transfer learning, frozen backbone",
            "dataset": "nuScenes mini",
            "epochs":  EPOCHS,
            "img_size": IMG_SIZE,
        }

        with open(run_dir / "metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n=== Final Metrics ===")
        for k, v in summary["final_metrics"].items():
            print(f"  {k}: {v}")
        print(f"  Training time: {elapsed_min:.1f} min")
        print(f"\nMetrics saved to: {run_dir / 'metrics_summary.json'}")

    except Exception as e:
        print(f"Could not extract metrics: {e}")


if __name__ == "__main__":
    train()