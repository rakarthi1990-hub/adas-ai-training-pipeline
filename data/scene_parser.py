"""
scene_parser.py
Parses nuScenes mini dataset: extracts scenes, samples,
camera tokens, and annotation metadata into structured format.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_nuscenes_table(data_root: str, table_name: str) -> List[Dict]:
    """Load a nuScenes JSON table by name."""
    path = Path(data_root) / "v1.0-mini" / f"{table_name}.json"
    with open(path, "r") as f:
        return json.load(f)


def parse_scenes(data_root: str) -> pd.DataFrame:
    """Extract all scenes with description and sample count."""
    scenes = load_nuscenes_table(data_root, "scene")
    samples = load_nuscenes_table(data_root, "sample")

    # Build sample count per scene
    sample_map = {}
    for s in samples:
        scene_token = s["scene_token"]
        sample_map[scene_token] = sample_map.get(scene_token, 0) + 1

    records = []
    for scene in scenes:
        records.append({
            "token": scene["token"],
            "name": scene["name"],
            "description": scene["description"],
            "sample_count": sample_map.get(scene["token"], 0),
            "log_token": scene["log_token"],
        })

    return pd.DataFrame(records)


def parse_annotations(data_root: str) -> pd.DataFrame:
    """Extract all object annotations with class, size, visibility."""
    annotations = load_nuscenes_table(data_root, "sample_annotation")
    instances = load_nuscenes_table(data_root, "instance")
    categories = load_nuscenes_table(data_root, "category")

    # Build lookup maps
    instance_map = {i["token"]: i for i in instances}
    category_map = {c["token"]: c["name"] for c in categories}

    records = []
    for ann in annotations:
        instance = instance_map.get(ann["instance_token"], {})
        cat_token = instance.get("category_token", "")
        category_name = category_map.get(cat_token, "unknown")

        # Simplify category to top-level class
        top_class = category_name.split(".")[1] if "." in category_name else category_name

        records.append({
            "token": ann["token"],
            "sample_token": ann["sample_token"],
            "category": top_class,
            "category_full": category_name,
            "visibility_token": ann.get("visibility_token", ""),
            "num_lidar_pts": ann.get("num_lidar_pts", 0),
            "num_radar_pts": ann.get("num_radar_pts", 0),
            "size_w": ann["size"][0],
            "size_l": ann["size"][1],
            "size_h": ann["size"][2],
        })

    return pd.DataFrame(records)


def parse_samples(data_root: str) -> pd.DataFrame:
    """Extract all samples with scene linkage and timestamp."""
    samples = load_nuscenes_table(data_root, "sample")

    records = []
    for s in samples:
        records.append({
            "token": s["token"],
            "scene_token": s["scene_token"],
            "timestamp": s["timestamp"],
        })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us")
    return df


def get_class_distribution(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Count annotations per class, sorted descending."""
    dist = (
        annotations_df["category"]
        .value_counts()
        .reset_index()
    )
    dist.columns = ["class", "count"]
    dist["percentage"] = (dist["count"] / dist["count"].sum() * 100).round(2)
    return dist


if __name__ == "__main__":
    cfg = load_config()
    data_root = cfg["dataset"]["root"]

    print("Parsing scenes...")
    scenes_df = parse_scenes(data_root)
    print(f"  Found {len(scenes_df)} scenes")

    print("Parsing samples...")
    samples_df = parse_samples(data_root)
    print(f"  Found {len(samples_df)} samples")

    print("Parsing annotations...")
    ann_df = parse_annotations(data_root)
    print(f"  Found {len(ann_df)} annotations")

    print("\nClass distribution:")
    print(get_class_distribution(ann_df).to_string(index=False))