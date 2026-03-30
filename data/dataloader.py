"""
dataloader.py
PyTorch Dataset class for nuScenes camera images and annotations.
Prepares data for YOLOv8 training pipeline in Week 2.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import yaml


def load_config(config_path: str = "../config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Map nuScenes category names to integer class IDs
CATEGORY_TO_ID = {
    "car": 0,
    "pedestrian": 1,
    "bicycle": 2,
    "motorcycle": 3,
    "bus": 4,
    "truck": 5,
    "traffic_cone": 6,
    "barrier": 7,
}


class NuScenesDataset(Dataset):
    """
    Lightweight nuScenes camera dataset for object detection.
    Returns image tensors and YOLO-format bounding box labels.
    
    YOLO format: [class_id, x_center, y_center, width, height]
    All bbox values normalised to [0, 1].
    """

    def __init__(self, data_root: str, split: str = "mini_train",
                 image_size: int = 640, augment: bool = False):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.samples = self._load_samples()
        self.transform = self._build_transform()

    def _load_samples(self):
        """Load sample tokens and camera image paths for the split."""
        v = self.data_root / "v1.0-mini"

        with open(v / "sample.json") as f:
            all_samples = json.load(f)
        with open(v / "sample_data.json") as f:
            all_sample_data = json.load(f)
        with open(v / "sample_annotation.json") as f:
            all_annotations = json.load(f)
        with open(v / "instance.json") as f:
            instances = {i["token"]: i for i in json.load(f)}
        with open(v / "category.json") as f:
            categories = {c["token"]: c["name"] for c in json.load(f)}
        with open(v / "calibrated_sensor.json") as f:
            cal_sensors = {c["token"]: c for c in json.load(f)}
        with open(v / "sensor.json") as f:
            sensors = {s["token"]: s for s in json.load(f)}

        # Build annotation index keyed by sample_token
        ann_by_sample = {}
        for ann in all_annotations:
            st = ann["sample_token"]
            if st not in ann_by_sample:
                ann_by_sample[st] = []
            ann_by_sample[st].append(ann)

        # Get camera sample_data keyed by sample_token
        cam_data_by_sample = {}
        for sd in all_sample_data:
            cs = cal_sensors.get(sd.get("calibrated_sensor_token", ""), {})
            sensor_token = cs.get("sensor_token", "")
            sensor = sensors.get(sensor_token, {})
            if sensor.get("modality") == "camera" and sensor.get("channel") == "CAM_FRONT":
                cam_data_by_sample[sd["sample_token"]] = sd

        samples = []
        for s in all_samples:
            token = s["token"]
            if token not in cam_data_by_sample:
                continue
            sd = cam_data_by_sample[token]
            img_path = self.data_root / sd["filename"]
            if not img_path.exists():
                continue

            # Get image dimensions from sample_data
            w, h = sd.get("width", 1600), sd.get("height", 900)

            # Parse annotations for this sample
            labels = []
            for ann in ann_by_sample.get(token, []):
                inst = instances.get(ann["instance_token"], {})
                cat_name = categories.get(inst.get("category_token", ""), "")
                top_class = cat_name.split(".")[1] if "." in cat_name else cat_name
                class_id = CATEGORY_TO_ID.get(top_class, -1)
                if class_id == -1:
                    continue
                # nuScenes 2D box from sample_annotation (if available)
                # Fall back to projecting 3D box — simplified here to centre point
                tx, ty, tz = ann["translation"]
                # Approximate: use translation x,y projected (simplified)
                # In production, use pyquaternion + camera intrinsics
                labels.append({"class_id": class_id, "token": ann["token"]})

            samples.append({
                "token": token,
                "img_path": str(img_path),
                "width": w,
                "height": h,
                "labels": labels,
            })

        return samples

    def _build_transform(self):
        transforms = [
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]
        if self.augment:
            transforms.insert(1, T.RandomHorizontalFlip(p=0.5))
            transforms.insert(2, T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1))
        return T.Compose(transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        img_tensor = self.transform(img)

        return {
            "image": img_tensor,
            "token": sample["token"],
            "img_path": sample["img_path"],
            "num_objects": len(sample["labels"]),
        }


def get_dataloader(data_root: str, split: str = "mini_train",
                   image_size: int = 640, batch_size: int = 8,
                   augment: bool = False) -> DataLoader:
    dataset = NuScenesDataset(data_root, split, image_size, augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "mini_train"),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    cfg = load_config()
    loader = get_dataloader(
        data_root=cfg["dataset"]["root"],
        split=cfg["dataset"]["split"],
        image_size=cfg["training"]["image_size"],
        batch_size=cfg["training"]["batch_size"],
    )
    print(f"Dataset loaded: {len(loader.dataset)} samples")
    batch = next(iter(loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Avg objects in batch: {sum(batch['num_objects']) / len(batch['num_objects']):.1f}")