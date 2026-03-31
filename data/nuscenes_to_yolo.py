"""
nuscenes_to_yolo.py  — v2
Correct 3-step projection: world → ego → camera → image.
This is the standard nuScenes projection pipeline.
"""

import json, os, shutil, random
from pathlib import Path
import numpy as np
import yaml
from pyquaternion import Quaternion
from tqdm import tqdm

DATA_ROOT      = Path("../data/nuscenes")
OUTPUT_ROOT    = Path("../datasets/nuscenes_yolo")
VAL_RATIO      = 0.2
RANDOM_SEED    = 42
CAMERA_CHANNEL = "CAM_FRONT"

CATEGORY_MAP = {
    "vehicle.car":                  0,
    "human.pedestrian.adult":       1,
    "human.pedestrian.child":       1,
    "human.pedestrian.wheelchair":  1,
    "vehicle.bicycle":              2,
    "vehicle.motorcycle":           3,
    "vehicle.bus.bendy":            4,
    "vehicle.bus.rigid":            4,
    "vehicle.truck":                5,
    "movable_object.trafficcone":   6,
    "movable_object.barrier":       7,
}

CLASS_NAMES = [
    "car","pedestrian","bicycle","motorcycle",
    "bus","truck","traffic_cone","barrier"
]


def load_table(v: Path, name: str):
    with open(v / f"{name}.json") as f:
        return json.load(f)


def world_to_image(world_xyz, ego_pose, cam_cs, img_w, img_h):
    """
    Full nuScenes projection pipeline:
      world → ego vehicle frame → camera frame → image pixels

    Returns (cx_norm, cy_norm, w_norm, h_norm) or None.
    """
    # Step 1: world → ego frame
    eq = Quaternion(ego_pose["rotation"])
    et = np.array(ego_pose["translation"])
    p_ego = eq.inverse.rotate(world_xyz - et)

    # Step 2: ego → camera frame
    cq = Quaternion(cam_cs["rotation"])
    ct = np.array(cam_cs["translation"])
    p_cam = cq.inverse.rotate(p_ego - ct)

    # Behind camera — discard
    if p_cam[2] <= 0.5:
        return None

    # Step 3: camera → image via intrinsic matrix
    K  = np.array(cam_cs["camera_intrinsic"])
    uv = K @ p_cam
    px, py = uv[0] / uv[2], uv[1] / uv[2]

    # Outside image bounds — discard
    if not (0 < px < img_w and 0 < py < img_h):
        return None

    return px, py, p_cam[2]   # x_px, y_px, depth_m


def project_box(ann, ego_pose, cam_cs, img_w, img_h):
    """
    Project 3D annotation box corners into 2D image,
    return YOLO-format [cx, cy, w, h] normalised, or None.
    """
    centre = np.array(ann["translation"])
    result = world_to_image(centre, ego_pose, cam_cs, img_w, img_h)
    if result is None:
        return None

    cx_px, cy_px, depth = result

    # Project 8 box corners for accurate 2D bounding box
    w3, l3, h3 = ann["size"]
    corners_local = np.array([
        [ l3/2,  w3/2,  h3/2],
        [ l3/2, -w3/2,  h3/2],
        [-l3/2,  w3/2,  h3/2],
        [-l3/2, -w3/2,  h3/2],
        [ l3/2,  w3/2, -h3/2],
        [ l3/2, -w3/2, -h3/2],
        [-l3/2,  w3/2, -h3/2],
        [-l3/2, -w3/2, -h3/2],
    ])

    # Rotate corners by annotation quaternion (object heading)
    ann_q = Quaternion(ann["rotation"])
    corners_world = np.array([
        centre + ann_q.rotate(c) for c in corners_local
    ])

    # Project all corners
    K = np.array(cam_cs["camera_intrinsic"])
    ego_q = Quaternion(ego_pose["rotation"])
    ego_t = np.array(ego_pose["translation"])
    cam_q = Quaternion(cam_cs["rotation"])
    cam_t = np.array(cam_cs["translation"])

    px_list, py_list = [], []
    for cw in corners_world:
        p_ego = ego_q.inverse.rotate(cw - ego_t)
        p_cam = cam_q.inverse.rotate(p_ego - cam_t)
        if p_cam[2] <= 0.1:
            continue
        uv = K @ p_cam
        px_list.append(uv[0] / uv[2])
        py_list.append(uv[1] / uv[2])

    if len(px_list) < 2:
        return None

    # 2D bounding box from projected corners — clamped to image
    x1 = max(0, min(px_list))
    x2 = min(img_w, max(px_list))
    y1 = max(0, min(py_list))
    y2 = min(img_h, max(py_list))

    box_w = x2 - x1
    box_h = y2 - y1

    # Discard tiny boxes (distant objects, < 8px)
    if box_w < 8 or box_h < 8:
        return None

    # YOLO normalised format
    cx_n = (x1 + box_w / 2) / img_w
    cy_n = (y1 + box_h / 2) / img_h
    w_n  = box_w / img_w
    h_n  = box_h / img_h

    return cx_n, cy_n, w_n, h_n


def convert():
    random.seed(RANDOM_SEED)
    v = DATA_ROOT / "v1.0-mini"

    print("Loading nuScenes tables...")
    samples     = load_table(v, "sample")
    sample_data = load_table(v, "sample_data")
    annotations = load_table(v, "sample_annotation")
    instances   = {i["token"]: i  for i in load_table(v, "instance")}
    categories  = {c["token"]: c["name"] for c in load_table(v, "category")}
    cal_sensors = {c["token"]: c  for c in load_table(v, "calibrated_sensor")}
    sensors     = {s["token"]: s  for s in load_table(v, "sensor")}
    ego_poses   = {e["token"]: e  for e in load_table(v, "ego_pose")}

    # Index CAM_FRONT sample_data by sample_token
    cam_by_sample = {}
    for sd in sample_data:
        cs = cal_sensors.get(sd.get("calibrated_sensor_token", ""), {})
        sensor = sensors.get(cs.get("sensor_token", ""), {})
        if (sensor.get("modality") == "camera"
                and sensor.get("channel") == CAMERA_CHANNEL):
            cam_by_sample[sd["sample_token"]] = (sd, cs)

    # Index annotations by sample_token
    ann_by_sample = {}
    for ann in annotations:
        ann_by_sample.setdefault(ann["sample_token"], []).append(ann)

    # Create output dirs
    for split in ["train", "val"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Train/val split
    valid_tokens = [s["token"] for s in samples if s["token"] in cam_by_sample]
    random.shuffle(valid_tokens)
    n_val       = max(1, int(len(valid_tokens) * VAL_RATIO))
    val_set     = set(valid_tokens[:n_val])

    stats = {"converted": 0, "labels_written": 0,
             "skipped_projection": 0, "skipped_class": 0,
             "empty_label_files": 0}

    for token in tqdm(valid_tokens, desc="Converting"):
        split    = "val" if token in val_set else "train"
        sd, cs   = cam_by_sample[token]

        img_src  = DATA_ROOT / sd["filename"]
        if not img_src.exists():
            continue

        img_w  = sd.get("width",  1600)
        img_h  = sd.get("height",  900)

        # Get ego pose at this sample_data timestamp
        ego_pose = ego_poses.get(sd.get("ego_pose_token", ""), {})
        if not ego_pose:
            continue

        # Copy image
        img_dst = OUTPUT_ROOT / "images" / split / f"{token}.jpg"
        shutil.copy2(img_src, img_dst)

        # Build YOLO labels
        lines = []
        for ann in ann_by_sample.get(token, []):
            inst     = instances.get(ann["instance_token"], {})
            cat_name = categories.get(inst.get("category_token", ""), "")
            class_id = CATEGORY_MAP.get(cat_name, -1)
            if class_id == -1:
                stats["skipped_class"] += 1
                continue

            box = project_box(ann, ego_pose, cs, img_w, img_h)
            if box is None:
                stats["skipped_projection"] += 1
                continue

            cx, cy, w, h = box
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats["labels_written"] += 1

        label_dst = OUTPUT_ROOT / "labels" / split / f"{token}.txt"
        with open(label_dst, "w") as f:
            f.write("\n".join(lines))

        if not lines:
            stats["empty_label_files"] += 1
        stats["converted"] += 1

    # Write dataset.yaml
    ds_yaml = {
        "path":  str(OUTPUT_ROOT.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(OUTPUT_ROOT / "dataset.yaml", "w") as f:
        yaml.dump(ds_yaml, f, default_flow_style=False)

    print("\n=== Conversion complete ===")
    for k, val in stats.items():
        print(f"  {k}: {val}")

    # Sanity check
    n_empty = stats["empty_label_files"]
    n_total = stats["converted"]
    pct_empty = n_empty / max(n_total, 1) * 100
    print(f"\n  Empty label files: {n_empty}/{n_total} ({pct_empty:.1f}%)")
    if pct_empty > 50:
        print("  WARNING: >50% empty labels — check DATA_ROOT path")
    elif pct_empty > 20:
        print("  NOTE: Some empty labels expected (objects outside FOV)")
    else:
        print("  Labels look healthy.")


if __name__ == "__main__":
    convert()