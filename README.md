# ADAS AI Training Pipeline
**Karthikeyan Rajan** | Senior ADAS Systems Engineer

End-to-end AI-based object detection and safety evaluation pipeline
for automotive ADAS perception, built on the nuScenes dataset.

Developed to demonstrate hands-on ML pipeline ownership alongside
production ADAS system architecture experience (LKA/LDW SOP delivery,
ADAS-CCU dual-SoC/MCU architecture, ISO 26262 / SOTIF).

---

## Project structure

| Folder | Purpose |
|---|---|
| `data/` | nuScenes parsing, exploration, YOLO conversion, DataLoader |
| `src/` | YOLOv8 training loop, evaluation, per-class metrics |
| `scene_mining/` | Safety-critical scenario extraction by ODD category |
| `kpi/` | Metrics framework, ODD-level KPI reporting |
| `safety/` | Failure chain analysis, SOTIF mapping |
| `outputs/` | Generated plots and reports per week |

---

## Results summary

### Week 1 — Dataset exploration

| Finding | Value | ADAS relevance |
|---|---|---|
| Total annotated samples | 404 | Training corpus size |
| Peak scene density | 156 objects/frame | Perception stress ceiling |
| Low-visibility objects (0–40%) | Identified | Primary safety-critical risk |
| Zero LiDAR point objects | Detected | Sensor fusion gap |

### Week 2 — YOLOv8 training pipeline

Model: YOLOv8n — transfer learning, frozen backbone (layers 0–9)
Dataset: nuScenes mini — 323 train / 81 val | Hardware: CPU only

| Metric | Train | Val |
|---|---|---|
| mAP@50 | 0.210 | 0.029 |
| Precision | 0.560 | 0.028 |
| Recall | 0.214 | 0.205 |
| Pedestrian AP@50 | — | 0.058 |
| Bicycle AP@50 | — | 0.003 |

**Key findings:**
- Large train/val gap confirms overfitting on 323 samples — production systems require 10k–100k+ frames per ODD
- Recall 0.21 consistent across splits — model misses ~79% of objects regardless of seen/unseen data
- Pedestrian AP 0.058, bicycle AP 0.003 — both below any production safety threshold
- In a real ADAS pipeline: recall 0.21 → missed detection → no braking decision → collision risk (ISO 26262 / SOTIF failure chain)

### Week 3 — Safety-critical scene mining

Safety-critical scenes identified: **358 / 404 (88.6%)**

| Mining category | Scenes | % of dataset | Safety priority |
|---|---|---|---|
| Low visibility (0–40%) | 353 | 87.4% | Critical |
| Dense urban (10+ objects) | 326 | 80.7% | High |
| Pedestrian risk (2+ peds) | 284 | 70.3% | Critical |

**Why 88.6% is expected:** nuScenes is a production urban dataset (Boston + Singapore city centres). Urban driving ODDs are inherently dense and pedestrian-heavy. High safety-critical scene proportion confirms the dataset is representative of real Level 2+ ADAS conditions.

**Root causes of recall 0.21 — identified by scene mining:**
1. 87.4% of scenes contain low-visibility objects → confidence collapse
2. 80.7% contain dense overlapping detections → suppressed by NMS
3. 70.3% contain 2+ pedestrians → safety-critical class underperformance

### Week 4 — Safety failure analysis & SOTIF mapping *(in progress)*

---

## Setup
```bash
pip install -r requirements.txt
# Register and download nuScenes mini (~4GB) from nuscenes.org
# Extract to ./data/nuscenes/

# Week 1 — explore dataset
cd data && python explore.py

# Week 2 — convert and train
python nuscenes_to_yolo.py
cd ../src && python train.py
python evaluate.py

# Week 3 — scene mining
cd ../scene_mining && python mine_scenarios.py
```

## Stack
Python · PyTorch · YOLOv8 (Ultralytics) · nuScenes devkit ·
Matplotlib · pyquaternion | ISO 26262 / SOTIF safety framing
