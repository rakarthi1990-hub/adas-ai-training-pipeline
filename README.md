\# ADAS AI Training Pipeline

\*\*Karthikeyan Rajan\*\* | Senior ADAS Systems Engineer



An end-to-end AI-based object detection and evaluation pipeline

for automotive ADAS perception, built on the nuScenes dataset.



\## Motivation

This project extends hands-on AI development experience to complement

production-grade ADAS system architecture and series delivery background

(LKA/LDW SOP). It targets the full AI development lifecycle:

dataset curation → training → scene mining → closed-loop KPI evaluation

→ safety analysis.



\## Project structure

| Folder | Purpose |

|---|---|

| `data/` | Dataset parsing, exploration, DataLoader |

| `src/` | Model definition, training loop, evaluation |

| `scene\_mining/` | Safety-critical scenario extraction |

| `kpi/` | Metrics computation and ODD-level reporting |

| `safety/` | Failure mode analysis, SOTIF mapping |

| `notebooks/` | Jupyter walkthroughs per week |



\## Week 1 — Dataset exploration (current)

\- nuScenes mini dataset parsed and explored

\- Class distribution, visibility analysis, object density visualised

\- Safety-critical observations documented (low visibility, sparse LiDAR)

\- PyTorch DataLoader ready for Week 2 training



\## Dataset

nuScenes mini — \[nuscenes.org](https://www.nuscenes.org)  

700 training samples | 8 object classes | camera + radar + LiDAR



\## Setup

```bash

pip install -r requirements.txt

\# Download nuScenes mini to ./data/nuscenes/

cd data \&\& python explore.py

```
## Week 1 — Key findings

| Metric | Value | ADAS relevance |
|---|---|---|
| Peak scene density | 156 objects/frame | Stress-test for perception models |
| Low-visibility objects (0–40%) | 5480 | Primary safety-critical detection risk |
| Zero LiDAR point objects | 4278 | Sensor fusion gap — camera-only fallback needed |
| Dominant classes | car, pedestrian | Aligns with ADAS safety-critical targets |

## Week 2 — Training results

Model: YOLOv8n — transfer learning, frozen backbone (layers 0–9)  
Dataset: nuScenes mini — 323 train / 81 val samples  
Hardware: CPU only | Training time: 7.3 minutes

| Metric | Train | Val |
|---|---|---|
| mAP@50 | 0.210 | 0.029 |
| Precision | 0.560 | 0.028 |
| Recall | 0.214 | 0.205 |
| Pedestrian AP@50 | — | 0.058 |
| Bicycle AP@50 | — | 0.003 |

### Key engineering findings

**Overfitting on small dataset** — large train/val mAP gap (0.21 vs 0.03)
confirms the model cannot generalise from 323 samples. Production systems
require 10,000–100,000+ annotated frames per ODD.

**Recall consistent across splits (0.21)** — the model misses ~79% of
objects regardless of seen/unseen data. Root cause: frozen backbone
trained on COCO (generic objects) not adapted to automotive distance
and scale characteristics.

**Safety-critical class performance** — pedestrian AP@50: 0.058,
bicycle AP@50: 0.003. Both below any production threshold. In a
real ADAS pipeline, recall this low maps directly to missed detection
→ no braking decision → collision risk (ISO 26262 / SOTIF failure chain).

**Implication for Week 3** — low recall is not uniformly distributed.
Scene mining will identify which specific conditions (occlusion, night,
dense urban, small object distance) drive the worst misses — enabling
targeted data collection strategy.

\## Stack

Python · PyTorch · nuScenes devkit · Matplotlib · YOLOv8 (Week 2)

