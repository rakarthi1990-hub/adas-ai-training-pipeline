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
| Low-visibility objects (0–40%) | [your number] | Primary safety-critical detection risk |
| Zero LiDAR point objects | [your number] | Sensor fusion gap — camera-only fallback needed |
| Dominant classes | car, pedestrian | Aligns with ADAS safety-critical targets |


\## Stack

Python · PyTorch · nuScenes devkit · Matplotlib · YOLOv8 (Week 2)

