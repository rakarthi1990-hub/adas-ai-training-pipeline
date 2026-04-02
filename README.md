# ADAS AI Training Pipeline

**Karthikeyan Rajan** | Senior ADAS Systems Engineer  
*AI-Integrated Architectures | Sensor Fusion | ISO 26262 | SOTIF*

---

An end-to-end AI-based object detection, evaluation, and safety analysis pipeline for automotive ADAS perception — built on the **nuScenes** autonomous driving dataset. 

Developed to bridge hands-on ML pipeline ownership with production ADAS system architecture experience: LKA/LDW SOP delivery, ADAS-CCU dual-SoC/MCU pipeline architecture, ISO 26262, and SOTIF.

---

## 🚀 Pipeline Overview

* **Week 1 — Dataset curation & exploration:** Class distribution · visibility analysis · scene density · LiDAR gaps
* **Week 2 — YOLOv8 transfer learning pipeline:** YOLO conversion · training · per-class evaluation · KPI metrics
* **Week 3 — Safety-critical scene mining:** ODD filtering · low visibility · dense urban · pedestrian risk
* **Week 4 — Safety failure analysis & SOTIF mapping:** Failure chains · risk matrix · production readiness assessment

---

## 📊 Results Summary

### Week 1 — Dataset Exploration
| Finding | Value | ADAS Relevance |
| :--- | :--- | :--- |
| **Total annotated samples** | 404 | Training corpus size |
| **Peak scene density** | 156 objects/frame | Perception stress ceiling |
| **Low-visibility objects** | Identified & quantified | Primary safety-critical risk |
| **Zero LiDAR point objects** | Detected | Sensor fusion gap — camera fallback needed |
| **Dominant classes** | Car, pedestrian | Aligns with ADAS safety-critical targets |

---

### Week 2 — YOLOv8 Training Pipeline
* **Model:** YOLOv8n — transfer learning, frozen backbone (layers 0–9)
* **Strategy:** Pretrained COCO weights, fine-tuned detection head on nuScenes domain data
* **Hardware:** CPU only | **Inference Speed:** 11.6ms/image

| Metric | Train | Val | Interpretation |
| :--- | :--- | :--- | :--- |
| **mAP@50** | 0.210 | 0.029 | Overfitting — small dataset |
| **Precision** | 0.560 | 0.028 | Poor generalisation to unseen scenes |
| **Recall** | 0.214 | 0.205 | Consistent — misses ~79% of objects |
| **Pedestrian AP@50** | — | 0.058 | Safety-critical gap |
| **Bicycle AP@50** | — | 0.003 | Near-zero — safety risk |

> **Key Engineering Findings:**
> * Large train/val mAP gap confirms overfitting. Production systems require 10k–100k+ annotated frames per ODD.
> * Recall 0.21 is consistent: the model learned spatial patterns but misses ~79% of objects due to the frozen COCO backbone not adapting to automotive scale.
> * Pedestrian/Bicycle AP reflects a failure chain mapping directly to: *missed detection → no braking decision → collision risk* (ISO 26262 / SOTIF).

---

### Week 3 — Safety-Critical Scene Mining
**Safety-critical scenes identified: 358 / 404 (88.6%)**

| Mining Category | Scenes | % of Dataset | Safety Priority | SOTIF Class |
| :--- | :--- | :--- | :--- | :--- |
| **Low visibility (0–40%)** | 353 | 87.4% | Critical | Known unsafe |
| **Dense urban (10+ obj/frame)** | 326 | 80.7% | High | Known unsafe |
| **Pedestrian risk (2+ peds)** | 284 | 70.3% | Critical | Known unsafe |

**Insight:** nuScenes is a production urban dataset (Boston/Singapore). High safety-critical scene proportion confirms the dataset is fully representative of real Level 2+ ADAS operating conditions.

---

### Week 4 — Safety Failure Analysis & SOTIF Mapping
| ODD Condition | Failure Chain | SOTIF Stage | ISO 26262 Ref |
| :--- | :--- | :--- | :--- |
| **Low visibility** | Confidence collapse → no detection → no braking | Known unsafe | ASIL-B minimum |
| **Dense urban** | NMS suppression → incomplete object list → collision | Known unsafe | SOTIF ISO/PAS 21448 |
| **Pedestrian risk** | Missed detection → no yield → pedestrian struck | Known unsafe | ASIL-C/D for EPB |

#### Production Readiness Assessment
| KPI | Current | Minimum Gate | Target |
| :--- | :--- | :--- | :--- |
| Overall mAP@50 | 0.029 | 0.40 | 0.65 |
| Pedestrian recall | ~0.20 | 0.80 | 0.90 |
| Bicycle recall | ~0.05 | 0.75 | 0.85 |

**Status:** 🔴 **NOT production ready.** This project identifies *why* the model fails and defines the roadmap (targeted augmentation of low-visibility/urban data) to reach safety gates.

---

## 📂 Project Structure

| Folder | Contents |
| :--- | :--- |
| `data/` | `scene_parser.py` · `explore.py` · `nuscenes_to_yolo.py` |
| `src/` | `train.py` · `evaluate.py` |
| `scene_mining/` | `mine_scenarios.py` |
| `safety/` | `failure_analysis.py` · `sotif_mapping.md` |
| `outputs/` | Week 1-4 reports, KPIs, and failure chain diagrams |

---

## 🛠 Setup & Reproduction

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Download nuScenes mini** (~4GB) from [nuscenes.org](https://www.nuscenes.org) and extract to `./data/nuscenes/`.
3.  **Run Pipeline:**
    ```bash
    # Week 1: Data Exploration
    python data/explore.py
    # Week 2: Conversion & Training
    python data/nuscenes_to_yolo.py
    python src/train.py
    # Week 3 & 4: Safety Analysis
    python scene_mining/mine_scenarios.py
    python safety/failure_analysis.py
    ```

---

## 💻 Tech Stack
* **Dataset:** nuScenes mini
* **Model:** YOLOv8n (Ultralytics)
* **Tools:** PyTorch · NumPy · Pandas · Matplotlib
* **Safety Framework:** ISO 26262 · SOTIF (ISO/PAS 21448)

---

## 👤 Author
**Karthikeyan Rajan** [LinkedIn](https://www.linkedin.com/in/karthikeyan-rajan-a77059111/) · [GitHub](https://github.com/rakarthi1990-hub)
