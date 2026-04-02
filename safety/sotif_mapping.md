\# SOTIF Mapping — ADAS AI Perception Pipeline

\*\*Author:\*\* Karthikeyan Rajan | Senior ADAS Systems Engineer

\*\*Standard:\*\* ISO/PAS 21448 (SOTIF) | ISO 26262

\*\*Date:\*\* April 2026



\---



\## Purpose



This document maps the experimental findings from the ADAS AI

Training Pipeline project to the SOTIF (Safety Of The Intended

Functionality) taxonomy. It demonstrates that AI perception

failures are not random — they are structured, identifiable, and

manageable through the SOTIF framework.



\---



\## SOTIF Taxonomy Applied to AI Perception



| SOTIF Stage | Definition | This Project |

|---|---|---|

| Unknown unsafe | Failure modes not yet identified | Starting point — before scene mining |

| Known unsafe | Failure modes identified, not yet mitigated | After Week 3 mining — 3 conditions documented |

| Mitigated | Countermeasures defined and validated | Week 4 recommendations |

| Acceptable residual risk | Remaining risk within ALARP | Production release gate |



\---



\## Scenario 1 — Low Visibility Perception Failure



\*\*SOTIF trigger:\*\* Performance limitation of AI perception in

low-visibility conditions (0–40% object visibility).



\*\*Evidence from this project:\*\*

\- 353 / 404 scenes (87.4%) contain low-visibility objects

\- Pedestrian AP@50 = 0.058 on val split

\- Failure chain: confidence collapse → no detection → no braking



\*\*SOTIF classification:\*\* Known unsafe scenario

(performance limitation within defined ODD)



\*\*Mitigation strategy:\*\*

\- Sensor fusion: radar confirmation of camera detections

\- ODD monitoring: automated visibility estimation

\- ODD exit: handover request when visibility < threshold

\- Data: 2,000+ targeted low-visibility training frames



\---



\## Scenario 2 — Dense Scene Detection Suppression



\*\*SOTIF trigger:\*\* Intended functionality limitation — NMS

suppresses valid detections when IoU overlap is high in

dense urban scenes.



\*\*Evidence from this project:\*\*

\- 326 / 404 scenes (80.7%) exceed 10-object density threshold

\- Overall recall 0.21 — NMS suppression confirmed as root cause

\- Failure chain: valid boxes suppressed → incomplete object list

&#x20; → incorrect path prediction → collision



\*\*SOTIF classification:\*\* Known unsafe scenario

(NMS behaviour is a documented limitation of anchor-based detectors)



\*\*Mitigation strategy:\*\*

\- Adaptive NMS: lower IoU threshold in high-density ODDs

\- Anchor-free detection head (e.g. FCOS, CenterNet)

\- Multi-camera fusion to reduce single-viewpoint occlusion



\---



\## Scenario 3 — Pedestrian Class Underperformance



\*\*SOTIF trigger:\*\* Insufficient training data coverage for

safety-critical object class (pedestrian, bicycle) in

production ODD conditions.



\*\*Evidence from this project:\*\*

\- Pedestrian AP@50 = 0.058 (val split)

\- Bicycle AP@50 = 0.003 (val split)

\- 284 / 404 scenes (70.3%) contain 2+ pedestrians

\- Failure chain: missed pedestrian → no yield → collision at crossing



\*\*SOTIF classification:\*\* Known unsafe scenario

(class imbalance and insufficient data — known ML limitation)



\*\*ISO 26262 reference:\*\* ASIL-C/D for emergency pedestrian

braking (EPB) functions. Current AP 0.058 is below ASIL-B minimum.



\*\*Mitigation strategy:\*\*

\- Class-weighted loss function (pedestrian weight ×3)

\- Dedicated pedestrian detection head

\- 3,000+ pedestrian-specific training frames

\- Higher input resolution (640px → 1280px) for small distant peds



\---



\## Production Readiness Gate Criteria



Before any of these scenarios can be closed as "mitigated" in a

production ADAS programme, the following KPI gates must be passed

in closed-loop simulation and real-world validation:



| KPI | Current | Minimum gate | Target |

|---|---|---|---|

| Overall mAP@50 | 0.029 | 0.40 | 0.65 |

| Pedestrian recall | \~0.20 | 0.80 | 0.90 |

| Bicycle recall | \~0.05 | 0.75 | 0.85 |

| False positive rate | — | < 0.05/frame | < 0.02/frame |

| Low-visibility recall | — | 0.60 | 0.75 |



\---



\## Summary



This project has progressed three safety scenarios from

\*\*unknown unsafe\*\* to \*\*known unsafe\*\* through:



1\. Quantitative model evaluation (Week 2 metrics)

2\. ODD-based scene mining (Week 3 — 88.6% safety-critical)

3\. Structured failure chain analysis (Week 4)



The next step in a production programme would be to move

these from known unsafe to mitigated through targeted data

collection, model improvements, and closed-loop KPI validation —

exactly the workflow described in BMW's Senior AI ADAS Engineer

job requirements.

