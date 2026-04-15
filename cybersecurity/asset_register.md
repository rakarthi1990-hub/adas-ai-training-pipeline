\# Asset Register — ADAS AI Perception Pipeline



\## Item

\*\*ADAS AI Perception Pipeline\*\*



\## Purpose

Define the main cybersecurity-relevant assets within the perception pipeline and evaluate them against Confidentiality, Integrity, and Availability (CIA) properties.



| Asset ID | Asset | Description | Confidentiality | Integrity | Availability |

|---|---|---|---|---|---|

| A1 | Sensor input frames | Camera/image inputs used for detection and evaluation | Low | High | High |

| A2 | Dataset and annotation pipeline | nuScenes-derived training data, labels, preprocessing flow | Low | High | Medium |

| A3 | Trained model weights (.pt, .onnx) | Exported model artifacts used in validation and deployment benchmarking | Medium | High | High |

| A4 | ONNX inference runtime | Runtime environment executing the deployed model | Low | High | High |

| A5 | Deployment and benchmarking outputs | Performance results, logs, latency reports, exported plots | Low | Medium | Medium |



\## Notes

\- \*\*Integrity\*\* is the highest-priority property for most assets because unauthorized modification can directly affect perception behavior.

\- \*\*Availability\*\* is critical for runtime and sensor inputs because denial of inference or missing input data can break ADAS timing expectations.

\- \*\*Confidentiality\*\* is generally lower priority in this project, but model artifacts still have moderate relevance because unauthorized extraction or replacement can affect deployment trust.

