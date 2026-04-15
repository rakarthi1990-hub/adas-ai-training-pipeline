\# Threat Analysis and Risk Assessment (TARA)

\## ADAS AI Perception Pipeline



\## Scope

This TARA focuses on key cybersecurity threats affecting the ADAS perception pipeline, aligned with ISO/SAE 21434 concepts.



\---



\## Risk Evaluation Criteria



\- \*\*Impact\*\*: Effect on system safety or functionality (Low / Medium / High)

\- \*\*Feasibility\*\*: Ease of executing the attack (Low / Medium / High)

\- \*\*Risk Level\*\*: Derived from Impact + Feasibility



\---



\## TARA Table



| Threat ID | Threat Scenario | Asset | Attack Path | CIA Impact | Impact | Feasibility | Risk Level | Treatment |

|---|---|---|---|---|---|---|---|---|

| T1 | Adversarial input causing missed detection (e.g., pedestrian not detected) | Sensor input frames | Manipulated or perturbed camera input | Integrity | High | Medium | High | Add robustness testing and input validation |

| T2 | Model weight tampering (malicious ONNX replacement) | Model artifacts (.pt, .onnx) | Unauthorized modification of model file before deployment | Integrity | High | Medium | High | Implement model integrity verification (hash/signing) |

| T3 | Data poisoning during training (corrupted annotations) | Dataset pipeline | Injection of incorrect labels or training samples | Integrity | High | Low | Medium | Validate dataset integrity and control data sources |

| T4 | Inference denial (latency spike / resource exhaustion) | Inference runtime | Repeated or oversized input causing performance degradation | Availability | Medium | Medium | Medium | Monitor runtime performance and enforce input limits |



\---



\## Key Observations



\- \*\*Integrity is the most critical property\*\* — most attacks aim to manipulate perception behavior.

\- Adversarial inputs and model tampering have the \*\*highest system impact\*\*, as they can directly lead to unsafe ADAS decisions.

\- Runtime availability is also important due to real-time constraints (\~33 ms latency budget).



\---



\## Traceability to Project Findings



\- T1 → Linked to low-visibility and missed detection scenarios identified in Week 3 (scene mining)

\- T2 → Linked to ONNX export and deployment artifacts from Week 5

\- T3 → Linked to dataset ingestion and preprocessing from Week 1–2

\- T4 → Linked to latency benchmarking results from Week 5



\---



\## Conclusion



This TARA highlights that cybersecurity threats in ADAS perception systems directly translate into safety risks. Therefore, cybersecurity controls must be integrated alongside safety validation and deployment constraints.

