\# TARA Report — ADAS AI Perception Pipeline



\## Objective

Document the main cybersecurity risks identified for the ADAS AI perception pipeline and summarize the selected treatment approach.



\## Item

\*\*ADAS AI Perception Pipeline\*\*



\## Threat Summary



| Threat ID | Threat Scenario | Asset | CIA Property | Risk Level | Treatment Decision |

|---|---|---|---|---|---|

| T1 | Adversarial input causing missed detection | Sensor input frames | Integrity | High | Add robustness testing and input validation |

| T2 | Model weight tampering (malicious ONNX replacement) | Model artifacts (.pt, .onnx) | Integrity | High | Add model integrity verification |

| T3 | Data poisoning during training | Dataset and annotation pipeline | Integrity | Medium | Validate data source and annotation integrity |

| T4 | Inference denial / latency spike | ONNX inference runtime | Availability | Medium | Monitor performance and constrain input usage |



\## Residual Risk View

\- T1 remains partially open because only a basic adversarial example test is implemented.

\- T2 risk is reduced through SHA256-based model integrity verification.

\- T3 remains dependent on trusted dataset sourcing and preprocessing controls.

\- T4 remains dependent on runtime environment and deployment constraints.



\## Conclusion

Cybersecurity threats affecting perception systems can directly influence safety-relevant behavior. Therefore, risk treatment must be considered alongside safety validation and deployment benchmarking.

