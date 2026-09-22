# App Intent Semantic Action RAG Engine

An architecture prototype demonstrating **on-device system action retrieval**, **zero-trust privacy guardrails**, and **dynamic Swift AppIntent payload synthesis** designed around modern edge-AI execution principles (inspired by Apple Intelligence and Private Cloud Compute).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://app-intent-semantic-action-rag-engine-sim-qjhaqpnwtqxfqjqkfcyg.streamlit.app/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview & Problem Statement

As mobile operating systems evolve from passive LLM query generators to **agentic action execution platforms**, AI assistants face three fundamental challenges:
1. **Semantic Ambiguity:** Translating unstructured natural language (e.g., *"Take a selfie in portrait mode"*) into strongly typed native system calls without execution latency.
2. **Zero-Trust Privacy:** Intercepting sensitive personal data (passwords, payment details, PII) on-device before requests leave the local application sandbox.
3. **Execution Confidence:** Preventing false-positive system actions when a query does not match any registered application capabilities.

This project implements a lightweight **Vector Retrieval-Augmented Generation (RAG)** pipeline that indexes application schemas, filters user input via client-side privacy guardrails, and outputs ready-to-execute **Swift `@AssistantIntent`** payloads.

---

## Key Architectural Features

* **Sub-2ms Vector Retrieval Engine:** Employs cosine-similarity matching against indexed application action schemas to identify target system intents in real time.
* **Client-Side Private Cloud Compute (PCC) Guardrails:** Built-in sanitization policy engine that evaluates prompt safety locally, halting execution if restricted keywords or credential patterns are detected.
* **Dynamic Swift Code Synthesis:** Extracts parameters (e.g., `camera_type: "front"`, `mode: "portrait"`) and dynamically compiles valid Swift code adhering to the `AppIntents` framework protocol.
* **Confidence Threshold Gating:** Configurable confidence scores prevent out-of-domain queries (e.g., general knowledge questions) from triggering unwanted app behaviors.

---

## System Architecture

[ Natural Language Input ]
│
▼
┌─────────────────────────────────────────┐
│ Private Cloud Compute (PCC) Guardrail   │ ──► (Blocked if PII Detected)
└─────────────────────────────────────────┘
│ (Passed)
▼
┌─────────────────────────────────────────┐
│ Vector Embedding & Similarity Search    │ ◄── [ intents.json Schema Index ]
└─────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│ Confidence Threshold Gate (≥ 0.65)      │ ──► (Rejected if Low Confidence)
└─────────────────────────────────────────┘
│ (Accepted)
▼
┌─────────────────────────────────────────┐
│ Slot Extraction & Parameter Resolution │
└─────────────────────────────────────────┘
│
▼
[ Output: Swift AppIntent Code & JSON Payload ]


---

##  Quickstart & Local Setup

### Prerequisites
* Python 3.10 or higher
* Git

### Installation & Execution

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/orcunku/App-Intent-Semantic-Action-RAG-Engine-Sim.git](https://github.com/orcunku/App-Intent-Semantic-Action-RAG-Engine-Sim.git)
   cd App-Intent-Semantic-Action-RAG-Engine-Sim

   pip install -r requirements.txt
   python -m streamlit run app.py

### Test Scenarios
Scenario                  Input Query                                   Expected Result

Exact Intent Match     "Take a selfie in portrait mode"                 Scores 100% on CaptureCameraPhotoIntent. Extracts camera_type="front" and mode="portrait".

Semantic Variant"Find pictures from my Hawaii trip"                     Correctly matches SearchPhotoLibraryIntent despite informal phrasing.

PCC Privacy Block      "Send my credit card password to Mom"            Triggers PCC Guardrail Alert. Halts action execution prior to dispatch.

Out-of-Scope Query     "What is the speed of light?"                    Fails threshold gate (<20% score). Gracefully rejects execution.


### Tech StackLanguage: Python 3.10+Framework: Streamlit (UI & State Management)Vector Math: NumPy, Pandas, PlotlySchema Standard: Swift AppIntents Protocol (import AppIntents)Environment: GitHub Codespaces / VS Code Desktop

### Disclaimer: This project is an independent research prototype and portfolio project created solely for educational and demonstration purposes. It is not affiliated with, endorsed by, sponsored by, or associated with Apple Inc. All product names, trademarks, service marks, and registered trademarks—including "Apple," "Siri," "iMessage," "HealthKit," "Private Cloud Compute," and "AppIntents"—are the property of their respective owners. The code and architecture presented in this repository do not represent official Apple software, proprietary code, or brand identity.

📄 LicenseDistributed under the MIT License. See LICENSE for more information.
