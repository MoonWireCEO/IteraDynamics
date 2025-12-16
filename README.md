````markdown
# Itera Dynamics: Quantitative Signal Platform

![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)
![Architecture](https://img.shields.io/badge/architecture-monorepo-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)

> **Market-Agnostic Regime-Adaptive Trading Architecture**

---

## 📖 Overview

**Itera Dynamics** is an institutional-grade quantitative research and execution platform designed for non-stationary markets. It features a modular, event-driven architecture that decouples signal generation (`apex_core`) from the live execution and monitoring layers (`src/`).

The system prioritizes **capital preservation** through a closed-loop governance layer, utilizing automated drift detection, volatility gating, and artifact lineage to ensure production safety. The primary operational environment is **Moonwire** (digital assets) running on a live, hourly schedule.

---

## 🏗 System Architecture

The core philosophy is a **Cybernetic Feedback Loop**: signals are not just fired; they are governed, monitored, and used to recalibrate the engine in real-time.

```mermaid
graph LR
    subgraph Data_Layer [Data Layer]
        A[Multi-Source Ingestion] --> B(Feature Engineering)
    end

    subgraph Cortex [Apex Cortex - The Brain]
        B --> C{Regime Filter}
        C -. "Weighting" .-> D[Ensemble Engine]
        B --> D
        D --> E{Risk Governance}
    end

    subgraph Exec [Execution & Monitoring]
        E -- "Pass" --> F[Order Routing via RealBroker]
        E -- "Fail" --> G((Kill Switch))
        F -. "Feedback/Fill" .-> D
        F -- "Logs & Metrics" --> H[Mission Control Dashboard]
    end

    style E fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#bbf,stroke:#333,stroke-width:2px
````

---

## 📂 Project Structure

Itera Dynamics is structured as a streamlined monorepo, focusing on clear separation between the core signal logic, the execution environment, and the monitoring layer.

```
IteraDynamics_Mono/
├── apex_core/                  # 🧠 The Core Library (Asset-Agnostic Signal Logic)
│   └── signal_generator.py     # Primary trading strategy execution
│
├── src/                        # ⚙️ Execution and Monitoring Layer
│   ├── real_broker.py          # Live API Adapter (Coinbase/GDAX)
│   └── dashboard.py            # Streamlit Mission Control Dashboard (Cross-Platform CSS Optimized)
│
├── data/                       # 📊 Execution Artifacts
│   ├── flight_recorder.csv     # Equity curve history
│   └── argus_execution.log     # Verbose scheduler/execution logs
│
├── run_live.py                 # 🚀 Main Entry Point: Hourly Scheduler (Using Python's Logging Module)
├── pyproject.toml              # Build System Configuration
└── README.md                   # System Documentation
```

---

## ⚡ Key Capabilities

### 🛡️ 1. Closed-Loop Governance

* **Drift Detection:** Monitors feature importance decay and Sharpe degradation in real-time.
* **Execution Gates:** Pre-trade checks for liquidity, spread, and account health.
* **Global Kill-Switch:** Automated "Shadow Mode" transition if drawdown breaches defined thresholds (Default: -25%).

### 🔬 2. Research Hygiene

* **Purged Walk-Forward Validation:** Implements 7-fold temporal splitting with Embargo Gaps to eliminate look-ahead bias.
* **Artifact Provenance:** Every signal is cryptographically anchored to the specific model version and data snapshot used.

### ⚙️ 3. Execution Engine

* **Real Broker Integration:** Direct, production-ready integration with Coinbase for live execution.
* **Hourly Scheduler:** Dedicated `run_live.py` service for disciplined, automated trading at market open (top of the hour).
* **Live Mission Control:** Cross-platform web dashboard providing real-time equity, exposure, and verbose log streaming.

---

## 🚀 Installation & Usage

### 1. Clone & Install

This project uses a `pyproject.toml` configuration for editable installs.

```bash
git clone https://github.com/YourUsername/IteraDynamics_Mono.git
cd IteraDynamics_Mono
pip install -e .
```

### 2. Configure Live Broker Access

Set your Coinbase API keys and Portfolio UUID in your environment variables or the `.env` file for the live execution broker (`src/real_broker.py`) to connect.

### 3. Run the Live Scheduler (Execution)

This is the main bot entry point. It runs silently, logging all execution and hold events.

```bash
python run_live.py
```

### 4. Launch Mission Control (Monitoring)

Run the dashboard in a separate terminal. Access it via the local URL (or your Cloudflare tunnel if configured).

```bash
python -m streamlit run src/dashboard.py
```

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

> **Disclaimer:** This software is for educational and research purposes only. Trading quantitative strategies involves substantial risk of loss.

```
