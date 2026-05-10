# 🚗 Vehicle Service Center: Advanced Hybrid Simulation Platform

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![SimPy](https://img.shields.io/badge/SimPy-Simulation-green.svg)](https://simpy.readthedocs.io/)
[![Mesa](https://img.shields.io/badge/Mesa-ABM-orange.svg)](https://mesa.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Premium_UI-red.svg)](https://streamlit.io/)

A state-of-the-art simulation platform evaluating operational efficiencies in a Heavy-Duty Vehicle Service Center. This platform implements a **Hybrid Modelling Approach**, fusing **Discrete Event Simulation (DES)** with **Agent-Based Modelling (ABM)** for unmatched behavioral accuracy.

---

## 🏗️ Core Architecture & Innovation

### 1. Hybrid Simulation Engine
The platform operates on a dual-layer architecture:
- **DES Layer (SimPy):** Manages the physical environment, resource constraints (Advisors, Inspection Bays, Service Bays), and the linear workflow of vehicles.
- **ABM Layer (Mesa):** Models each customer as an autonomous agent with distinct personality types (**Conservative, Steady, Aggressive**) and dynamic emotional states (**Liu-Zhen Model**).

### 2. Statistical Rigor
- **NHPP Arrival Process:** Uses a **Non-Homogeneous Poisson Process** with Lewis-Shedler thinning to model time-dependent arrival rates (rush hours vs. off-peak).
- **Lognormal Service Times:** All service durations follow Lognormal distributions for realistic right-skewed non-negative time modeling.
- **Priority Queuing:** Resources utilize `simpy.PriorityResource` where agitated/aggressive agents can receive queue priority under specific conditions.

### 3. Psychological Decision Engine
- **Dynamic Balking:** Agents evaluate queue lengths at entry and decide to leave if the wait exceeds their personality-driven threshold.
- **Continuous Reneging:** Implementation of the **Liu-Zhen Continuous Emotional Decay**. Emotion decays exponentially over wait time: $E(t) = E_0 e^{-\alpha t}$. If $E(t)$ falls below a threshold, the agent reneges (leaves the queue).

---

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/ak-rahul/vehicle-service-sim.git
cd vehicle-service-sim

# Set up environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install optimized dependencies
pip install -r requirements.txt
```

### 2. Run the Premium Dashboard (Recommended)
Experience the full analytical power with interactive visualizations and optimization tools.
```bash
streamlit run app.py
```

### 3. Command Line Interface (CLI)
Run head-to-head comparisons of DES and Hybrid models with detailed KPI output.
```bash
python run.py --mode both --time 600 --verbose
```

---

## 📊 Analytical Capabilities

- **KPI Comparison:** Evaluate Throughput Rate, Balk Rate, Renege Rate, and Multi-stage Wait Times.
- **Vectorized Analysis:** High-performance data processing using vectorized Pandas operations.
- **Parallel Optimization Sweep:** Brute-force parameter optimization parallelized across CPU cores using `ProcessPoolExecutor`.

---

## 🏆 Resource Optimization Results
The platform includes an automated search engine to minimize total wait time. 
**Typical Optimal Setup:**
- **Advisors:** 2
- **Inspection Bays:** 2
- **General Bays:** 6
- **Express Bays:** 2
*(Results vary based on arrival intensity $\lambda$ and service rate $\mu$)*

---

## 📜 Documentation & methodology
Detailed methodology, including the Liu-Zhen model parameters and flowchart logic, can be found in the `docs/` directory.

---
© Developed for Advanced Agentic Coding.
