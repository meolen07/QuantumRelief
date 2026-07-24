# QuantumRelief

**Quantum Intelligence. Human Relief.**

Team 5 — **Quantrio** · QC4SG SEA Hackathon

[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://quantumrelief.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](runtime.txt)
[![PennyLane](https://img.shields.io/badge/QML-PennyLane-19B244)](https://pennylane.ai)
[![License](https://img.shields.io/badge/License-see%20LICENSE-blue)](LICENSE)

Live demo: **[quantumrelief.streamlit.app](https://quantumrelief.streamlit.app)**

---

## Overview

QuantumRelief predicts **next-hop emergency escape routes** on the Manila **Intramuros** road network under expanding earthquake and exit-traffic hazards. A **Hybrid Quantum–Classical FiLM** model (PennyLane PHN) is the hero path; Classical Dijkstra is shown for comparison only.

Adapted from Haboury et al., *[Quantum Machine Learning for Disaster Response](https://arxiv.org/abs/2307.15682)* (Furubira → Manila). Surfaces: **Streamlit Crisis UX** + **FastAPI B2B Quantum Routing API**.

| Checkpoint | Notes |
| --- | --- |
| Hybrid QML (`film_hybrid.pt`) | Val acc ≈ **0.92** · Quantum contribution ≈ **34%** · `demo_mode=False` |
| Route smoke | **3/3** EXIT REACHED (no Dijkstra assist) |

---

## Key features

- **Hybrid QML hero** — PennyLane PHN FiLM; bold green path on the map
- **Classical baseline** — Dijkstra dashed overlay for judge comparison
- **Dynamic hazards** — expanding \(r_{epi}\) / \(r_{exit}\) rings scrubbed by simulation time `t`
- **Crisis UX** — Folium map-click Start / Epicenter / Exit
- **B2B API** — FastAPI `/api/v1/calculate_route` for partners and ops systems
- **Offline-ready** — cached GraphML, dataset, and trained checkpoints shipped in-repo

---

## Architecture

```mermaid
flowchart LR
  UX[Streamlit Crisis UX] --> RS[routing_service]
  API[FastAPI B2B API] --> RS
  RS --> G[OSMnx / NetworkX graph]
  RS --> Dyn[Dynamic weights]
  RS --> HQ[Hybrid FiLM PHN]
  HQ --> PL[PennyLane]
  RS --> CF[Classical FiLM]
```

| Paper (Furubira) | QuantumRelief (Manila) |
| --- | --- |
| OSMnx city graph | Intramuros bbox, degree-capped, cached GraphML |
| 3 exits + random epicenter | Perimeter exits + map-click epicenter |
| Algorithm 1 dynamic weights | `src/dynamic_simulation.py` |
| Table I input size 36 | Same layout, local km projection |
| Classical + Quantum FiLM PHN | Classical ablation + **Hybrid QML hero** |

Radii: \(r_{epi} = 0.5 + \sqrt{0.0002\, t}\), \(r_{exit} = \sqrt{0.00075\, t}\).

Neighbor logits are masked to real degree; a light Dijkstra assist may finish a stalled path — branding remains **Hybrid QML**.

---

## Quick start — Streamlit

```bash
cd QuantumRelief
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Graph, dataset, and checkpoints under `data/` and `models/` are included. OSM download runs only if the GraphML cache is missing.

---

## Quantum Routing API — FastAPI

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-api.txt
uvicorn api:app --reload --host 0.0.0.0 --port 8000
# or: python api.py
```

```bash
curl -s http://127.0.0.1:8000/
# → {"status":"QuantumRelief API is running"}

curl -s -X POST http://127.0.0.1:8000/api/v1/calculate_route \
  -H "Content-Type: application/json" \
  -d '{
    "start_coords": [14.5895, 120.9750],
    "epicenter_coords": [14.5850, 120.9780],
    "exit_coords": [14.5920, 120.9720]
  }'
```

OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## How to use (Crisis UX)

1. Sidebar: set click mode **Start → Epicenter → Exit**
2. **Click the Folium map** (Start/Exit snap to nearest road node)
3. Keep **Hybrid QML (PennyLane)** selected
4. Press **Calculate Escape Route**
5. Scrub simulation time **`t`** — red \(r_{epi}\) / yellow \(r_{exit}\) expand
6. Compare **bold green Hybrid** vs **dashed Dijkstra**; read Exit Reached + Quantum Contribution

In-app: expander **How to use QuantumRelief** (sidebar **How to use**).

Optional VN: *Chọn mode → click bản đồ → Calculate → kéo slider `t`.*

---

## Project structure

```
QuantumRelief/
  runtime.txt              # Streamlit Cloud: python-3.11
  requirements.txt         # Cloud / Streamlit (numpy → torch → pennylane)
  requirements-api.txt     # FastAPI + uvicorn
  app.py                   # Crisis UX — map-click + Hybrid hero
  api.py                   # B2B Quantum Routing API
  data/                    # GraphML + routing_dataset.npz
  models/                  # film_classical.pt, film_hybrid.pt
  src/
    graph_setup.py         # OSMnx / NetworkX / exits
    dynamic_simulation.py  # Algorithm 1 weights
    dataset_generation.py  # Table I vectors + Dijkstra labels
    film_model.py          # Classical FiLM
    quantum_hybrid.py      # PennyLane Hybrid PHN
    routing_service.py     # Shared Hybrid helpers (API + app)
  scripts/
    retrain_models.py
    generate_pitch_deck.py
```

---

## Models & data

| Asset | Role |
| --- | --- |
| `models/film_hybrid.pt` | Hybrid QML FiLM (PennyLane PHN) — demo hero |
| `models/film_classical.pt` | Classical FiLM ablation |
| `data/manila_intramuros_graph.graphml` | Cached Intramuros road graph |
| `data/routing_dataset.npz` | Training / eval samples |

**Retrain** (optional; CPU-bound — prefer shipped Hybrid checkpoint for demos):

```bash
source .venv/bin/activate
python -u scripts/retrain_models.py 400 100 10 6
```

**Smoke checks:**

```bash
python -c "from src.quantum_hybrid import quantum_status, load_hybrid_model; print(quantum_status()); load_hybrid_model()"
python -c "from src.graph_setup import load_or_build_graph; print(load_or_build_graph().number_of_nodes())"
```

---

## Deploy (Streamlit Community Cloud)

1. Push to GitHub (`meolen07/QuantumRelief`)
2. [share.streamlit.io](https://share.streamlit.io) → select repo → deploy / reboot if deps changed
3. Confirm logs: Python **3.11** (`runtime.txt`), `numpy` before `torch`, PennyLane import OK

Cloud pins live in **`requirements.txt`**. API deps stay in **`requirements-api.txt`** so Cloud stays lean.

Keep `numpy==1.26.4` before `torch==2.2.2` for Cloud ABI safety. If PennyLane install times out, Classical FiLM still runs; Hybrid shows unavailable.

---

## Team

**Quantrio** (Team 5) · QC4SG — SEA Hackathon  
Manila Intramuros emergency routing with Hybrid QML.

<!-- Contact: add emails / LinkedIn here -->

---

## Citation

Haboury et al., *A Hybrid Quantum-Classical Neural Network for Disaster Response*, [arXiv:2307.15682](https://arxiv.org/abs/2307.15682). QuantumRelief adapts the Furubira FiLM / PHN pipeline to Manila Intramuros.
