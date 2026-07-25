# QuantumRelief

**Quantum Intelligence. Human Relief.**

Team 5 — **Quantrio** · QC4SG SEA Quantathon

[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://quantumrelief.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](runtime.txt)
[![PennyLane](https://img.shields.io/badge/QML-PennyLane-19B244)](https://pennylane.ai)
[![License](https://img.shields.io/badge/License-see%20LICENSE-blue)](LICENSE)

Live demo: **[quantumrelief.streamlit.app](https://quantumrelief.streamlit.app)**

**Tagline:** Hybrid delivers near-Dijkstra quality with quantum-classical local inference.

---

## Overview

QuantumRelief predicts **next-hop emergency escape routes** on the Manila **Intramuros** road network under expanding earthquake and exit-traffic hazards. A **Hybrid Quantum–Classical FiLM** model (PennyLane PHN) is the hero path; Classical FiLM is an ablation; Dijkstra is the full-information optimal baseline.

Adapted from Haboury et al., *[Quantum Machine Learning for Disaster Response](https://arxiv.org/abs/2307.15682)* (Furubira → Manila).

**Two surfaces**

| Surface | Audience | Role |
| --- | --- | --- |
| **B2C Emergency Escape** | Citizens | Map-click Start / Epicenter / Exit → 3-way Hybrid · Classical · Dijkstra |
| **Command Center (God View)** | B2G commanders | City-wide evacuation: flood / bridge hazards, arterial heatmap |

God View is honest about scale: **Dijkstra handles bulk routing**; Hybrid QML runs on **≤4 hero corridors**. “Scaled citizens” (`batch × 1,428`) is a pitch narrative (~14k at batch 10), not 14k Hybrid inferences.

**UI palette (Lovable-aligned):** deep navy · **cyan** Hybrid · **gold** Classical · orange accents · **red** hazard · white/light dashed Dijkstra.

---

## Results (latest hard retrain)

From `data/retrain_report.json` — `hard=true`, **~18,932** samples, **hazard×2**, 1000 episodes, 24 route trials.

| Metric | Hybrid | Classical | Dijkstra |
| --- | --- | --- | --- |
| Val accuracy | ≈ **0.890** | ≈ **0.886** | — |
| Mean travel time | ≈ **10.37** | ≈ **9.83** | ≈ **9.98** |
| Exit reached | **100%** | **100%** | **100%** |
| Path overlap vs Dijkstra | ≈ **57.4%** | ≈ **63.5%** | — |
| Quantum contribution | ≈ **41.7%** | — | — |

Hybrid beats Classical on ≈ **83.3%** of trials and stays near Dijkstra on ≈ **83.3%**. Checkpoints: `film_hybrid.pt`, `film_classical.pt`.

### Quantum Contribution (≈41.7%)

Live metric from `HybridFiLMNetwork.combine` (`Linear(10→5)`):

```
W = model.combine.weight          # shape (5, 10)
c_mag = mean(|W[:, 0:5]|)         # classical FiLM columns
q_mag = mean(|W[:, 5:10]|)        # PennyLane quantum columns
Quantum Contribution % = 100 × q_mag / (c_mag + q_mag)
```

Implemented in `src/quantum_hybrid.py` → `estimate_quantum_contribution_pct`.

### Latency note

On Calculate, the UI times Hybrid / Classical / Dijkstra rollouts (ms). **Hybrid is slower on classical simulators** (`PennyLane default.qubit`). Roadmap: a **real QPU** accelerates complex operators; Classical FiLM remains the production fallback.

### Quantum Advantage stress scenarios

Hard start / epicenter / exit pairs live in `data/demo_scenarios.json`. Regenerate:

```bash
python -u scripts/find_advantage_scenarios.py 60 5 42
```

Streamlit sidebar: **Load Quantum Advantage scenario** → auto-runs 3-way compare.

---

## Key features

- **Hybrid QML hero** — PennyLane PHN FiLM; **cyan** path on the map
- **Classical FiLM ablation** — **gold** overlay (same FiLM, no quantum branch)
- **Dijkstra baseline** — **white dashed** overlay with full Algorithm 1 dynamic weights
- **3-way metrics** — travel time, exit reached, path overlap, quantum contribution, latency (ms)
- **Quantum Advantage scenarios** — curated hard cases in the Streamlit sidebar
- **Dynamic hazards** — expanding \(r_{epi}\) / \(r_{exit}\) rings scrubbed by simulation time `t`
- **B2C Emergency Escape** — Folium map-click Start / Epicenter / Exit
- **Command Center (God View)** — Dijkstra bulk + ≤4 Hybrid heroes; scaled-citizens narrative
- **B2B API** — FastAPI `/api/v1/calculate_route` with optional Classical / Dijkstra fields
- **Offline-ready** — cached GraphML, dataset, and trained checkpoints shipped in-repo

---

## Architecture

```mermaid
flowchart LR
  UX[Streamlit B2C + God View] --> RS[routing_service]
  API[FastAPI B2B API] --> RS
  RS --> G[OSMnx / NetworkX graph]
  RS --> Dyn[Dynamic weights]
  RS --> HQ[Hybrid FiLM PHN]
  HQ --> PL[PennyLane]
  RS --> CF[Classical FiLM]
  RS --> DJ[Dijkstra oracle]
```

| Paper (Furubira) | QuantumRelief (Manila) |
| --- | --- |
| OSMnx city graph | Intramuros bbox, degree-capped, cached GraphML |
| 3 exits + random epicenter | Perimeter exits + map-click epicenter |
| Algorithm 1 dynamic weights | `src/dynamic_simulation.py` |
| Table I input size 36 | Same layout, local km projection |
| Classical + Quantum FiLM PHN | Classical ablation + **Hybrid QML hero** + Dijkstra baseline |

Radii: \(r_{epi} = 0.5 + \sqrt{0.0002\, t}\), \(r_{exit} = \sqrt{0.00075\, t}\). Hard training uses `hazard_intensity=2.0`.

Neighbor logits are masked to real degree; near-ties break toward the live Dijkstra next hop; a light Dijkstra assist may finish a stalled path — branding remains **Hybrid QML**. Travel times are honest path sums (never forged).

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
```

```bash
curl -s http://127.0.0.1:8000/

curl -s -X POST http://127.0.0.1:8000/api/v1/calculate_route \
  -H "Content-Type: application/json" \
  -d '{
    "start_coords": [14.5895, 120.9750],
    "epicenter_coords": [14.5850, 120.9780],
    "exit_coords": [14.5920, 120.9720],
    "include_comparison": true
  }'
```

OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## How to use — B2C Emergency Escape

1. Top radio: **B2C Emergency Escape**
2. Sidebar: set click mode **Start → Epicenter → Exit** *(or load a Quantum Advantage scenario)*
3. **Click the Folium map** (Start/Exit snap to nearest road node)
4. Keep comparison overlays ON (**cyan** Hybrid · **gold** Classical · **dashed** Dijkstra)
5. Press **Calculate Escape Route**
6. Scrub simulation time **`t`** — red \(r_{epi}\) / gold \(r_{exit}\) expand
7. Read the **3-way dashboard**: travel times, quantum contribution, latency (ms)

In-app: expander **How to use QuantumRelief**, **What is Quantum Contribution?**

---

## How to use — Command Center (God View)

1. Switch the top radio to **Command Center (God View)**
2. Sidebar: set **Flood / sector hazard**, optionally **Block Main Highway Bridge**
3. Keep batch at **8–10** (max 20). Hybrid QML runs only on ≤**4** hero agents; the rest is Dijkstra bulk
4. Optionally sync the **B2C epicenter**, or enter lat/lon manually
5. Click **Trigger City-Wide Evacuation Simulation** (does **not** auto-run on tab open)
6. Read metrics: **Simulated agents**, **Scaled citizens (narrative)**, escape success %, quantum contribution, congestion alert, batch latency
7. Map: **cyan** = Hybrid hero arterials · **light** = Dijkstra bulk · **red** = danger / blocked bridge

**Honest architecture:** Scaled citizens = `batch × 1,428` for pitch narrative. Only the small Hybrid sample runs QML; bulk fleet routing is Dijkstra on the hazard-weighted graph.

---

## Project structure

```
QuantumRelief/
  runtime.txt              # Streamlit Cloud: python-3.11
  requirements.txt         # Cloud / Streamlit (numpy → torch → pennylane)
  requirements-api.txt     # FastAPI + uvicorn
  app.py                   # B2C Emergency Escape + Command Center (God View)
  api.py                   # B2B Quantum Routing API
  data/                    # GraphML + routing_dataset.npz + retrain_report.json
                           # + demo_scenarios.json + hard_seeds.json
  models/                  # film_classical.pt, film_hybrid.pt
  src/
    graph_setup.py         # OSMnx / NetworkX / exits
    dynamic_simulation.py  # Algorithm 1 weights
    dataset_generation.py  # Table I vectors + Dijkstra labels
    film_model.py          # Classical FiLM
    quantum_hybrid.py      # PennyLane Hybrid PHN (+ quantum contribution %)
    routing_service.py     # Shared Hybrid + Classical + Dijkstra (API + app)
    god_view.py            # God View (Dijkstra bulk + Hybrid hero sample)
  scripts/
    retrain_models.py
    find_advantage_scenarios.py
    generate_pitch_deck.py
```

---

## Models & data

| Asset | Role |
| --- | --- |
| `models/film_hybrid.pt` | Hybrid QML FiLM (PennyLane PHN) — demo hero |
| `models/film_classical.pt` | Classical FiLM ablation |
| `data/manila_intramuros_graph.graphml` | Cached Intramuros road graph |
| `data/routing_dataset.npz` | Training / eval samples (~18.9k hard) |
| `data/retrain_report.json` | Val acc + 3-way route smoke metrics |
| `data/demo_scenarios.json` | Curated Quantum Advantage stress scenarios |
| `data/hard_seeds.json` | Classical-failure seeds for hard oversample |

**Hard retrain** (recommended; CPU-bound Hybrid; periodic checkpoints mid-run):

```bash
source .venv/bin/activate
# Full hard train (defaults: 1000 episodes, hazard×2, oversample 8×)
caffeinate -dimsu python -u scripts/retrain_models.py --hard --episodes 1000 \
  --classical-epochs 100 --hybrid-a 10 --hybrid-b 6 --hybrid-max-samples 3500
# Dataset only:
python -u scripts/retrain_models.py --hard --episodes 1000 --dataset-only
# Or via module:
python -u -m src.dataset_generation --episodes 1000 --hard
```

Hard mode widens earthquake/traffic radii (`hazard_intensity=2.0`) and oversamples seeds from `data/hard_seeds.json` (synced from `demo_scenarios.json` / `find_advantage_scenarios.py`) 8× mixed with random episodes.

**Smoke checks:**

```bash
python -c "from src.quantum_hybrid import quantum_status, load_hybrid_model; print(quantum_status()); load_hybrid_model()"
python -c "from src.graph_setup import load_or_build_graph; print(load_or_build_graph().number_of_nodes())"
```

---

## Deploy (Streamlit Community Cloud)

1. Push to GitHub (`meolen07/QuantumRelief`), including updated `models/*.pt` and `data/retrain_report.json`
2. [share.streamlit.io](https://share.streamlit.io) → select repo → **reboot the app** after model / dataset uploads so `@st.cache_resource` reloads checkpoints
3. Confirm logs: Python **3.11** (`runtime.txt`), `numpy` before `torch`, PennyLane import OK

Cloud pins live in **`requirements.txt`**. API deps stay in **`requirements-api.txt`** so Cloud stays lean.

Keep `numpy==1.26.4` before `torch==2.2.2` for Cloud ABI safety. If PennyLane install times out, Classical FiLM still runs; Hybrid shows unavailable.

---

## Team

**Quantrio** (Team 5) · QC4SG — SEA Quantathon  
Manila Intramuros emergency routing with Hybrid QML.

---

## Citation

Haboury et al., *A Hybrid Quantum-Classical Neural Network for Disaster Response*, [arXiv:2307.15682](https://arxiv.org/abs/2307.15682). QuantumRelief adapts the Furubira FiLM / PHN pipeline to Manila Intramuros.
