# QuantumRelief

**Quantum Intelligence. Human Relief.**

Team 5 — **Quantrio** · QC4SG SEA Quantathon 2026

[Streamlit](https://quantumrelief.streamlit.app)
[Python](runtime.txt)
[PennyLane](https://pennylane.ai)
[License](LICENSE)

Live demo: **[quantumrelief.streamlit.app](https://quantumrelief.streamlit.app)**

**Tagline:** The map is always dynamic — Hybrid QML routes under changing edge costs.

---

## Overview

Cities do not wait for a disaster to break routing. **Traffic, closed streets, and congestion** rewrite edge costs every hour. Static shortest path assumes a frozen graph; QuantumRelief treats the map as **live dynamics**.

**Earthquake Escape** (Manila Intramuros) is the **Quantathon extreme stress demo** inside **Live Escape** — optional hazard rings on top of everyday road disruptions. The broader product is the same engine applied to daily dynamics: **B2B API + simulation now**, **B2G2C Live Escape** as civic proof for later.

A **Hybrid Quantum–Classical FiLM** model (PennyLane PHN) is the hero path; Classical FiLM is an ablation; Dijkstra is the full-information optimal baseline under Algorithm 1 dynamic weights.

Adapted from Haboury et al., *[Quantum Machine Learning for Disaster Response](https://arxiv.org/abs/2307.15682)* (Furubira → Manila).

**Honest scope today:** Manila OSM + **simulated** dynamics (quake / exit-traffic rings). Real traffic feeds (TomTom / HERE) are a roadmap plug-in — not integrated in this demo.

**One surface**


| Surface         | Audience                              | Role                                                                                                      |
| --------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Live Escape** | Citizens + gov demos (B2G2C flagship) | Your location → change the map (disruption and/or epi) → safest & fastest · Hybrid · Classical · Dijkstra |


`src/god_view.py` remains in-repo for optional command-center experiments but is **not** wired into the Streamlit UI.

**UI palette:** deep navy · **cyan** Hybrid · **gold** Classical · orange accents · **red** hazard · white/light dashed Dijkstra.

---

## Problem & solution


| Beat              | Story                                                                                               |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| **Problem**       | Static Dijkstra / A* corridors fail when edge costs move — jams, closures, cascading congestion.    |
| **Extreme case**  | Earthquake + exit surge = the hardest dynamic-hazard regime (Escape demo).                          |
| **Everyday case** | Same weight-update idea for traffic / closed streets / fleet shocks.                                |
| **Solution**      | Dynamic edge weights + **Hybrid FiLM∥PHN** local next-hop vs Classical ablation vs Dijkstra oracle. |
| **Business**      | Sell **B2B routing API / sim** now; grow **B2G2C Escape** trust over time.                          |


---

## Results (latest hard retrain)

From `data/retrain_report.json` — balance hard fine-tune (`λ_safe=0.32`, hard_frac=0.45, best **F16**) from `film_hybrid_pre_balance.pt`; fair eval **32** trials (hard_seeds + random).


| Metric                      | Hybrid      | Classical   | Dijkstra    |
| --------------------------- | ----------- | ----------- | ----------- |
| Val accuracy                | ≈ **0.920** | ≈ **0.886** | —           |
| Mean travel time            | ≈ **12.83** | ≈ **13.89** | ≈ **12.19** |
| Mean safety score (min-epi) | ≈ **0.51**  | ≈ **0.46**  | ≈ **0.56**  |
| Exit reached                | **100%**    | **100%**    | **100%**    |
| Path overlap vs Dijkstra    | ≈ **55.6%** | ≈ **51.0%** | —           |
| Quantum contribution        | ≈ **77.6%** | —           | —           |


Hybrid **mean travel ≤ Classical** (Δ ≈ −1.06) with **~75%** travel wins and **~81%** near Dijkstra. Mean **min-epi** safety also edges Classical (Δ ≈ +0.045). Catastrophic blowups (H travel > 1.25× C) ≈ **9.4%** (down from ~10.7%). **Promote: YES** — `film_hybrid_hardft.pt` → serving `models/film_hybrid.pt`. UI **HERO** badge only when Hybrid strictly beats Classical on travel, or travel-tie + higher safety.

**Safety score** (path rollout, **min-epi based**, higher = safer; UI-scale ~0.05–2.0):

```
safety_score = min_epi_km − 0.15 · mean(log1p(w_edge))
```

- `min_epi_km` — closest km approach of any path node to the epicenter (**primary**; matches map rings)  
- `mean_epi_km` — also reported; mean alone can mis-rank paths that dive near epi then run a long far tail  
- `w_edge` — Algorithm-1 travel weight on each hop (light secondary hazard penalty)  
- Training uses a related idea via `λ_safe · L_safe` in `src/safety_loss.py` (soft preference for safer next hops; Dijkstra CE stays primary)

**Promotion rule:** copy candidate → `film_hybrid.pt` only when mean travel ≤ Classical (or travel within 2% + higher safety), mean safety ≥ Classical, and catastrophic rate (H>1.25×C) does not worsen.

### Quantum Contribution (≈77.6%)

Live metric from `HybridFiLMNetwork.combine` (`Linear(10→5)`):

```
W = model.combine.weight          # shape (5, 10)
c_mag = mean(|W[:, 0:5]|)         # classical FiLM columns
q_mag = mean(|W[:, 5:10]|)        # PennyLane quantum columns
Quantum Contribution % = 100 × q_mag / (c_mag + q_mag)
```

Implemented in `src/quantum_hybrid.py` → `estimate_quantum_contribution_pct`.

### Latency note

On **Find safest & fastest route**, the UI times Hybrid / Classical / Dijkstra rollouts (ms). **Hybrid is slower on classical simulators** (`PennyLane default.qubit`). Roadmap: a **real QPU** accelerates complex operators; Classical FiLM remains the production fallback.

### Quantum Advantage stress scenarios

Hard start / epicenter / exit pairs live in `data/demo_scenarios.json`. Regenerate:

```bash
python -u scripts/find_advantage_scenarios.py 60 5 42
```

---

## Key features

- **Hybrid QML hero** — PennyLane PHN FiLM; **cyan** path on the map
- **Classical FiLM ablation** — **gold** overlay (same FiLM, no quantum branch)
- **Dijkstra baseline** — **white dashed** overlay with full Algorithm 1 dynamic weights
- **3-way metrics** — travel time, safety, exit reached, path overlap, quantum contribution, latency (ms)
- **Auto-best exit** — silent ranking; one recommended-exit line in the panel
- **Location** — Folium map click, snapped to nearest graph node (no address field)
- **Road disruptions (first-class)** — Congestion (×5) / Closed corridor soft (×8); amber dashed overlay + edge count
- **Epicenter** — optional extreme hazard (**Random epicenter**); expanding red rings
- **Dynamic hazards** — expanding r_{epi} / r_{exit} rings scrubbed by simulation time `t`
- **Live Escape UI** — Folium 2D · left ~2/3 map · right ~1/3 scrollable panel (Quantathon flagship)
- **B2B API** — FastAPI `/api/v1/calculate_route` with optional Classical / Dijkstra fields
- **Offline-ready** — cached GraphML, dataset, and trained checkpoints shipped in-repo

---

## Architecture

```mermaid
flowchart LR
  UX[Streamlit Live Escape] --> RS[routing_service]
  API[FastAPI B2B API] --> RS
  RS --> H[Hybrid QML FiLM]
  RS --> C[Classical FiLM]
  RS --> D[Dijkstra oracle]
  H --> PL[PennyLane PHN]
  RS --> G[Intramuros GraphML]
  RS --> Dyn[Algorithm 1 dynamic weights]
```



---

## Quick start

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Optional API:

```bash
pip install -r requirements-api.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

```bash
curl -s http://127.0.0.1:8000/api/v1/calculate_route \
  -H "Content-Type: application/json" \
  -d '{
    "start_coords": [14.5895, 120.9750],
    "epicenter_coords": [14.5850, 120.9780],
    "exit_coords": [14.5920, 120.9720],
    "include_comparison": true
  }'
```

OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Graph, dataset, and checkpoints under `data/` and `models/` are included. OSM download runs only if the GraphML cache is missing.

---

## How to use — Live Escape (flagship demo)

**Framing:** the map is always dynamic. Earthquake is the optional extreme stress case — not the only story.

1. **Click the map** to set your location (apartment / trip start) — snapped to the nearest road node
2. **Change the map** (encourage at least one):
   - **Congestion** or **Closed corridor (soft)** — amber dashed soft costs; edge count shown
   - and/or **Random epicenter** — extreme hazard rings
3. Read the **Best exit** line (auto-recommended)
4. Press **Find safest & fastest route** — **cyan** Hybrid · **gold** Classical · **white dashed** Dijkstra
5. Scrub hazard time `**t`** when an epicenter is active — red r_{epi} / gold r_{exit} expand
6. Read right-panel **3-way metrics**: travel, safety, quantum contribution, latency — **HERO** only when Hybrid wins

Layout: left **~2/3** Folium map (fixed) · right **~1/3** scrollable controls + metrics.

On first load, a curated Hybrid-advantage scenario auto-runs when present (disruption nudge only). If none, a soft congestion corridor is seeded so amber corridors are visible without hunting.

---

## Project structure

```
QuantumRelief/
  runtime.txt              # Streamlit Cloud: python-3.11
  requirements.txt         # Cloud / Streamlit (numpy → torch → pennylane)
  requirements-api.txt     # FastAPI + uvicorn
  app.py                   # Live Escape flagship (Folium 2D; Escape-only)
  api.py                   # B2B Quantum Routing API
  data/                    # GraphML + routing_dataset.npz + retrain_report.json
                           # + demo_scenarios.json + hard_seeds.json
  models/                  # film_classical.pt, film_hybrid.pt
  src/
    graph_setup.py         # OSMnx / NetworkX / exits
    dynamic_simulation.py  # Algorithm 1 weights (disaster = extreme dynamics)
    dataset_generation.py  # Table I vectors + Dijkstra labels
    film_model.py          # Classical FiLM
    safety_loss.py         # Safety aux loss (λ_safe · L_safe)
    quantum_hybrid.py      # PennyLane Hybrid PHN (+ quantum contribution %)
    routing_service.py     # Shared Hybrid + Classical + Dijkstra (API + app)
    god_view.py            # Unused by app (optional command-center experiments)
  scripts/
    retrain_models.py
    find_advantage_scenarios.py
    generate_pitch_deck.py
```

---

## Models & data


| Asset                                  | Role                                        |
| -------------------------------------- | ------------------------------------------- |
| `models/film_hybrid.pt`                | Hybrid QML FiLM (PennyLane PHN) — demo hero |
| `models/film_classical.pt`             | Classical FiLM ablation                     |
| `data/manila_intramuros_graph.graphml` | Cached Intramuros road graph                |
| `data/routing_dataset.npz`             | Training / eval samples (~18.9k hard)       |
| `data/retrain_report.json`             | Val acc + 3-way route smoke metrics         |
| `data/demo_scenarios.json`             | Curated Quantum Advantage stress scenarios  |
| `data/hard_seeds.json`                 | Classical-failure seeds for hard oversample |


**Hard retrain** (recommended; CPU-bound Hybrid; periodic checkpoints mid-run):

```bash
source .venv/bin/activate
# Hybrid push on existing hard dataset (~18.9k):
caffeinate -dimsu python -u scripts/retrain_models.py --hard --reuse-dataset \
  --skip-classical --hybrid-a 20 --hybrid-b 8 --hybrid-max-samples 6500 \
  --hard-repeats 6 --eval-trials 28 --lambda-safe 0.35
# Full regen + Classical + Hybrid:
caffeinate -dimsu python -u scripts/retrain_models.py --hard --episodes 1000 \
  --classical-epochs 100 --hybrid-a 20 --hybrid-b 8 --hybrid-max-samples 6500 \
  --lambda-safe 0.35
# Dataset only:
python -u scripts/retrain_models.py --hard --episodes 1000 --dataset-only
# Or via module:
python -u -m src.dataset_generation --episodes 1000 --hard
```

Hard mode widens earthquake/traffic radii (`hazard_intensity=2.0`) and oversamples seeds from `data/hard_seeds.json` (synced from `demo_scenarios.json` / `find_advantage_scenarios.py`) 8× mixed with random episodes. Hybrid subset additionally oversamples hard-seed trajectories at intensity 1.0 + 2.0.

**Smoke checks:**

```bash
python -c "from src.quantum_hybrid import quantum_status, load_hybrid_model; print(quantum_status()); load_hybrid_model()"
python -c "from src.graph_setup import load_or_build_graph; print(load_or_build_graph().number_of_nodes())"
```

---

## Business & roadmap


| Horizon   | Focus                                                                                                                            |
| --------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Now**   | Live Escape + Hybrid / Classical / Dijkstra · B2B `/api/v1/calculate_route` · simulated dynamics (disruption + optional quake) |
| **Next**  | Plug-in live traffic / closure weights (TomTom / HERE) · multi-district graphs · fleet pilots                                    |
| **Later** | Real QPU offload · offline edge · SEA city transfer · fuller B2G2C Live Escape product                                           |


---

## Deploy (Streamlit Community Cloud)

1. Push to GitHub (`meolen07/QuantumRelief`), including updated `models/*.pt` and `data/retrain_report.json`
2. [share.streamlit.io](https://share.streamlit.io) → select repo → **reboot the app** after model / dataset uploads so `@st.cache_resource` reloads checkpoints
3. Confirm logs: Python **3.11** (`runtime.txt`), `numpy` before `torch`, PennyLane import OK

Cloud pins live in `**requirements.txt`**. API deps stay in `**requirements-api.txt**` so Cloud stays lean.

Keep `numpy==1.26.4` before `torch==2.2.2` for Cloud ABI safety. If PennyLane install times out, Classical FiLM still runs; Hybrid shows unavailable.

---

## Team

**Quantrio** (Team 5) · QC4SG — SEA Quantathon 2026  
Dynamic-edge routing with Hybrid QML — Live Escape on Manila Intramuros.

---

## Citation

Haboury et al., *A Hybrid Quantum-Classical Neural Network for Disaster Response*, [arXiv:2307.15682](https://arxiv.org/abs/2307.15682). QuantumRelief adapts the Furubira FiLM / PHN pipeline to Manila Intramuros; Escape is the extreme dynamic case of a broader changing-map product.