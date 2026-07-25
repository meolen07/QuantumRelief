# QuantumRelief — Hackathon HANDOFF

**Team Quantrio** · QC4SG — SEA Hackathon  
Repo: https://github.com/meolen07/QuantumRelief  
Cloud: https://quantumrelief.streamlit.app

**Tagline:** The map is always dynamic — Hybrid QML routes under changing edge costs.

**Thesis (approved):** Hybrid QML finds strong routes even without a disaster, because edge costs always move (traffic, closures, congestion). **Live Escape** is the Quantathon surface; earthquake is the **optional extreme** stress demo; daily dynamics are the broader product & money story. **B2B API / sim now · B2G2C Live Escape later.** Honest today: Manila OSM + simulated dynamics; TomTom/HERE is roadmap.

**Demo surface:** **Live Escape** Folium 2D only. God View is **not** in the UI (`src/god_view.py` kept unused). Dynamic map first; earthquake = optional extreme stress.

Use this if `git push` fails (auth). Upload the files below via GitHub web UI or a machine with credentials.

## Elevator story (memorize)

Cities rewrite the map every hour — jams, closed streets, cascading congestion — so static shortest path is already wrong. QuantumRelief treats routing as **local next-hop under live edge costs**, with a Hybrid FiLM∥PHN hero, Classical ablation, and Dijkstra oracle. **Live Escape** on Manila Intramuros is the Quantathon surface: road disruptions are first-class; expanding hazard rings are the **optional extreme** case of the same always-on problem. We ship a B2B routing API and simulation now; Live Escape builds civic trust for a longer B2G2C game. Scope is honest — OSM + simulated dynamics today; real traffic feeds plug in later.

## Demo commands

```bash
# Streamlit Live Escape UX (Hybrid QML hero + 3-way compare)
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

# B2B API (optional Classical / Dijkstra fields)
pip install -r requirements-api.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Status at handoff

| Item | Status |
| --- | --- |
| `models/film_hybrid.pt` | **Promote YES** — balance hardft F16 (`λ_safe=0.32`, hard_frac=0.45, val_acc≈**0.920**, q≈**77.6%**) |
| `models/film_classical.pt` | Trained Classical FiLM (val_acc≈**0.886**) |
| Dataset | 1000 episodes · **18932** hard samples |
| 3-way route smoke (32 trials) | Exit **100%** all engines |
| Mean travel time | Hybrid **12.83** · Classical **13.89** · Dijkstra **12.19** |
| Hybrid beats Classical (travel) | **75%** of trials · mean travel ≤ Classical (Δ ≈ −1.06) |
| Mean safety (min-epi) | Hybrid **≈0.51** · Classical **≈0.46** · Dijkstra **≈0.56** |
| Catastrophic (H>1.25×C) | **≈9.4%** (improved vs prior ~10.7%) |
| Safety score | Path rollout: `min_epi_km − 0.15·mean(log1p(w))` (UI + report; not mean-epi) |
| Path overlap vs Dijkstra | Hybrid **55.6%** · Classical **51.0%** · near-Dij **81%** |
| Crisis UX | Cyan Hybrid · Gold Classical · Dashed Dijkstra |
| Pitch deck | `scripts/generate_pitch_deck.py` → 20 slides: **Ondoy (Ketsana) 2009** real-world case (replaces fictional Ana Reyes vignette) + quantitative **Intramuros Escape Corridor** (`qa_2` H≈16.1 / C≈20.0 / D≈16.1) |
| UI | **Live Escape** · dynamic map first · **HERO** only on travel beat or travel-tie+safer · Congestion / Closed corridor (amber soft) · epi optional · no God View |
| API `POST /api/v1/calculate_route` | Hybrid + optional `classical` / `dijkstra` |
| `runtime.txt` | `python-3.11` |

Sample 3-way travel times (first 3 eval trials):

| Trial | Hybrid | Classical | Dijkstra |
| --- | ---: | ---: | ---: |
| 0 | 7.3 | 7.3 | 7.3 |
| 1 | 15.5 | 15.5 | 16.3 |
| 2 | 9.5 | 17.3 | 12.2 |

## Files to upload / sync to GitHub

**Must upload (completion work):**

```
README.md
HANDOFF.md
.gitignore
runtime.txt
requirements.txt
requirements-api.txt
requirements-optional.txt
app.py
api.py
LICENSE
QuantumRelief_Quantrio_Pitch.pptx
docs/QuantumRelief_pitch.pptx
src/__init__.py
src/utils.py
src/graph_setup.py
src/dynamic_simulation.py
src/dataset_generation.py
src/film_model.py
src/quantum_hybrid.py
src/routing_service.py
src/god_view.py
src/safety_loss.py
scripts/retrain_models.py
scripts/generate_pitch_deck.py
models/film_classical.pt
models/film_hybrid.pt
data/manila_intramuros_graph.graphml
data/manila_intramuros_meta.json
data/routing_dataset.npz
data/retrain_report.json
data/retrain_log.txt
```

**Do NOT upload:**

```
.venv/
__pycache__/
cache/
*.pyc
.DS_Store
paper_extract.txt
*.log
models/*_partial.pt
2307.15682.pdf   # optional (large); paper is linked in README
```

## Judge demo script (60s)

1. **Hook (10s):** “Static shortest path fails when the map moves — traffic every day, disaster as the extreme case.”
2. Open Streamlit → **Live Escape** — “Your trip under a changing map.”
3. **Click map** for location → **Congestion** or **Closed corridor** (amber dashed; show edge count) → optional **Random epicenter** → **Best exit** one-liner.
4. Press **Find safest & fastest route** → cyan Hybrid · gold Classical · white dashed Dijkstra.
5. Scrub `t` if epi is on; read Hybrid / Classical / Dijkstra **travel** + **safety** + Quantum Contribution — **HERO** only when Hybrid wins.
6. **Close (10s):** “Same engine for daily dynamics — simulated disruptions stand in for traffic until TomTom/HERE. B2B API now, Live Escape proves civic trust.”

Story line if Hybrid wins: beats Classical on travel (or travel-tie + safer); approaches Dijkstra with local inference only.

## Retrain (optional)

```bash
python -u scripts/retrain_models.py 500 120 12 8 3500
# episodes classical_epochs hybrid_A hybrid_B hybrid_max_samples
```

Periodic Hybrid checkpoints are written during Phase A/B so long trains survive interrupts.

## Live Escape — Road disruptions (first-class)

Soft congestion / soft-closed corridor (no live traffic API yet):

- UI: panel step **2 · Change the map** → **Congestion** (×5) / **Closed corridor** (×8 soft) / **Clear disruptions**
- Shows disrupted **edge count** + kind label; amber dashed overlay explained in legend
- Helper: `sample_random_disruptions` + `apply_edge_disruptions` in `src/dynamic_simulation.py`
- Wired through `predict_escape_route` / `dijkstra_escape_route` / `compare_three_way` / exit ranking
- Folium: amber dashed (`#F5A623`); stable `st_folium` key `qr_map_escape`
- First load: advantage scenario auto-runs when present (disruption = nudge only). Else seeds congestion so amber is visible.
- Caption honesty: stand-in until TomTom / HERE
- Epicenter remains optional extreme hazard under the same step
