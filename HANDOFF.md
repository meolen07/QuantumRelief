# QuantumRelief — Hackathon HANDOFF

**Team Quantrio** · QC4SG — SEA Hackathon  
Repo: https://github.com/meolen07/QuantumRelief  
Cloud: https://quantumrelief.streamlit.app

**Tagline:** Hybrid delivers near-Dijkstra quality with quantum-classical local inference.

**Demo surface:** Escape-only **B2G2C** (Folium 2D). God View is **not** in the UI (`src/god_view.py` kept unused).

Use this if `git push` fails (auth). Upload the files below via GitHub web UI or a machine with credentials.

## Demo commands

```bash
# Streamlit Escape UX (Hybrid QML hero + 3-way compare)
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
| `models/film_hybrid.pt` | **Trained** PHN (`demo_mode=False`, val_acc≈**0.922**, q≈**37.9%**) |
| `models/film_classical.pt` | Trained Classical FiLM (val_acc≈**0.875**, 120 epochs) |
| Dataset | 500 episodes · **9536** samples |
| 3-way route smoke (24 trials) | Exit **100%** all engines · assist **0%** |
| Mean travel time | Hybrid **9.78** · Classical **10.85** · Dijkstra **9.98** |
| Hybrid beats Classical | **91.7%** of trials |
| Hybrid near Dijkstra | **95.8%** of trials |
| Path overlap vs Dijkstra | Hybrid **71.4%** · Classical **64.3%** |
| Crisis UX | Cyan Hybrid · Gold Classical · Dashed Dijkstra |
| UI | Escape-only B2G2C · no God View · no apartment presets |
| API `POST /api/v1/calculate_route` | Hybrid + optional `classical` / `dijkstra` |
| `runtime.txt` | `python-3.11` |

Sample 3-way travel times (first 3 smoke trials):

| Trial | Hybrid | Classical | Dijkstra |
| --- | ---: | ---: | ---: |
| 0 | 18.0 | 18.0 | 20.5 |
| 1 | 7.3 | 7.3 | 7.3 |
| 2 | 16.0 | 16.0 | 17.7 |

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

1. Open Streamlit → **Escape** (only surface)
2. **Click map** (or enter lat/lon) for your location → **Random epicenter**
3. Show **Best exit** one-liner (auto-recommended)
4. Press **Find route** → cyan Hybrid · gold Classical · white dashed Dijkstra
5. Scrub `t`; read Hybrid / Classical / Dijkstra travel times + Quantum Contribution
6. Story: Hybrid beats Classical; Hybrid approaches Dijkstra with local inference only

## Retrain (optional)

```bash
python -u scripts/retrain_models.py 500 120 12 8 3500
# episodes classical_epochs hybrid_A hybrid_B hybrid_max_samples
```

Periodic Hybrid checkpoints are written during Phase A/B so long trains survive interrupts.
