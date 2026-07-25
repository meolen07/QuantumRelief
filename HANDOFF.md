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
| `models/film_hybrid.pt` | **Promoted** hardft (`λ_safe=0.35`, val_acc≈**0.883**, q≈**77.6%**) |
| `models/film_classical.pt` | Trained Classical FiLM (val_acc≈**0.886**) |
| Dataset | 1000 episodes · **18932** hard samples |
| 3-way route smoke (28 trials) | Exit **100%** all engines |
| Mean travel time | Hybrid **13.65** · Classical **14.40** · Dijkstra **12.29** |
| Hybrid beats Classical (travel) | **78.6%** of trials · mean travel ≤ Classical |
| Mean safety (min-epi) | Hybrid **≈0.49** · Classical **≈0.45** · Dijkstra **≈0.56** |
| Safety score | Path rollout: `min_epi_km − 0.15·mean(log1p(w))` (UI + report; not mean-epi) |
| Path overlap vs Dijkstra | Hybrid **59.0%** · Classical **49.3%** |
| Crisis UX | Cyan Hybrid · Gold Classical · Dashed Dijkstra |
| UI | Escape-only B2G2C · travel + **safety** metrics · no God View |
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
2. **Click map** for your location → **Random epicenter**
3. Show **Best exit** one-liner (auto-recommended)
4. Press **Find route** → cyan Hybrid · gold Classical · white dashed Dijkstra
5. Scrub `t`; read Hybrid / Classical / Dijkstra **travel** + **safety** + Quantum Contribution
6. Story: Hybrid beats Classical on travel (or travel-tie + safer); Hybrid approaches Dijkstra with local inference only

## Retrain (optional)

```bash
python -u scripts/retrain_models.py 500 120 12 8 3500
# episodes classical_epochs hybrid_A hybrid_B hybrid_max_samples
```

Periodic Hybrid checkpoints are written during Phase A/B so long trains survive interrupts.
