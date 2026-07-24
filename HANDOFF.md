# QuantumRelief — Hackathon HANDOFF

**Team Quantrio** · QC4SG — SEA Hackathon  
Repo: https://github.com/meolen07/QuantumRelief  
Cloud: https://quantumrelief.streamlit.app

Use this if `git push` fails (auth). Upload the files below via GitHub web UI or a machine with credentials.

## Demo commands

```bash
# Streamlit Crisis UX (Hybrid QML hero)
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

# B2B API
pip install -r requirements-api.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Status at handoff

| Item | Status |
| --- | --- |
| `models/film_hybrid.pt` | **Trained** PHN (`demo_mode=False`, val_acc≈0.92, q≈34%) |
| `models/film_classical.pt` | Trained Classical FiLM |
| Hybrid route smoke | 3/3 EXIT REACHED (no Dijkstra assist) |
| API `GET /` | OK |
| API `POST /api/v1/calculate_route` | OK — Hybrid path + exit_reached |
| `runtime.txt` | `python-3.11` |
| `.gitignore` | `.venv`, `__pycache__`, `cache/` |

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
2307.15682.pdf   # optional (large); paper is linked in README
```

## Judge demo script (60s)

1. Open Streamlit → expand **How to use QuantumRelief** if needed  
2. Mode **Start** → click map → **Epicenter** → click → **Exit** → click  
3. Confirm **Hybrid QML (PennyLane)** selected  
4. **Calculate Escape Route** → green Hybrid reaches exit; dashed Dijkstra comparison  
5. Scrub `t` for hazard rings; note Exit Reached + Quantum Contribution  

## Retrain (optional)

```bash
python -u scripts/retrain_models.py 400 100 10 6
```

Shorter Hybrid finish used for this handoff: 2000 samples, Phase A=5 / B=3 → `models/film_hybrid.pt`.
