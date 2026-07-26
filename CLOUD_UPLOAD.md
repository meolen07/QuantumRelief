# Streamlit Cloud upload — Earthquake Escape product build

Exact file list and verify checklist for QuantumRelief on Streamlit Community Cloud after the **Earthquake Escape** product pass (location → epicenter / hazard `t` → best evacuate exit → Hybrid · Classical · Dijkstra).

## Must upload / sync

```
app.py
api.py
runtime.txt
requirements.txt
requirements-api.txt
requirements-optional.txt
README.md
HANDOFF.md
CLOUD_UPLOAD.md
LICENSE
.gitignore

src/__init__.py
src/utils.py
src/graph_setup.py
src/dynamic_simulation.py
src/dataset_generation.py
src/film_model.py
src/quantum_hybrid.py
src/routing_service.py
src/traffic_provider.py
src/mock_traffic_feed.py
src/safety_loss.py
src/god_view.py              # unused by app; keep if repo already ships it

models/film_classical.pt
models/film_hybrid.pt

data/manila_intramuros_graph.graphml
data/manila_intramuros_meta.json
data/routing_dataset.npz
data/retrain_report.json
data/demo_scenarios.json
```

Optional (pitch / docs only — not required for Cloud app boot):

```
scripts/generate_pitch_deck.py
scripts/retrain_models.py
QuantumRelief_Quantrio_Pitch.pptx
docs/QuantumRelief_pitch.pptx
```

## Do NOT upload

```
.venv/
__pycache__/
cache/
*.pyc
.DS_Store
*.log
*.pid
models/*_partial.pt
models/film_hybrid_*.pt      # search / bak / smoke / hardft copies
data/*_log.txt
data/find_advantage_live*
data/proof_demo_live.json
paper_extract.txt
2307.15682.pdf               # large; linked from README
```

## Cloud settings

| Setting | Value |
| --- | --- |
| Python | `runtime.txt` → `python-3.11` |
| Main file | `app.py` |
| Requirements | `requirements.txt` (numpy **before** torch) |
| Traffic mode | default `QR_TRAFFIC_MODE=demo` — **no** TomTom/HERE keys |
| Secrets | none required for mock feed |

## After upload — reboot & verify

1. **Reboot** the Streamlit Cloud app (Manage app → Reboot).
2. Open the app — brand **Earthquake Escape**; badge: **Live conditions · simulated feed**.
3. **Conditions now** shows a quake-forward scenario (often Earthquake Escape · plaza / Pasig) + `as_of` + Manila time-of-day + post-quake / hazard incident labels.
4. **Your escape** — map click sets **Your location** (apartment/start); **Best evacuate exit** is recommended; optional override among exits. Epicenter + **Hazard time t** are primary (visible).
5. Map shows **red hazard rings**; gold flag = evacuate exit (not arbitrary A→B as the main story).
6. **Find safest & fastest escape route** — Hybrid / Classical / Dijkstra draw; **HERO** only on true Hybrid win.
7. If Hybrid travel > 1.25× Classical (or Hybrid fails / very slow): UI shows **Hybrid deferred · showing Classical**, Classical primary, Hybrid path faded, **no HERO**.
8. Collapsed **Run judge demo** / arbitrary destination / Ondoy-like flood remain secondary.
9. Confirm no SessionInfo / black-map crash on map click + scrubber (stable `st_folium` key; no `feature_group_to_add`).

## Quick local smoke (before upload)

```bash
source .venv/bin/activate
python -m py_compile app.py api.py src/mock_traffic_feed.py src/graph_setup.py src/routing_service.py src/traffic_provider.py
python -c "
from src.graph_setup import load_or_build_graph, snap_to_nearest_node
from src.mock_traffic_feed import get_mock_traffic_feed, SCENARIO_CATALOG, TIME_OF_DAY_POOLS
G = load_or_build_graph()
assert any(s['id']=='quake_core' for s in SCENARIO_CATALOG)
assert TIME_OF_DAY_POOLS['rush'][0] == 'quake_core'
feed = get_mock_traffic_feed(force_reload=True)
feed.force_scenario('quake_core')
snap = feed.current(G)
assert snap.has_disaster and snap.epicenter_lonlat
print('ok', snap.scenario_name, snap.epicenter_lonlat)
"
streamlit run app.py
```
