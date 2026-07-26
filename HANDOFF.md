# QuantumRelief — Hackathon HANDOFF

**Team Quantrio** · QC4SG — SEA Hackathon  
Repo: https://github.com/meolen07/QuantumRelief  
Cloud: https://quantumrelief.streamlit.app

**Tagline:** Earthquake Escape — safest & fastest evacuate exit under expanding hazard.  
**Judge / demo defense Q&A:** [`docs/TECHNICAL_QA.md`](docs/TECHNICAL_QA.md)

**Thesis (approved):** Hybrid QML finds strong escape routes when the map moves — epicenter rings expand with `t`, exits compete, post-quake damage rewrites edge costs. **Earthquake Escape** is the Quantathon B2G2C flagship; Ondoy-like flood is a related dynamic-hazard case study. **Architecture:** Demo = same Escape app + `MockTrafficProvider`; Production = same Escape app + `LiveTrafficProvider` (TomTom/HERE stub). Default `QR_TRAFFIC_MODE=demo` (no traffic keys).


**Demo surface:** **Earthquake Escape** Folium 2D only. **3-step audience flow:** location (optional) → **Run demo** (pinned `qa_1`) → map + metrics. Multi-exit markers stay on the map; ranked list / congestion / place mode live under **Advanced**.

Use this if `git push` fails (auth). Upload the files below via GitHub web UI or a machine with credentials.

## Elevator story (memorize)

Apartment in Manila shakes. You may know a few evacuate areas — not which is safest and fastest. QuantumRelief ranks **several exits**, recommends the **best**, and routes you there with a Hybrid FiLM∥PHN hero, Classical ablation, and Dijkstra oracle — under expanding hazard rings and post-quake road damage. **Earthquake Escape** on Manila Intramuros is the Quantathon flagship. **Same Escape surface for demo and production** — only the traffic provider changes (`MockTrafficProvider` vs live feed stub).


## Demo commands

```bash
# Streamlit Earthquake Escape UX (Hybrid QML hero + 3-way compare)
# Default: Demo · mock post-quake / hazard feed (no traffic keys)
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

# Optional: live stub (needs TRAFFIC_API_KEY or shows configure warning)
# QR_TRAFFIC_MODE=live streamlit run app.py
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
| Pitch deck | `scripts/generate_pitch_deck.py` → 20 slides: **Earthquake Escape** flagship + **Ondoy (Ketsana) 2009** related case + Intramuros Escape corridor (`qa_2`) |
| UI | **Earthquake Escape** · 3 steps (location → **Run demo** → metrics) · Advanced collapsed · Hybrid deferred fallback · HERO only on win · no address field |
| `runtime.txt` | `python-3.11` |

Sample 3-way travel times (first 3 eval trials):

| Trial | Hybrid | Classical | Dijkstra |
| --- | ---: | ---: | ---: |
| 0 | 7.3 | 7.3 | 7.3 |
| 1 | 15.5 | 15.5 | 16.3 |
| 2 | 9.5 | 17.3 | 12.2 |

## Files to upload / sync to GitHub

See **[`CLOUD_UPLOAD.md`](CLOUD_UPLOAD.md)** for the exact Earthquake Escape product file list + Streamlit Cloud reboot/verify checklist.

**Must upload (completion work):**

```
README.md
HANDOFF.md
CLOUD_UPLOAD.md
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
src/traffic_provider.py
src/mock_traffic_feed.py
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
data/demo_scenarios.json
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

1. **Hook (8s):** “Apartment shakes in Manila — which escape is safest and fastest while hazard moves?”
2. Open Streamlit — app auto-arms pinned **qa_1** and runs (or press **Run demo** once).
3. **Point at the map (20s):** cyan Hybrid · gold Classical · white dashed Dijkstra · gold pins = evacuate exits.
4. **Read metrics (20s):** Hybrid travel **below** Classical · HERO when Hybrid wins · scrub **Hazard time t** if rings are on.
5. **Close (12s):** “Hazard moved the map — Hybrid finds a better escape than Classical, approaching Dijkstra with local inference only.”

Story line: **hazard moved the map → Hybrid finds a better escape** (travel beat; approaches Dijkstra with local inference only).

Manual / Advanced: map click for location · exit override · congestion presets · **Find escape route (current map)**.

## Retrain (optional)

```bash
python -u scripts/retrain_models.py 500 120 12 8 3500
# episodes classical_epochs hybrid_A hybrid_B hybrid_max_samples
```

Periodic Hybrid checkpoints are written during Phase A/B so long trains survive interrupts.

## Earthquake Escape — Hazard + post-quake damage

Epicenter / hazard rings (primary) + soft post-quake damage / blocked / flood via **traffic provider** (production-shaped):

- **Architecture one-liner:** Demo = `MockTrafficProvider`; Production = same Escape app + `LiveTrafficProvider` (TomTom/HERE stub).
- Config: `QR_TRAFFIC_MODE=demo|live` (default **`demo`**) · live needs `TRAFFIC_API_KEY` (graceful message if missing)
- UI badge: **Live conditions · simulated feed** vs **Live conditions · live feed**
- Product flow: **Step 1 Location** (optional) → **Step 2 Run demo** (pinned `qa_1`) → **Step 3 map + metrics**
- Mock catalog: Earthquake Escape (epi) leads daytime pools · post-quake damaged roads (×~5) · blocked corridors (×~8) · flooded corridor (×~12, related case)
- Disaster in feed → red rings on map · **Hazard time t** scrub (visible after arm)
- UI: multi-exit Folium markers (gold = recommended, cyan = routing) · short “Recommended exit · …” one-liner · ranked list / congestion / place mode / feed under **Advanced** · Hybrid deferred fallback · HERO only on win · primary **Run demo**
- Provider: `src/traffic_provider.py` → wraps `MockTrafficFeed`; apply via `apply_provider_disruptions` in `routing_service`
- Folium: amber dashed post-quake damage (`#F5A623`); red rings when disaster active; stable `st_folium` key `qr_map_escape`
- No address field · arbitrary A→B destination is Advanced-only
- Ranking: `rank_evacuate_areas` / `recommend_best_exit` in `routing_service` (Dijkstra travel + epi-distance safety → combined score)
