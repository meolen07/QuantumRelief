"""
QuantumRelief — Crisis-Driven Streamlit dashboard (Phase 4).

B2C Emergency Escape (Folium 2D only): apartment start → hazard →
ranked evacuate areas → Hybrid QML hero with Classical + Dijkstra overlays.
Layout: left ~2/3 map, right ~1/3 controls + metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

import folium
import networkx as nx
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dynamic_simulation import damage_radius, exit_radius
from src.film_model import ensure_trained_model
from src.graph_setup import (
    load_or_build_graph,
    random_epicenter,
    select_exit_nodes,
)
from src.quantum_hybrid import (
    QUANTUM_CONTRIBUTION_FORMULA,
    estimate_quantum_contribution_pct,
    ensure_hybrid_model,
    quantum_status,
)
from src.god_view import init_god_view_state, render_god_view, render_god_view_controls
from src.routing_service import (
    compare_three_way,
    dijkstra_escape_route,
    nearest_node as _rs_nearest_node,
    path_travel_time as _rs_path_travel_time,
    predict_escape_route,
    recommend_best_exit,
    route_overlap_accuracy as _rs_route_overlap,
)
from src.utils import DATA_DIR, get_graph_origin

st.set_page_config(
    page_title="QuantumRelief",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Preset apartment / current-location pins (lat, lon) inside Intramuros.
APARTMENT_PRESETS = [
    {"id": "default", "label": "Your apartment (Intramuros)", "lat": 14.5908, "lon": 120.9752},
    {"id": "fort", "label": "Near Fort Santiago", "lat": 14.5940, "lon": 120.9708},
    {"id": "cathedral", "label": "Near Manila Cathedral", "lat": 14.5896, "lon": 120.9734},
]

# --- Crisis Core aesthetic (Lovable-aligned): deep navy + cyan Hybrid + gold Classical ---
# Map route palette (shared with Folium overlays + God View)
HYBRID_ROUTE_COLOR = "#00E5FF"
CLASSICAL_ROUTE_COLOR = "#F5C542"
DIJKSTRA_ROUTE_COLOR = "#E8EEF6"
HAZARD_ROUTE_COLOR = "#FF4D6A"
EXIT_RING_COLOR = "#F5C542"
ORANGE_ACCENT = "#FF8A4C"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    :root {
      --qr-bg: #0a0b10;
      --qr-navy: #0a0f1e;
      --qr-deep: #0e1528;
      --qr-panel: rgba(14, 22, 40, 0.78);
      --qr-cyan: #00E5FF;
      --qr-gold: #F5C542;
      --qr-orange: #FF8A4C;
      --qr-hazard: #FF4D6A;
      --qr-dij: #E8EEF6;
      --qr-mist: #9AA8BC;
      --qr-ink: #E8EEF6;
    }
    .stApp {
      background:
        radial-gradient(900px 480px at 18% -8%, rgba(0,229,255,0.10) 0%, transparent 55%),
        radial-gradient(700px 400px at 88% 12%, rgba(255,138,76,0.07) 0%, transparent 50%),
        linear-gradient(180deg, #0a0f1e 0%, #0a0b10 55%, #07080d 100%);
      color: var(--qr-ink);
      font-family: 'DM Sans', system-ui, sans-serif;
    }
    h1, h2, h3, h4 {
      font-family: 'DM Sans', system-ui, sans-serif !important;
      letter-spacing: -0.01em;
      color: #f4f7fb !important;
    }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0a0f1e 0%, #0c1220 100%);
      border-right: 1px solid rgba(0,229,255,0.12);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    .qr-header {
      display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1rem;
      margin: 0 0 0.35rem 0;
    }
    .qr-brand {
      font-family: 'DM Sans', system-ui, sans-serif;
      font-size: 2.35rem;
      font-weight: 700;
      color: #f4f7fb;
      margin: 0;
      line-height: 1.05;
      letter-spacing: -0.02em;
    }
    .qr-brand span { color: var(--qr-cyan); }
    .qr-online {
      display: inline-flex; align-items: center; gap: 0.35rem;
      font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
      text-transform: uppercase; padding: 0.32rem 0.75rem;
      border-radius: 999px;
      color: var(--qr-cyan);
      background: rgba(0,229,255,0.10);
      border: 1px solid rgba(0,229,255,0.35);
      box-shadow: 0 0 18px rgba(0,229,255,0.12);
    }
    .qr-online .dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--qr-cyan);
      box-shadow: 0 0 8px rgba(0,229,255,0.8);
    }
    .qr-tagline {
      font-family: 'DM Sans', system-ui, sans-serif;
      font-size: 1.15rem;
      font-weight: 600;
      color: #fff;
      letter-spacing: -0.01em;
      margin: 0.15rem 0 0.1rem 0;
    }
    .qr-tag {
      color: var(--qr-mist);
      font-size: 0.92rem;
      margin: 0.1rem 0 0.75rem 0;
    }
    .qr-team {
      display: inline-flex; flex-wrap: wrap; gap: 0.45rem; align-items: center;
      margin: 0 0 0.85rem 0;
    }
    .qr-team .chip {
      font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em;
      text-transform: uppercase; padding: 0.28rem 0.7rem; border-radius: 999px;
      border: 1px solid rgba(255,138,76,0.4); color: #ffb08a;
      background: rgba(255,138,76,0.1);
    }
    .qr-team .chip.soft {
      border-color: rgba(154,168,188,0.28); color: var(--qr-mist);
      background: rgba(14,22,40,0.65);
    }
    .qr-steps {
      display: flex; gap: 0.45rem; flex-wrap: wrap;
      margin-bottom: 0.85rem;
    }
    .qr-step {
      background: rgba(14,22,40,0.85);
      border: 1px solid rgba(154,168,188,0.18);
      border-radius: 999px;
      padding: 0.42rem 0.85rem;
      font-size: 0.8rem;
      color: var(--qr-mist);
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .qr-step b { color: var(--qr-cyan); margin-right: 0.35rem; }
    .qr-step.active {
      border-color: rgba(0,229,255,0.55);
      color: #fff;
      box-shadow: 0 0 0 1px rgba(0,229,255,0.25), 0 0 20px rgba(0,229,255,0.14);
      background: rgba(0,229,255,0.12);
    }
    .qr-step.done {
      border-color: rgba(0,229,255,0.35);
      color: #9eecf8;
    }
    .qr-card {
      background: linear-gradient(160deg, rgba(16,24,42,0.92), rgba(10,15,30,0.88));
      border: 1px solid rgba(154,168,188,0.16);
      border-radius: 18px;
      padding: 1rem 1.05rem;
      height: 100%;
      backdrop-filter: blur(10px);
      box-shadow: 0 8px 28px rgba(0,0,0,0.28);
      position: relative;
    }
    .qr-card.win {
      border-color: rgba(0,229,255,0.5);
      box-shadow: 0 0 0 1px rgba(0,229,255,0.18), 0 8px 32px rgba(0,229,255,0.08);
    }
    .qr-card.hybrid {
      border-color: rgba(0,229,255,0.45);
      box-shadow: 0 0 24px rgba(0,229,255,0.1);
    }
    .qr-card.classical { border-color: rgba(245,197,66,0.35); }
    .qr-card.dijkstra { border-color: rgba(232,238,246,0.22); }
    .qr-hero-pill {
      position: absolute; top: 0.75rem; right: 0.75rem;
      font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em;
      text-transform: uppercase; padding: 0.2rem 0.5rem; border-radius: 999px;
      color: #041018; background: var(--qr-cyan);
      box-shadow: 0 0 14px rgba(0,229,255,0.45);
    }
    .qr-card .label {
      color: var(--qr-mist);
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      margin-bottom: 0.35rem;
      padding-right: 3.2rem;
    }
    .qr-card .value {
      font-family: 'DM Sans', system-ui, sans-serif;
      font-size: 1.85rem;
      font-weight: 700;
      color: #fff;
      line-height: 1.1;
    }
    .qr-card .value.accent { color: var(--qr-cyan); }
    .qr-card .value.gold { color: var(--qr-gold); }
    .qr-card .value.dij { color: var(--qr-dij); }
    .qr-card .sub {
      color: var(--qr-mist);
      font-size: 0.82rem;
      margin-top: 0.3rem;
    }
    .qr-ro {
      background: rgba(10,15,30,0.65);
      border: 1px solid rgba(154,168,188,0.16);
      border-radius: 14px;
      padding: 0.55rem 0.75rem;
      font-size: 0.82rem;
      color: var(--qr-mist);
      margin-bottom: 0.4rem;
    }
    .qr-ro strong { color: #fff; }
    .qr-badge {
      display: inline-block;
      padding: 0.35rem 0.75rem;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.04em;
    }
    .qr-badge.ok {
      background: rgba(0,229,255,0.14); color: var(--qr-cyan);
      border: 1px solid rgba(0,229,255,0.4);
      box-shadow: 0 0 16px rgba(0,229,255,0.14);
    }
    .qr-badge.warn {
      background: rgba(255,138,76,0.12); color: #ffb08a;
      border: 1px solid rgba(255,138,76,0.35);
    }
    .qr-click-panel {
      background: linear-gradient(135deg, rgba(0,229,255,0.08), rgba(14,22,40,0.92));
      border: 1px solid rgba(0,229,255,0.28);
      border-radius: 16px;
      padding: 0.8rem 0.9rem;
      margin: 0.4rem 0 0.7rem 0;
    }
    .qr-click-panel .title {
      font-family: 'DM Sans', system-ui, sans-serif;
      font-size: 0.98rem; font-weight: 700; color: #fff;
      margin-bottom: 0.25rem;
    }
    .qr-footer {
      margin-top: 1.5rem; padding-top: 0.85rem;
      border-top: 1px solid rgba(154,168,188,0.12);
      color: var(--qr-mist); font-size: 0.8rem;
      display: flex; flex-wrap: wrap; gap: 0.75rem; justify-content: space-between;
    }
    .qr-map-hint {
      background: rgba(14,22,40,0.72);
      border-left: 3px solid var(--qr-cyan);
      border-radius: 0 12px 12px 0;
      padding: 0.55rem 0.85rem;
      margin: 0.35rem 0 0.65rem 0;
      color: var(--qr-mist); font-size: 0.9rem;
    }
    .qr-map-hint b { color: #fff; }
    div[data-testid="stMetricValue"] { color: #f4f7fb; }
    /* Primary CTAs — cyan glow pills */
    div[data-testid="stSidebar"] button[kind="primary"],
    button[kind="primary"] {
      font-weight: 700 !important;
      letter-spacing: 0.04em;
      min-height: 3rem;
      border-radius: 999px !important;
      background: linear-gradient(135deg, #00E5FF 0%, #00B8D4 100%) !important;
      color: #041018 !important;
      border: none !important;
      box-shadow: 0 0 22px rgba(0,229,255,0.35), 0 4px 14px rgba(0,0,0,0.25) !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"],
    button[kind="secondary"] {
      border-radius: 999px !important;
      border: 1px solid rgba(154,168,188,0.28) !important;
      background: rgba(14,22,40,0.7) !important;
      color: var(--qr-ink) !important;
    }
    /* Active place-mode pill (cyan fill) */
    div[data-testid="stSidebar"] button[kind="primary"].place-active,
    button[data-testid="baseButton-primary"] {
      /* covered by primary rule above */
    }
    section.main .block-container {
      padding-top: 0.75rem;
      padding-bottom: 1rem;
      max-width: 1680px;
      padding-left: 1rem;
      padding-right: 1rem;
    }
    /* B2C: hide empty collapsed sidebar chrome so map gets full width */
    html.qr-b2c [data-testid="stSidebar"] {
      display: none !important;
    }
    html.qr-b2c [data-testid="stSidebarCollapsedControl"] {
      display: none !important;
    }
    .qr-panel {
      background: linear-gradient(165deg, rgba(14,22,40,0.94), rgba(10,15,30,0.9));
      border: 1px solid rgba(154,168,188,0.16);
      border-radius: 16px;
      padding: 0.85rem 0.95rem;
      margin-bottom: 0.65rem;
    }
    .qr-panel h3 {
      margin: 0 0 0.45rem 0 !important;
      font-size: 0.95rem !important;
      color: #fff !important;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .qr-rec {
      background: linear-gradient(135deg, rgba(0,229,255,0.14), rgba(14,22,40,0.95));
      border: 1px solid rgba(0,229,255,0.45);
      border-radius: 14px;
      padding: 0.75rem 0.85rem;
      margin: 0.4rem 0 0.65rem 0;
    }
    .qr-rec .title { color: #fff; font-weight: 700; font-size: 1.05rem; }
    .qr-rec .meta { color: var(--qr-mist); font-size: 0.82rem; margin-top: 0.25rem; }
    .qr-exit-row {
      display: flex; justify-content: space-between; gap: 0.5rem;
      padding: 0.4rem 0.55rem; margin: 0.25rem 0;
      border-radius: 10px;
      background: rgba(10,15,30,0.55);
      border: 1px solid rgba(154,168,188,0.12);
      font-size: 0.8rem; color: var(--qr-mist);
    }
    .qr-exit-row.best {
      border-color: rgba(0,229,255,0.45);
      background: rgba(0,229,255,0.08);
      color: #fff;
    }
    .qr-map-wrap {
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid rgba(154,168,188,0.14);
      box-shadow: 0 12px 36px rgba(0,0,0,0.35);
    }
    /* Top-level surface switcher */
    div[data-testid="stRadio"] > div {
      gap: 0.35rem;
      background: rgba(10,15,30,0.65);
      border: 1px solid rgba(154,168,188,0.16);
      border-radius: 999px;
      padding: 0.3rem;
    }
    div[data-testid="stRadio"] label {
      background: transparent !important;
      border-radius: 999px !important;
      padding: 0.45rem 0.95rem !important;
      font-family: 'DM Sans', system-ui, sans-serif !important;
      font-weight: 600 !important;
      font-size: 0.95rem !important;
      letter-spacing: 0.02em;
      color: var(--qr-mist) !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stRadio"] label:has(input:checked) {
      background: rgba(0,229,255,0.16) !important;
      color: #fff !important;
      border: 1px solid rgba(0,229,255,0.4);
      box-shadow: 0 0 16px rgba(0,229,255,0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Manila road graph…")
def get_graph():
    return load_or_build_graph()


@st.cache_resource(show_spinner="Preparing Classical FiLM…")
def get_classical_model():
    model, ds = ensure_trained_model(epochs=25, n_episodes=50)
    return model, ds["mean"], ds["std"]


@st.cache_resource(show_spinner="Preparing Hybrid QML (PennyLane)…")
def get_hybrid_model():
    model, ds = ensure_hybrid_model(epochs=25, n_episodes=50)
    return model, ds["mean"], ds["std"]


def nearest_node(G: nx.Graph, lat: float, lon: float, candidates=None):
    """Snap a map click to the nearest graph node (haversine-ish Euclidean deg)."""
    return _rs_nearest_node(G, lat, lon, candidates=candidates)


def path_travel_time(G: nx.Graph, path: List) -> float:
    """Sum edge travel weights along a path (minutes-scale nominal units)."""
    return _rs_path_travel_time(G, path)


def route_overlap_accuracy(pred: List, oracle: List) -> float:
    """Node-set overlap vs Dijkstra oracle (demo-friendly accuracy %)."""
    return _rs_route_overlap(pred, oracle)


def _no_click(layer):
    """Stop Folium overlays from stealing map clicks (Leaflet interactive=False)."""
    try:
        # Folium path_options() silently drops interactive= from Circle() kwargs —
        # always set it on the serialized options dict after construction.
        layer.options["interactive"] = False
        if "bubblingMouseEvents" in layer.options:
            layer.options["bubblingMouseEvents"] = False
        # Popups/tooltips re-enable hit-testing in Leaflet; drop them on overlays.
        if hasattr(layer, "popup"):
            layer.popup = None
        if hasattr(layer, "tooltip"):
            layer.tooltip = None
    except Exception:
        pass
    return layer


def _set_epicenter(lat: float, lon: float) -> None:
    """Write canonical epicenter coords (never bind these keys to number_input)."""
    st.session_state["epi_lat"] = float(lat)
    st.session_state["epi_lon"] = float(lon)
    # Keep Advanced inputs in sync if they exist (separate widget keys).
    st.session_state["epi_lat_input"] = float(lat)
    st.session_state["epi_lon_input"] = float(lon)


def build_base_map(G, exits, map_center, map_zoom: int = 16):
    """Build road graph map. map_center is [lat, lon]."""
    m = folium.Map(
        location=list(map_center),
        zoom_start=int(map_zoom),
        tiles="CartoDB dark_matter",
    )
    for u, v in G.edges():
        u_lat, u_lon = G.nodes[u]["y"], G.nodes[u]["x"]
        v_lat, v_lon = G.nodes[v]["y"], G.nodes[v]["x"]
        line = folium.PolyLine(
            [[u_lat, u_lon], [v_lat, v_lon]],
            color="#2a3548",
            weight=1.5,
            opacity=0.4,
        )
        _no_click(line).add_to(m)

    for i, ex in enumerate(exits):
        marker = folium.CircleMarker(
            location=[G.nodes[ex]["y"], G.nodes[ex]["x"]],
            radius=8,
            color=ORANGE_ACCENT,
            fill=True,
            fill_color=ORANGE_ACCENT,
            fill_opacity=0.85,
            popup=f"Exit {i + 1}",
        )
        _no_click(marker).add_to(m)
    return m


def predict_route(
    G,
    model,
    mean,
    std,
    start,
    dest,
    epicenter_lonlat,
    max_steps: int | None = None,
):
    """Thin wrapper — shared Hybrid / Classical rollout (routing_service)."""
    return predict_escape_route(
        G, model, mean, std, start, dest, epicenter_lonlat, max_steps=max_steps
    )


def dijkstra_route(G, start, dest, epicenter_lonlat, max_steps=120):
    """Oracle node-wise Dijkstra under the same dynamics."""
    path, _radii, _env, travel, _meta = dijkstra_escape_route(
        G, start, dest, epicenter_lonlat, max_steps=max_steps
    )
    return path, travel


# Curated QA scenarios (data/demo_scenarios.json) can still be loaded
# programmatically via _load_demo_scenarios / _apply_demo_scenario — no UI buttons.


def _load_demo_scenarios() -> list:
    """Curated Quantum Advantage scenarios from data/demo_scenarios.json."""
    path = DATA_DIR / "demo_scenarios.json"
    if not path.exists():
        return []
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload.get("scenarios") or [])
    except Exception:
        return []


def _apply_demo_scenario(G, exits, scenario: dict) -> str:
    """Set start / epicenter / exit from a curated scenario; queue auto-calculate."""
    _clear_route_results()
    start = scenario.get("start_node")
    dest = scenario.get("dest_node")
    # Coerce node ids to graph key types
    if start not in G.nodes:
        start = nearest_node(
            G,
            float(scenario["start_lat"]),
            float(scenario["start_lon"]),
            candidates=[n for n in G.nodes() if n not in exits],
        )
    if dest not in G.nodes:
        dest = nearest_node(
            G,
            float(scenario["exit_lat"]),
            float(scenario["exit_lon"]),
            candidates=exits,
        )
    st.session_state["start_node"] = start
    st.session_state["dest_node"] = dest
    _set_epicenter(float(scenario["epi_lat"]), float(scenario["epi_lon"]))
    st.session_state["map_center"] = [
        float(scenario["epi_lat"]),
        float(scenario["epi_lon"]),
    ]
    st.session_state["select_mode"] = "Start"
    st.session_state["flow_step"] = 2
    st.session_state["pending_calculate"] = True
    st.session_state["show_classical_overlay"] = True
    st.session_state["show_dijkstra_overlay"] = True
    title = scenario.get("title", "Quantum Advantage")
    msg = f"Loaded {title} — calculating 3-way compare…"
    st.session_state["map_status"] = msg
    return msg


def _clear_route_results():
    """Drop calculated route so a new apartment / epicenter / exit can be chosen."""
    for k in (
        "path",
        "classical_path",
        "dij_path",
        "radii_trace",
        "qml_travel",
        "classical_travel",
        "dij_travel",
        "sample_x",
        "q_contrib",
        "accuracy",
        "classical_accuracy",
        "model_used",
        "demo_hybrid",
        "epi",
        "start",
        "dest",
        "route_meta",
        "classical_meta",
        "exit_reached",
        "classical_reached",
        "dij_reached",
        "compare_narrative",
        "show_classical_overlay",
        "show_dijkstra_overlay",
        "latency_ms",
        "is_hybrid_route",
    ):
        st.session_state.pop(k, None)


def _apartment_preset_by_id(preset_id: str) -> dict:
    for p in APARTMENT_PRESETS:
        if p["id"] == preset_id:
            return p
    return APARTMENT_PRESETS[0]


def _refresh_exit_ranking(G, exits) -> List[dict]:
    """Rank candidate evacuate areas; set recommended dest unless user locked an exit."""
    start = st.session_state["start_node"]
    epi = (float(st.session_state["epi_lon"]), float(st.session_state["epi_lat"]))
    best, ranking = recommend_best_exit(G, start, exits, epi)
    st.session_state["exit_ranking"] = ranking
    st.session_state["recommended_exit"] = best
    if ranking and not st.session_state.get("exit_override"):
        st.session_state["dest_node"] = best
    return ranking


def _set_apartment(G, exits, lat: float, lon: float, *, preset_id: Optional[str] = None) -> None:
    """Snap apartment pin to nearest non-exit road node."""
    candidates = [n for n in G.nodes() if n not in exits]
    node = nearest_node(G, float(lat), float(lon), candidates)
    st.session_state["start_node"] = node
    st.session_state["apartment_lat"] = float(G.nodes[node]["y"])
    st.session_state["apartment_lon"] = float(G.nodes[node]["x"])
    if preset_id is not None:
        st.session_state["apartment_preset"] = preset_id
        st.session_state["apartment_select"] = preset_id
    st.session_state["map_center"] = [
        float(st.session_state["apartment_lat"]),
        float(st.session_state["apartment_lon"]),
    ]


def _init_session(G, exits, nodes, origin):
    xs = [G.nodes[n]["x"] for n in nodes]
    ys = [G.nodes[n]["y"] for n in nodes]
    if "select_mode" not in st.session_state:
        # B2C: map click places earthquake epicenter by default
        st.session_state["select_mode"] = "Epicenter"
    if "apartment_preset" not in st.session_state:
        st.session_state["apartment_preset"] = APARTMENT_PRESETS[0]["id"]
    if "start_node" not in st.session_state:
        preset = _apartment_preset_by_id(st.session_state["apartment_preset"])
        _set_apartment(G, exits, preset["lat"], preset["lon"], preset_id=preset["id"])
    if "apartment_lat" not in st.session_state:
        n0 = st.session_state["start_node"]
        st.session_state["apartment_lat"] = float(G.nodes[n0]["y"])
        st.session_state["apartment_lon"] = float(G.nodes[n0]["x"])
    if "dest_node" not in st.session_state:
        st.session_state["dest_node"] = exits[0]
    if "epi_lat" not in st.session_state:
        # Default quake slightly SE of apartment so routes are non-trivial
        st.session_state["epi_lat"] = float(np.mean(ys)) - 0.0015
        st.session_state["epi_lon"] = float(np.mean(xs)) + 0.0012
    if "epi_lat_input" not in st.session_state:
        st.session_state["epi_lat_input"] = float(st.session_state["epi_lat"])
        st.session_state["epi_lon_input"] = float(st.session_state["epi_lon"])
    if "flow_step" not in st.session_state:
        st.session_state["flow_step"] = 1
    if "map_center" not in st.session_state:
        st.session_state["map_center"] = [
            float(st.session_state.get("apartment_lat", origin[1])),
            float(st.session_state.get("apartment_lon", origin[0])),
        ]
    if "map_zoom" not in st.session_state:
        st.session_state["map_zoom"] = 16
    if "map_status" not in st.session_state:
        st.session_state["map_status"] = (
            "You are in your apartment. Click the map to place the earthquake epicenter."
        )
    if "exit_override" not in st.session_state:
        st.session_state["exit_override"] = False
    if "exit_ranking" not in st.session_state:
        _refresh_exit_ranking(G, exits)


def _apply_map_click(G, exits, lat: float, lon: float) -> str:
    """B2C map click: Epicenter (default) or Apartment. Dedup by coordinates only."""
    mode = st.session_state.get("select_mode", "Epicenter")
    _clear_route_results()
    st.session_state["flow_step"] = 1
    st.session_state["map_center"] = [float(lat), float(lon)]

    if mode == "Apartment":
        _set_apartment(G, exits, lat, lon, preset_id="custom")
        st.session_state["select_mode"] = "Epicenter"
        st.session_state["exit_override"] = False
        _refresh_exit_ranking(G, exits)
        msg = (
            f"Apartment → node {st.session_state['start_node']}. "
            "Next: click the earthquake epicenter."
        )
    else:
        _set_epicenter(lat, lon)
        st.session_state["exit_override"] = False
        _refresh_exit_ranking(G, exits)
        best = st.session_state.get("recommended_exit", st.session_state["dest_node"])
        msg = (
            f"Epicenter → {lat:.5f}, {lon:.5f}. "
            f"Recommended evacuate area → node {best}."
        )

    st.session_state["map_status"] = msg
    return msg


def main():
    st.markdown(
        '<div class="qr-header">'
        '<div class="qr-brand">Quantum<span>Relief</span></div>'
        '<span class="qr-online"><span class="dot"></span>⚡ Hybrid QML · Online</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qr-tagline">Emergency Escape · Manila · Intramuros</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qr-tag">Wake in your apartment during an earthquake — '
        "we rank evacuate areas and route you out with Hybrid QML.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qr-map-hint" style="margin-top:0.15rem">'
        "<b>Safest + fastest escape:</b> Hybrid QML recommends the best evacuate area, "
        "then compares routes under a live quake sweep. "
        f"<b style='color:{HYBRID_ROUTE_COLOR}'>Cyan = Hybrid QML</b> · "
        f"<b style='color:{CLASSICAL_ROUTE_COLOR}'>Gold = Classical FiLM</b> · "
        f"<b style='color:{DIJKSTRA_ROUTE_COLOR}'>White dashed = Dijkstra</b> · "
        f"<b style='color:{HAZARD_ROUTE_COLOR}'>Red = earthquake</b>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qr-team">'
        '<span class="chip">Team 5 — Quantrio</span>'
        '<span class="chip soft">QC4SG · SEA Quantathon 2026</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    qstat = quantum_status()
    pl_ok = qstat["pennylane_available"]

    try:
        G = get_graph()
    except Exception as e:
        st.error(f"Failed to load Manila graph: {e}")
        st.stop()

    exits = select_exit_nodes(G, n_exits=3, seed=42)
    nodes = list(G.nodes())
    origin = get_graph_origin(G)
    _init_session(G, exits, nodes, origin)

    # Apply pending map click BEFORE widgets (avoids Streamlit overwriting keyed selectboxes)
    if "_map_click" in st.session_state:
        lat_p, lon_p = st.session_state.pop("_map_click")
        msg = _apply_map_click(G, exits, float(lat_p), float(lon_p))
        try:
            st.toast(msg, icon="📍")
        except Exception:
            pass

    # Top-level surfaces (radio = tab-equivalent; only active branch runs — sidebar-safe)
    surface = st.radio(
        "App surface",
        options=["B2C Emergency Escape", "Command Center (God View)"],
        horizontal=True,
        key="app_surface",
        label_visibility="collapsed",
    )

    if surface == "Command Center (God View)":
        # Init session before sidebar widgets so epicenter / batch defaults stick
        init_god_view_state(G, exits, origin)
        with st.sidebar:
            controls = render_god_view_controls()
        hybrid_model = mean = std = None
        if pl_ok:
            try:
                hybrid_model, mean, std = get_hybrid_model()
            except Exception as gv_exc:
                st.warning(
                    f"Hybrid model unavailable for God View — {gv_exc}. "
                    "Metrics still render; trigger uses Dijkstra bulk (no Hybrid heroes)."
                )
        render_god_view(
            G,
            exits,
            hybrid_model,
            mean,
            std,
            pennylane_ok=pl_ok,
            controls=controls,
        )
        return

    # ---------- B2C Emergency Escape: 2/3 map · 1/3 controls ----------
    # Hide collapsed sidebar chrome (controls live in the right panel).
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    flow = int(st.session_state.get("flow_step", 1))
    steps_html = "".join(
        f'<div class="qr-step'
        f'{" active" if flow == i else ""}'
        f'{" done" if flow > i else ""}'
        f'"><b>{i}</b>{label}</div>'
        for i, label in [
            (1, "Apartment + quake"),
            (2, "Rank evacuate areas"),
            (3, "Calculate escape"),
            (4, "Compare routes"),
        ]
    )
    st.markdown(f'<div class="qr-steps">{steps_html}</div>', unsafe_allow_html=True)

    if "howto_seen" not in st.session_state:
        st.session_state["howto_seen"] = False
    with st.expander(
        "How to use Emergency Escape",
        expanded=not st.session_state["howto_seen"],
    ):
        st.markdown(
            """
**You wake in a Manila apartment. An earthquake hits. Which evacuate area is safest and fastest?**

1. Pick **your apartment** (default or a nearby preset)
2. **Click the map** (or Random) to place the **earthquake epicenter**
3. Read the **recommended evacuate area** among candidates you may already know
4. Press **Find safest & fastest escape** — **cyan Hybrid** · **gold Classical** · **white dashed Dijkstra**
5. Scrub **`t`** — watch the red quake ring expand; compare travel times

*Flood / bridge / multi-citizen sims stay on Command Center (God View).*
            """
        )
        if st.button("Got it — hide next time", key="howto_ack"):
            st.session_state["howto_seen"] = True
            st.rerun()

    start_options = [n for n in nodes if n not in exits]
    if st.session_state["start_node"] not in start_options:
        st.session_state["start_node"] = start_options[0]
    if st.session_state["dest_node"] not in exits:
        st.session_state["dest_node"] = exits[0]
        st.session_state["exit_override"] = False

    # Keep ranking fresh when points change without a stored ranking
    if not st.session_state.get("exit_ranking"):
        _refresh_exit_ranking(G, exits)

    map_col, panel_col = st.columns([2, 1], gap="medium")

    # ==================================================================
    # RIGHT PANEL (~1/3) — controls + ranking + metrics
    # ==================================================================
    with panel_col:
        badge = (
            f'<span class="qr-badge ok">PennyLane · {qstat["n_qubits"]}-qubit HQNN</span>'
            if pl_ok
            else '<span class="qr-badge warn">PennyLane unavailable · Classical only</span>'
        )
        st.markdown(
            f'<div class="qr-panel"><h3>Mission</h3>{badge}'
            "<p style='color:#9AA8BC;font-size:0.85rem;margin:0.55rem 0 0 0'>"
            "Earthquake is the hazard. We rank known evacuate areas by "
            "<b style='color:#fff'>safety + speed</b>, then route you with Hybrid QML."
            "</p></div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="qr-panel"><h3>1 · Your apartment</h3></div>', unsafe_allow_html=True)
        preset_labels = {p["id"]: p["label"] for p in APARTMENT_PRESETS}
        preset_labels["custom"] = "Custom (map click)"
        preset_ids = [p["id"] for p in APARTMENT_PRESETS] + (
            ["custom"] if st.session_state.get("apartment_preset") == "custom" else []
        )
        cur_preset = st.session_state.get("apartment_preset", APARTMENT_PRESETS[0]["id"])
        if cur_preset not in preset_ids:
            preset_ids.append(cur_preset)
            preset_labels.setdefault(cur_preset, cur_preset)

        chosen = st.selectbox(
            "Apartment / current location",
            options=preset_ids,
            index=preset_ids.index(cur_preset) if cur_preset in preset_ids else 0,
            format_func=lambda i: preset_labels.get(i, i),
            key="apartment_select",
        )
        if chosen != st.session_state.get("apartment_preset") and chosen != "custom":
            preset = _apartment_preset_by_id(chosen)
            _set_apartment(G, exits, preset["lat"], preset["lon"], preset_id=preset["id"])
            _clear_route_results()
            st.session_state["exit_override"] = False
            _refresh_exit_ranking(G, exits)
            st.session_state["map_status"] = f"Apartment set · {preset['label']}"
            st.session_state["flow_step"] = 1
            st.rerun()

        place_mode = st.session_state.get("select_mode", "Epicenter")
        pm1, pm2 = st.columns(2)
        with pm1:
            if st.button(
                "Place apartment",
                use_container_width=True,
                type="primary" if place_mode == "Apartment" else "secondary",
                help="Next map click snaps your apartment to the nearest road node.",
            ):
                st.session_state["select_mode"] = "Apartment"
                st.rerun()
        with pm2:
            if st.button(
                "Place epicenter",
                use_container_width=True,
                type="primary" if place_mode == "Epicenter" else "secondary",
                help="Next map click sets the earthquake epicenter.",
            ):
                st.session_state["select_mode"] = "Epicenter"
                st.rerun()

        st.caption(st.session_state.get("map_status", ""))

        st.markdown('<div class="qr-panel"><h3>2 · Earthquake</h3></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="qr-ro"><strong>Epicenter</strong><br/>'
            f'{st.session_state["epi_lat"]:.5f}, {st.session_state["epi_lon"]:.5f}</div>',
            unsafe_allow_html=True,
        )
        er1, er2 = st.columns(2)
        with er1:
            if st.button("Random epicenter", use_container_width=True):
                (lon_r, lat_r), _ = random_epicenter(G)
                _set_epicenter(lat_r, lon_r)
                _clear_route_results()
                st.session_state["exit_override"] = False
                _refresh_exit_ranking(G, exits)
                st.session_state["flow_step"] = max(st.session_state.get("flow_step", 1), 2)
                st.session_state["map_status"] = (
                    f"Epicenter set to {lat_r:.5f}, {lon_r:.5f}"
                )
                st.rerun()
        with er2:
            if st.button("Reset route", use_container_width=True):
                _clear_route_results()
                st.session_state["flow_step"] = 1
                st.session_state["map_status"] = (
                    "Route cleared — adjust apartment or epicenter."
                )
                st.rerun()

        with st.expander("Advanced epicenter lat/lon", expanded=False):
            def _on_epi_manual_change():
                st.session_state["epi_lat"] = float(st.session_state["epi_lat_input"])
                st.session_state["epi_lon"] = float(st.session_state["epi_lon_input"])
                _clear_route_results()
                st.session_state["exit_override"] = False
                st.session_state["flow_step"] = 1
                st.session_state["_need_rank_refresh"] = True

            e1, e2 = st.columns(2)
            with e1:
                st.number_input(
                    "Lat",
                    format="%.5f",
                    key="epi_lat_input",
                    on_change=_on_epi_manual_change,
                )
            with e2:
                st.number_input(
                    "Lon",
                    format="%.5f",
                    key="epi_lon_input",
                    on_change=_on_epi_manual_change,
                )

        if st.session_state.pop("_need_rank_refresh", False):
            _refresh_exit_ranking(G, exits)

        ranking = st.session_state.get("exit_ranking") or _refresh_exit_ranking(G, exits)
        st.session_state["flow_step"] = max(st.session_state.get("flow_step", 1), 2)

        st.markdown(
            '<div class="qr-panel"><h3>3 · Evacuate areas</h3>'
            "<p style='color:#9AA8BC;font-size:0.8rem;margin:0'>"
            "You may know these safe zones — we pick the best one for this quake."
            "</p></div>",
            unsafe_allow_html=True,
        )

        if ranking:
            best = ranking[0]
            st.markdown(
                f'<div class="qr-rec"><div class="title">Recommended · {best["label"]}</div>'
                f'<div class="meta">Score {best["combined_score"]:.0f}/100 · {best["why"]}</div>'
                f'<div class="meta" style="margin-top:0.35rem;color:#00E5FF">'
                f'Node {best["exit_node"]} · '
                f'{"reachable" if best["exit_reached"] else "blocked"}</div></div>',
                unsafe_allow_html=True,
            )
            for row in ranking:
                cls = " best" if row.get("recommended") else ""
                tag = " ★ BEST" if row.get("recommended") else ""
                t_txt = (
                    f'{row["travel_time"]:.1f}'
                    if row["exit_reached"] and np.isfinite(row["travel_time"])
                    else "—"
                )
                st.markdown(
                    f'<div class="qr-exit-row{cls}">'
                    f'<span>#{row["rank"]} {row["label"]}{tag}</span>'
                    f'<span>{t_txt} · {row["safety_km"]:.2f} km · '
                    f'{row["combined_score"]:.0f}</span></div>',
                    unsafe_allow_html=True,
                )

            exit_choices = [r["exit_node"] for r in ranking]
            labels = {
                r["exit_node"]: (
                    f'{"★ " if r.get("recommended") else ""}'
                    f'{r["label"]} · score {r["combined_score"]:.0f}'
                )
                for r in ranking
            }
            dest_idx = (
                exit_choices.index(st.session_state["dest_node"])
                if st.session_state["dest_node"] in exit_choices
                else 0
            )
            picked = st.selectbox(
                "Evacuate area (override)",
                options=exit_choices,
                index=dest_idx,
                format_func=lambda n: labels.get(n, str(n)),
                help="Default is the recommended area. Override if you prefer another known exit.",
            )
            if picked != st.session_state["dest_node"]:
                st.session_state["dest_node"] = picked
                rec = st.session_state.get("recommended_exit")
                st.session_state["exit_override"] = bool(rec is not None and picked != rec)
                _clear_route_results()
                st.session_state["map_status"] = f"Evacuate area set → node {picked}"
                st.rerun()
        else:
            st.warning("No evacuate areas available on this graph.")

        st.markdown('<div class="qr-panel"><h3>4 · Escape engine</h3></div>', unsafe_allow_html=True)
        if pl_ok:
            st.caption("Hero: Hybrid QML (PennyLane PHN). Overlays optional.")
        else:
            st.caption(qstat["note"])

        compare_classical = st.checkbox(
            "Show Classical FiLM (gold)",
            value=True,
            key="b2c_cmp_classical",
        )
        compare_dij = st.checkbox(
            "Show Dijkstra (white dashed)",
            value=True,
            key="b2c_cmp_dij",
        )

        run = st.button(
            "Find safest & fastest escape",
            type="primary",
            use_container_width=True,
        )
        if st.session_state.pop("pending_calculate", False):
            run = True
        st.caption(
            f"Manila graph · {G.number_of_nodes()} nodes · {G.number_of_edges()} edges"
        )

        # ---- Calculate (results stored; metrics rendered below in panel) ----
        start = st.session_state["start_node"]
        dest = st.session_state["dest_node"]
        epi_lat = float(st.session_state["epi_lat"])
        epi_lon = float(st.session_state["epi_lon"])

        if run:
            st.session_state["flow_step"] = 3
            use_hybrid = bool(pl_ok)
            hybrid_fell_back = False
            try:
                with st.spinner("Ranking locked · running Hybrid · Classical · Dijkstra…"):
                    hybrid_model = None
                    mean = std = None
                    if use_hybrid:
                        try:
                            hybrid_model, mean, std = get_hybrid_model()
                        except Exception as hybrid_exc:
                            err = str(hybrid_exc).lower()
                            if "numpy" not in err and "pennylane" not in err:
                                raise
                            hybrid_fell_back = True
                            use_hybrid = False
                            st.warning(
                                "Hybrid QML runtime glitch "
                                f"({type(hybrid_exc).__name__}: {hybrid_exc}). "
                                "Falling back to Classical FiLM as hero."
                            )

                    classical_model, c_mean, c_std = get_classical_model()
                    if mean is None:
                        mean, std = c_mean, c_std

                    hero_model = hybrid_model if use_hybrid else classical_model
                    label = (
                        "Hybrid QML (HQNN)"
                        if use_hybrid and not hybrid_fell_back
                        else "Classical FiLM (ablation)"
                    )

                    cmp = compare_three_way(
                        G,
                        hero_model,
                        classical_model if compare_classical else None,
                        mean,
                        std,
                        start,
                        dest,
                        (epi_lon, epi_lat),
                        include_classical=bool(compare_classical),
                        include_dijkstra=bool(compare_dij),
                    )

                    h = cmp["hybrid"]
                    path = h["path"]
                    radii_trace = h["radii_trace"]
                    qml_travel = h["travel_time"]
                    sample_x = h.get("sample_x")
                    route_meta = h["meta"]
                    reached = bool(h["exit_reached"]) and path[-1] == dest

                    if not path or len(path) < 2:
                        raise RuntimeError(
                            "No escape hops found — try another epicenter or apartment."
                        )

                    classical_path = None
                    classical_travel = 0.0
                    classical_meta = {}
                    classical_reached = False
                    classical_accuracy = 0.0
                    if compare_classical and cmp.get("classical"):
                        c = cmp["classical"]
                        classical_path = c["path"]
                        classical_travel = float(c["travel_time"])
                        classical_meta = c.get("meta") or {}
                        classical_reached = bool(c["exit_reached"])
                        classical_accuracy = float(
                            c.get("overlap_vs_dijkstra_pct") or 0.0
                        )

                    dij_path, dij_travel = (None, 0.0)
                    dij_reached = False
                    if compare_dij and cmp.get("dijkstra"):
                        d = cmp["dijkstra"]
                        dij_path = d["path"]
                        dij_travel = float(d["travel_time"])
                        dij_reached = bool(d["exit_reached"])

                    q_contrib = None
                    if use_hybrid and not hybrid_fell_back and hybrid_model is not None:
                        raw_q = h.get("quantum_contribution")
                        if raw_q is not None:
                            try:
                                q_contrib = float(raw_q)
                            except (TypeError, ValueError):
                                q_contrib = None
                        if q_contrib is None or q_contrib <= 0:
                            q_contrib = estimate_quantum_contribution_pct(
                                hybrid_model, sample_x
                            )

                    accuracy = float(h.get("overlap_vs_dijkstra_pct") or 0.0)
                    if dij_path and accuracy <= 0:
                        accuracy = route_overlap_accuracy(path, dij_path)
                    if classical_path and dij_path and classical_accuracy <= 0:
                        classical_accuracy = route_overlap_accuracy(
                            classical_path, dij_path
                        )

                    st.session_state.update(
                        {
                            "path": path,
                            "classical_path": classical_path if compare_classical else None,
                            "dij_path": dij_path if compare_dij else None,
                            "radii_trace": radii_trace,
                            "qml_travel": qml_travel,
                            "classical_travel": classical_travel,
                            "dij_travel": dij_travel,
                            "sample_x": sample_x,
                            "q_contrib": q_contrib,
                            "accuracy": accuracy,
                            "classical_accuracy": classical_accuracy,
                            "model_used": label,
                            "route_meta": route_meta,
                            "classical_meta": classical_meta,
                            "exit_reached": reached,
                            "classical_reached": classical_reached,
                            "dij_reached": dij_reached,
                            "compare_narrative": cmp.get("narrative") or {},
                            "latency_ms": cmp.get("latency_ms") or {},
                            "show_classical_overlay": bool(compare_classical),
                            "show_dijkstra_overlay": bool(compare_dij),
                            "demo_hybrid": bool(
                                getattr(hero_model, "demo_mode", False)
                                and use_hybrid
                                and not hybrid_fell_back
                            ),
                            "is_hybrid_route": bool(use_hybrid and not hybrid_fell_back),
                            "epi": (epi_lon, epi_lat),
                            "start": start,
                            "dest": dest,
                            "flow_step": 4,
                        }
                    )
                    try:
                        st.toast(
                            "Escape ready — scrub t · compare Hybrid / Classical / Dijkstra.",
                            icon="✅",
                        )
                    except Exception:
                        pass
            except Exception as e:
                detail = str(e)
                hint = ""
                if "numpy" in detail.lower():
                    hint = (
                        " Hint: Streamlit Cloud needs `numpy==1.26.4` listed before "
                        "`torch==2.2.2` in requirements.txt."
                    )
                st.error(f"Route calculation failed: {e}.{hint}")

        # ---- Metrics (right panel) ----
        path = st.session_state.get("path")
        if path:
            qml_travel = float(st.session_state.get("qml_travel", 0.0))
            classical_travel = st.session_state.get("classical_travel")
            dij_travel = st.session_state.get("dij_travel")
            accuracy = float(st.session_state.get("accuracy", 0.0))
            classical_accuracy = float(st.session_state.get("classical_accuracy", 0.0))
            _q_raw = st.session_state.get("q_contrib")
            try:
                q_contrib = float(_q_raw) if _q_raw is not None else None
            except (TypeError, ValueError):
                q_contrib = None
            model_used = st.session_state.get("model_used", "Hybrid QML")
            narrative = st.session_state.get("compare_narrative") or {}
            reached = bool(st.session_state.get("exit_reached", False))
            is_hybrid = "Hybrid" in str(model_used)
            classical_path = st.session_state.get("classical_path")
            dij_path = st.session_state.get("dij_path")

            beats_classical = (
                classical_path is not None
                and classical_travel is not None
                and reached
                and (
                    qml_travel <= float(classical_travel) * 1.02
                    or (
                        accuracy >= classical_accuracy
                        and qml_travel <= float(classical_travel) * 1.08
                    )
                )
            )
            near_dij = (
                dij_travel is not None
                and dij_path
                and reached
                and qml_travel <= float(dij_travel) * 1.25
            )
            if narrative.get("hybrid_beats_classical") is not None:
                beats_classical = bool(narrative["hybrid_beats_classical"])
            if narrative.get("hybrid_near_dijkstra") is not None:
                near_dij = bool(narrative["hybrid_near_dijkstra"])

            st.markdown(
                '<div class="qr-panel"><h3>5 · Escape metrics</h3></div>',
                unsafe_allow_html=True,
            )
            win = " win" if beats_classical or (reached and is_hybrid) else ""
            st.markdown(
                f'<div class="qr-card hybrid{win}" style="margin-bottom:0.45rem">'
                f'<span class="qr-hero-pill">HERO</span>'
                f'<div class="label">Hybrid travel</div>'
                f'<div class="value accent">{qml_travel:.1f}</div>'
                f'<div class="sub">Cyan · local quantum-classical</div></div>',
                unsafe_allow_html=True,
            )
            c_val = (
                f"{float(classical_travel):.1f}"
                if classical_travel is not None and classical_path
                else "—"
            )
            d_val = (
                f"{float(dij_travel):.1f}"
                if dij_travel is not None and dij_path
                else "—"
            )
            st.markdown(
                f'<div class="qr-card classical" style="margin-bottom:0.45rem">'
                f'<div class="label">Classical · Dijkstra</div>'
                f'<div class="value gold" style="font-size:1.45rem">{c_val}'
                f' <span style="color:#9AA8BC;font-size:0.9rem">/</span> '
                f'<span class="dij">{d_val}</span></div>'
                f'<div class="sub">Gold ablation · white dashed oracle</div></div>',
                unsafe_allow_html=True,
            )

            q_val = (
                f"{q_contrib:.1f}%"
                if is_hybrid and q_contrib is not None and q_contrib > 0
                else ("N/A" if is_hybrid else "—")
            )
            st.markdown(
                f'<div class="qr-card" style="margin-bottom:0.45rem">'
                f'<div class="label">Exit · quality · quantum</div>'
                f'<div class="value" style="font-size:1.25rem">'
                f'{"YES" if reached else "NO"} · {accuracy:.0f}% · {q_val}</div>'
                f'<div class="sub">Reached · overlap vs Dijkstra · PHN contrib</div></div>',
                unsafe_allow_html=True,
            )

            story = (
                "Hybrid beats Classical · near Dijkstra"
                if beats_classical and near_dij
                else (
                    "Hybrid beats Classical"
                    if beats_classical
                    else (
                        "Hybrid approaches Dijkstra"
                        if near_dij
                        else f"{model_used} · local inference"
                    )
                )
            )
            st.markdown(
                f'<div class="qr-card{" win" if beats_classical or near_dij else ""}">'
                f'<div class="label">Verdict</div>'
                f'<div class="value" style="font-size:1.15rem">{story}</div></div>',
                unsafe_allow_html=True,
            )

            latency = st.session_state.get("latency_ms") or {}
            if latency:
                def _ms(v):
                    return f"{v:.0f}" if isinstance(v, (int, float)) else "—"

                st.caption(
                    "Latency · H "
                    f"{_ms(latency.get('hybrid'))} ms · C "
                    f"{_ms(latency.get('classical'))} · D "
                    f"{_ms(latency.get('dijkstra'))}"
                )

            with st.expander("What is Quantum Contribution?", expanded=False):
                q_line = (
                    f"≈ **{q_contrib:.1f}%** this run"
                    if q_contrib is not None and q_contrib > 0
                    else "N/A this run"
                )
                st.markdown(
                    f"""
**Live from Hybrid checkpoint** ({q_line}).

`W = model.combine.weight` · classical cols 0–4 · quantum cols 5–9

```
Quantum Contribution % = 100 × mean(|W_q|) / (mean(|W_c|) + mean(|W_q|))
```

{QUANTUM_CONTRIBUTION_FORMULA}
                    """
                )
        else:
            st.info(
                "Set apartment + epicenter, review the recommended evacuate area, "
                "then press **Find safest & fastest escape**."
            )

    # ==================================================================
    # LEFT MAP (~2/3) — Folium 2D only
    # ==================================================================
    with map_col:
        path = st.session_state.get("path")
        classical_path = st.session_state.get("classical_path")
        dij_path = st.session_state.get("dij_path")
        radii_trace = st.session_state.get("radii_trace")
        start_draw = st.session_state["start_node"]
        dest_draw = st.session_state["dest_node"]
        epi = (float(st.session_state["epi_lon"]), float(st.session_state["epi_lat"]))
        ranking = st.session_state.get("exit_ranking") or []

        t_show = 0
        step_reveal = None
        if radii_trace and path and len(path) >= 2:
            max_t = max(0, len(radii_trace) - 1)
            t_show = st.slider(
                "Scrub hazard time  t",
                0,
                max_t,
                max_t,
                help="Expanding earthquake (red) and exit congestion (gold).",
            )
            step_reveal = min(t_show + 1, len(path) - 1)
            st.session_state["flow_step"] = max(st.session_state.get("flow_step", 4), 4)

        mode = st.session_state.get("select_mode", "Epicenter")
        st.markdown(
            f'<div class="qr-map-hint"><b>Map click: {mode}</b> — '
            f'{st.session_state.get("map_status", "Click to place the epicenter.")}'
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="qr-map-wrap">', unsafe_allow_html=True)
        m = build_base_map(
            G,
            exits,
            st.session_state["map_center"],
            int(st.session_state.get("map_zoom", 16)),
        )

        # Highlight recommended / ranked evacuate areas
        for row in ranking:
            color = HYBRID_ROUTE_COLOR if row.get("recommended") else ORANGE_ACCENT
            radius = 11 if row.get("recommended") else 8
            marker = folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=(
                    f'{row["label"]} · rank {row["rank"]} · '
                    f'score {row["combined_score"]:.0f}'
                ),
            )
            _no_click(marker).add_to(m)

        r_epi = damage_radius(t_show)
        r_exit = exit_radius(t_show)
        if radii_trace and 0 <= t_show < len(radii_trace):
            r_epi = float(radii_trace[t_show]["r_epi"])
            r_exit = float(radii_trace[t_show]["r_exit"])

        for frac, op in [(1.0, 0.10), (0.75, 0.16), (0.3, 0.28)]:
            ring = folium.Circle(
                location=[epi[1], epi[0]],
                radius=frac * r_epi * 1000.0,
                color=HAZARD_ROUTE_COLOR,
                weight=2 if frac == 1.0 else 1,
                fill=True,
                fill_color=HAZARD_ROUTE_COLOR,
                fill_opacity=op,
            )
            _no_click(ring).add_to(m)

        exit_lat = G.nodes[dest_draw]["y"]
        exit_lon = G.nodes[dest_draw]["x"]
        for frac, op in [(1.0, 0.10), (0.75, 0.16), (0.5, 0.22)]:
            ring = folium.Circle(
                location=[exit_lat, exit_lon],
                radius=max(frac * r_exit * 1000.0, 12.0),
                color=EXIT_RING_COLOR,
                weight=2 if frac == 1.0 else 1,
                fill=True,
                fill_color=EXIT_RING_COLOR,
                fill_opacity=op,
            )
            _no_click(ring).add_to(m)

        _no_click(
            folium.Marker(
                [epi[1], epi[0]],
                icon=folium.Icon(color="red", icon="warning-sign"),
            )
        ).add_to(m)
        _no_click(
            folium.Marker(
                [G.nodes[start_draw]["y"], G.nodes[start_draw]["x"]],
                icon=folium.Icon(color="blue", icon="home"),
            )
        ).add_to(m)
        _no_click(
            folium.Marker(
                [exit_lat, exit_lon],
                icon=folium.Icon(color="orange", icon="flag"),
            )
        ).add_to(m)

        route_label = st.session_state.get("model_used", "Hybrid QML (HQNN)")
        if st.session_state.get("is_hybrid_route", "Hybrid" in str(route_label)):
            route_label = "Hybrid QML · HQNN (quantum-classical PHN)"

        if dij_path and len(dij_path) >= 2:
            coords_d = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in dij_path]
            dij_line = folium.PolyLine(
                coords_d,
                color=DIJKSTRA_ROUTE_COLOR,
                weight=3,
                opacity=0.75,
                dash_array="8 10",
                popup="Dijkstra · full dynamic weights",
            )
            _no_click(dij_line).add_to(m)

        if classical_path and len(classical_path) >= 2:
            coords_c = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in classical_path]
            class_line = folium.PolyLine(
                coords_c,
                color=CLASSICAL_ROUTE_COLOR,
                weight=4,
                opacity=0.88,
                popup="Classical FiLM (ablation)",
            )
            _no_click(class_line).add_to(m)

        if path and len(path) >= 2:
            end_i = step_reveal if step_reveal is not None else len(path) - 1
            partial = path[: end_i + 1]
            coords = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in partial]
            route = folium.PolyLine(
                coords,
                color=HYBRID_ROUTE_COLOR,
                weight=6,
                opacity=0.95,
                popup=f"{route_label} escape route",
            )
            _no_click(route).add_to(m)
            for n in partial:
                dot = folium.CircleMarker(
                    [G.nodes[n]["y"], G.nodes[n]["x"]],
                    radius=4,
                    color=HYBRID_ROUTE_COLOR,
                    fill=True,
                    fill_opacity=0.95,
                )
                _no_click(dot).add_to(m)

        map_data = st_folium(
            m,
            key="qr_map_b2c",
            height=720,
            use_container_width=True,
            returned_objects=["last_clicked"],
            center=st.session_state["map_center"],
            zoom=int(st.session_state.get("map_zoom", 16)),
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if map_data and map_data.get("last_clicked"):
            click = map_data["last_clicked"]
            if click and "lat" in click and "lng" in click:
                lat_c, lon_c = float(click["lat"]), float(click["lng"])
                click_key = (round(lat_c, 6), round(lon_c, 6))
                if click_key != st.session_state.get("_last_click_key"):
                    st.session_state["_last_click_key"] = click_key
                    st.session_state["_map_click"] = (lat_c, lon_c)
                    st.rerun()

    st.markdown(
        '<div class="qr-footer">'
        "<span>Team 5 — Quantrio · QC4SG — SEA Quantathon 2026</span>"
        "<span>B2C Emergency Escape · 2D Folium · Quantum Intelligence. Human Relief.</span>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
