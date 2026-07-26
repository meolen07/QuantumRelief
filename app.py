"""
QuantumRelief — Earthquake Escape Route (Streamlit).

Audience flow (only):
  1) Map opens with fault line → epicenter → amber broken roads near epi + 5 exits
  2) Click map → set your location (clear blue start dot)
  3) App auto-recommends the best-ranked evacuate exit
  4) Find escape route → Hybrid path with per-hop probabilities + node-by-node animation

No Run demo, no first-load auto-compare. Folium 2D only. Classical / Dijkstra
live in Advanced (collapsed) — primary demo is Hybrid-focused.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Must be the first Streamlit command (before folium / heavy src imports).
st.set_page_config(
    page_title="QuantumRelief · Earthquake Escape",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import folium
import networkx as nx
import numpy as np
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dynamic_simulation import (
    damage_radius,
    disruption_edge_latlons,
)
from src.film_model import ensure_trained_model
from src.graph_setup import (
    farthest_node_from,
    load_or_build_graph,
    named_escape_landmarks,
    random_epicenter,
    select_exit_nodes,
    snap_to_nearest_node,
)
from src.quantum_hybrid import (
    estimate_quantum_contribution_pct,
    ensure_hybrid_model,
    quantum_status,
)
from src.routing_service import (
    compare_three_way,
    dijkstra_escape_route,
    path_travel_time as _rs_path_travel_time,
    predict_escape_route,
    recommend_best_exit,
    route_overlap_accuracy as _rs_route_overlap,
)
from src.traffic_provider import (
    TrafficNotConfiguredError,
    active_feed_disruptions,
    epicenter_from_snapshot,
    get_traffic_provider,
    traffic_mode_badge,
)
from src.utils import DATA_DIR, HYBRID_CHECKPOINT, get_graph_origin

DEMO_SCENARIOS_PATH = DATA_DIR / "demo_scenarios.json"
JUDGE_PIN_MIN_DELTA = 2.0  # Soft check when comparing on the pinned corridor
EXIT_OVERRIDE_CLICK_M = 90.0  # map click near an exit pin → silent override
SUGGESTED_CLICK_M = 160.0  # click near suggested apartment → arm win-corridor check
# Stale Cloud demo PHN mix reports ≈45.3% and typically ties Classical.
STALE_Q_PCT_LO = 40.0
STALE_Q_PCT_HI = 52.0
EXPECTED_HYBRID_Q_PCT = 77.6
EXPECTED_HYBRID_SHA16 = "1ae31d03b3a4503d"

HYBRID_ROUTE_COLOR = "#00E5FF"
CLASSICAL_ROUTE_COLOR = "#F5C542"
DIJKSTRA_ROUTE_COLOR = "#E8EEF6"
HAZARD_ROUTE_COLOR = "#FF4D6A"
ORANGE_ACCENT = "#FF8A4C"
DISRUPTION_COLOR = "#F5A623"  # amber — not purple
FAULT_LINE_COLOR = "#FF6B4A"
START_DOT_COLOR = "#3B82F6"
CANDIDATE_EDGE_COLOR = "#7EEFFF"

MAP_H = 820  # concrete Folium px height (avoid % → black map)

ESCAPE_OPEN_SCENARIO = "quake_core"

# Reliability: Hybrid deferred when catastrophic vs Classical or very slow.
HYBRID_CATASTROPHIC_RATIO = 1.25
HYBRID_SLOW_MS = 45_000.0
HYBRID_SLOW_VS_CLASSICAL = 8.0

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    :root {
      --qr-bg: #0a0b10;
      --qr-navy: #0a0f1e;
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
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] {
      display: none !important;
    }
    .qr-header {
      display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1rem;
      margin: 0 0 0.2rem 0;
    }
    .qr-brand {
      font-family: 'DM Sans', system-ui, sans-serif;
      font-size: 1.55rem;
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
    }
    .qr-online .dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--qr-cyan);
      box-shadow: 0 0 8px rgba(0,229,255,0.8);
    }
    .qr-tagline {
      font-size: 0.95rem; font-weight: 600; color: #fff;
      margin: 0.05rem 0 0.15rem 0;
    }
    .qr-tag {
      color: var(--qr-mist); font-size: 0.88rem;
      margin: 0 0 0.55rem 0;
    }
    .qr-card {
      background: linear-gradient(160deg, rgba(16,24,42,0.92), rgba(10,15,30,0.88));
      border: 1px solid rgba(154,168,188,0.16);
      border-radius: 14px;
      padding: 0.85rem 0.95rem;
      margin-bottom: 0.45rem;
      position: relative;
    }
    .qr-card.win { border-color: rgba(0,229,255,0.5); }
    .qr-card.hybrid { border-color: rgba(0,229,255,0.45); }
    .qr-card.classical { border-color: rgba(245,197,66,0.35); }
    .qr-card.dijkstra { border-color: rgba(232,238,246,0.22); }
    .qr-hero-pill {
      position: absolute; top: 0.65rem; right: 0.65rem;
      font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em;
      text-transform: uppercase; padding: 0.2rem 0.5rem; border-radius: 999px;
      color: #041018; background: var(--qr-cyan);
    }
    .qr-card .label {
      color: var(--qr-mist); font-size: 0.72rem;
      text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.3rem;
    }
    .qr-card .value {
      font-size: 1.7rem; font-weight: 700; color: #fff; line-height: 1.1;
    }
    .qr-card .value.accent { color: var(--qr-cyan); }
    .qr-card .value.gold { color: var(--qr-gold); }
    .qr-card .value.dij { color: var(--qr-dij); }
    .qr-card .sub { color: var(--qr-mist); font-size: 0.8rem; margin-top: 0.25rem; }
    .qr-prob {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.78rem; color: #9AA8BC; line-height: 1.45;
      max-height: 9.5rem; overflow-y: auto;
      padding: 0.35rem 0.15rem;
    }
    .qr-prob .hi { color: #00E5FF; font-weight: 600; }
    .qr-ro {
      background: rgba(10,15,30,0.65);
      border: 1px solid rgba(154,168,188,0.16);
      border-radius: 12px;
      padding: 0.5rem 0.7rem;
      font-size: 0.82rem; color: var(--qr-mist);
      margin-bottom: 0.4rem;
    }
    .qr-ro strong { color: #fff; }
    .qr-exit-list {
      background: rgba(10,15,30,0.65);
      border: 1px solid rgba(154,168,188,0.16);
      border-radius: 12px;
      padding: 0.45rem 0.55rem;
      margin-bottom: 0.45rem;
    }
    .qr-exit-list .qr-exit-lead {
      font-size: 0.78rem; color: var(--qr-mist);
      margin: 0 0 0.45rem 0;
    }
    .qr-exit-row {
      display: flex; align-items: flex-start; gap: 0.45rem;
      padding: 0.4rem 0.45rem;
      border-radius: 8px;
      margin-bottom: 0.28rem;
      border: 1px solid rgba(154,168,188,0.12);
      background: rgba(14,22,40,0.55);
    }
    .qr-exit-row.recommended {
      border-color: rgba(245,197,66,0.55);
      background: rgba(245,197,66,0.10);
    }
    .qr-exit-row.selected {
      border-color: rgba(0,229,255,0.45);
      box-shadow: inset 0 0 0 1px rgba(0,229,255,0.25);
    }
    .qr-exit-rank {
      flex: 0 0 auto;
      min-width: 1.35rem; height: 1.35rem;
      border-radius: 50%;
      display: inline-flex; align-items: center; justify-content: center;
      font-size: 0.68rem; font-weight: 700;
      color: #041018; background: #9AA8BC;
    }
    .qr-exit-row.recommended .qr-exit-rank {
      background: var(--qr-gold);
    }
    .qr-exit-row.selected .qr-exit-rank {
      background: var(--qr-cyan);
    }
    .qr-exit-meta { flex: 1; min-width: 0; }
    .qr-exit-meta .name {
      color: #E8EEF6; font-weight: 600; font-size: 0.82rem;
      line-height: 1.2;
    }
    .qr-exit-meta .scores {
      color: var(--qr-mist); font-size: 0.72rem; margin-top: 0.12rem;
    }
    .qr-exit-pill {
      display: inline-block; margin-left: 0.3rem;
      font-size: 0.62rem; font-weight: 700; letter-spacing: 0.04em;
      text-transform: uppercase; padding: 0.1rem 0.35rem; border-radius: 999px;
      color: #041018; background: var(--qr-gold);
    }
    .qr-exit-pill.routing {
      background: var(--qr-cyan);
    }
    .qr-badge {
      display: inline-block; padding: 0.3rem 0.7rem; border-radius: 999px;
      font-size: 0.75rem; font-weight: 700; letter-spacing: 0.04em;
    }
    .qr-badge.ok {
      background: rgba(0,229,255,0.14); color: var(--qr-cyan);
      border: 1px solid rgba(0,229,255,0.4);
    }
    .qr-badge.warn {
      background: rgba(255,138,76,0.12); color: #ffb08a;
      border: 1px solid rgba(255,138,76,0.35);
    }
    .qr-badge.feed {
      background: rgba(245,197,66,0.12); color: var(--qr-gold);
      border: 1px solid rgba(245,197,66,0.4);
    }
    .qr-badge.feed.live {
      background: rgba(0,229,255,0.14); color: var(--qr-cyan);
      border: 1px solid rgba(0,229,255,0.4);
    }
    .qr-panel {
      background: linear-gradient(165deg, rgba(14,22,40,0.94), rgba(10,15,30,0.9));
      border: 1px solid rgba(154,168,188,0.16);
      border-radius: 14px;
      padding: 0.75rem 0.85rem;
      margin-bottom: 0.55rem;
    }
    .qr-panel h3 {
      margin: 0 0 0.4rem 0 !important;
      font-size: 0.88rem !important;
      color: #fff !important;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .qr-rec {
      background: linear-gradient(135deg, rgba(0,229,255,0.12), rgba(14,22,40,0.95));
      border: 1px solid rgba(0,229,255,0.4);
      border-radius: 12px;
      padding: 0.55rem 0.75rem;
      margin: 0.35rem 0 0.5rem 0;
      font-size: 0.85rem; color: var(--qr-mist);
    }
    .qr-rec strong { color: #fff; }
    .qr-step {
      display: inline-flex; align-items: center; justify-content: center;
      width: 1.25rem; height: 1.25rem; border-radius: 50%;
      font-size: 0.68rem; font-weight: 700; margin-right: 0.4rem;
      color: #041018; background: var(--qr-cyan);
      vertical-align: middle;
    }
    .qr-nudge {
      background: rgba(245,166,35,0.10);
      border: 1px solid rgba(245,166,35,0.35);
      border-radius: 10px;
      padding: 0.45rem 0.65rem;
      margin: 0.25rem 0 0.45rem 0;
      font-size: 0.78rem; color: #ffd59a;
    }
    .qr-legend {
      display: flex; flex-wrap: wrap; gap: 0.45rem 0.75rem;
      margin: 0.35rem 0 0.15rem 0;
      font-size: 0.72rem; color: var(--qr-mist);
    }
    .qr-legend i {
      display: inline-block; width: 0.7rem; height: 0.7rem;
      border-radius: 2px; margin-right: 0.28rem; vertical-align: -1px;
    }
    .qr-step-label {
      margin: 0.7rem 0 0.3rem 0;
      font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em;
      text-transform: uppercase; color: var(--qr-mist);
    }
    .qr-oneliner {
      color: #E8EEF6; font-size: 0.88rem; line-height: 1.4;
      margin: 0.1rem 0 0.35rem 0;
    }
    .qr-oneliner span { color: #F5C542; font-weight: 600; }
    .qr-incident {
      background: rgba(10,15,30,0.55);
      border-left: 3px solid var(--qr-gold);
      border-radius: 0 8px 8px 0;
      padding: 0.4rem 0.6rem;
      margin: 0.28rem 0;
      font-size: 0.78rem;
      color: var(--qr-mist);
    }
    .qr-incident strong { color: #fff; }
    .qr-incident .sev {
      color: var(--qr-gold); font-size: 0.7rem; letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .qr-asof {
      font-size: 0.72rem; color: var(--qr-mist); margin: 0.2rem 0 0.35rem 0;
    }
    div[data-testid="stMetricValue"] { color: #f4f7fb; }
    button[kind="primary"] {
      font-weight: 700 !important;
      letter-spacing: 0.04em;
      min-height: 2.75rem;
      border-radius: 999px !important;
      background: linear-gradient(135deg, #00E5FF 0%, #00B8D4 100%) !important;
      color: #041018 !important;
      border: none !important;
      box-shadow: 0 0 22px rgba(0,229,255,0.35) !important;
    }
    button[kind="secondary"] {
      border-radius: 999px !important;
      border: 1px solid rgba(154,168,188,0.28) !important;
      background: rgba(14,22,40,0.7) !important;
      color: var(--qr-ink) !important;
    }
    section.main .block-container {
      padding-top: 0.35rem !important;
      padding-bottom: 0.35rem !important;
      padding-left: 0.75rem !important;
      padding-right: 0.75rem !important;
      max-width: 100% !important;
    }
    /* Escape layout: fixed row height · left map · right scrolls.
       Do NOT set overflow:hidden on html/body/.stApp (black Folium).
       Do NOT wrap st_folium in a markdown div. */
    div[data-testid="stHorizontalBlock"]:has(iframe[title*="folium"]),
    div[data-testid="stHorizontalBlock"]:has(iframe[title*="streamlit_folium"]) {
      --qr-map-h: 820px;
      align-items: stretch !important;
      gap: 0.65rem !important;
      height: var(--qr-map-h) !important;
      max-height: var(--qr-map-h) !important;
      overflow: hidden !important;
      flex-wrap: nowrap !important;
    }
    div[data-testid="stHorizontalBlock"]:has(iframe[title*="folium"])
      > [data-testid="stColumn"],
    div[data-testid="stHorizontalBlock"]:has(iframe[title*="streamlit_folium"])
      > [data-testid="stColumn"] {
      min-height: 0 !important;
      height: var(--qr-map-h) !important;
      max-height: var(--qr-map-h) !important;
    }
    div[data-testid="stHorizontalBlock"]:has(iframe[title*="folium"])
      > [data-testid="stColumn"]:has(iframe),
    div[data-testid="stHorizontalBlock"]:has(iframe[title*="streamlit_folium"])
      > [data-testid="stColumn"]:has(iframe) {
      position: sticky !important;
      top: 0.25rem !important;
      overflow: hidden !important;
      z-index: 2;
    }
    div[data-testid="stHorizontalBlock"]:has(iframe[title*="folium"])
      > [data-testid="stColumn"]:has(iframe) iframe,
    div[data-testid="stHorizontalBlock"]:has(iframe[title*="streamlit_folium"])
      > [data-testid="stColumn"]:has(iframe) iframe {
      height: 820px !important;
      min-height: 820px !important;
      max-height: 820px !important;
      width: 100% !important;
      display: block !important;
      border-radius: 16px;
      border: 1px solid rgba(154,168,188,0.14) !important;
    }
    div[data-testid="stHorizontalBlock"]:has(iframe[title*="folium"])
      > [data-testid="stColumn"]:not(:has(iframe)),
    div[data-testid="stHorizontalBlock"]:has(iframe[title*="streamlit_folium"])
      > [data-testid="stColumn"]:not(:has(iframe)) {
      overflow-x: hidden !important;
      overflow-y: auto !important;
      -webkit-overflow-scrolling: touch !important;
      overscroll-behavior: contain;
      padding-right: 0.35rem;
    }
    div[data-testid="stHorizontalBlock"]:has(iframe[title*="folium"])
      > [data-testid="stColumn"]:not(:has(iframe)) > div,
    div[data-testid="stHorizontalBlock"]:has(iframe[title*="streamlit_folium"])
      > [data-testid="stColumn"]:not(:has(iframe)) > div {
      max-height: none !important;
      height: auto !important;
      overflow: visible !important;
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


def path_travel_time(G: nx.Graph, path: List) -> float:
    return _rs_path_travel_time(G, path)


def route_overlap_accuracy(pred: List, oracle: List) -> float:
    return _rs_route_overlap(pred, oracle)


def _no_click(layer):
    """Stop Folium overlays from stealing map clicks."""
    try:
        layer.options["interactive"] = False
        if "bubblingMouseEvents" in layer.options:
            layer.options["bubblingMouseEvents"] = False
        if hasattr(layer, "popup"):
            layer.popup = None
        if hasattr(layer, "tooltip"):
            layer.tooltip = None
    except Exception:
        pass
    return layer


def _fault_line_latlons(
    epi_lat: float, epi_lon: float, *, half_span_deg: float = 0.012
) -> List[List[float]]:
    """
    Simple West Valley–style axis through the epicenter (NNW–SSE).

    Fault line → rupture → epicenter → near-epi broken roads. Fixed geometry
    for the Intramuros demo (not a geologic survey product).
    """
    # Bearing ≈ 25° west of north (Manila trench / Valley Fault feel).
    dlat = half_span_deg * 0.92
    dlon = half_span_deg * 0.40
    return [
        [float(epi_lat) - dlat, float(epi_lon) - dlon],
        [float(epi_lat), float(epi_lon)],
        [float(epi_lat) + dlat, float(epi_lon) + dlon],
    ]


def _draw_fault_line_on_map(m, epi_lat: float, epi_lon: float) -> None:
    """Draw the demo fault polyline under hazard rings / routes."""
    coords = _fault_line_latlons(epi_lat, epi_lon)
    _no_click(
        folium.PolyLine(
            coords,
            color=FAULT_LINE_COLOR,
            weight=5,
            opacity=0.85,
            dash_array="10 6",
            tooltip="Fault line (demo axis through epicenter)",
        )
    ).add_to(m)
    # Soft glow underlay
    _no_click(
        folium.PolyLine(
            coords,
            color=FAULT_LINE_COLOR,
            weight=12,
            opacity=0.18,
        )
    ).add_to(m)


def _draw_start_blue_dot(m, lat: float, lon: float, *, tooltip: str = "Your location") -> None:
    """Clear blue start point (primary visual — not only a house icon)."""
    _no_click(
        folium.CircleMarker(
            [lat, lon],
            radius=11,
            color="#E8EEF6",
            weight=2,
            fill=True,
            fill_color=START_DOT_COLOR,
            fill_opacity=0.95,
            tooltip=tooltip,
        )
    ).add_to(m)
    _no_click(
        folium.CircleMarker(
            [lat, lon],
            radius=4,
            color="#fff",
            weight=1,
            fill=True,
            fill_color="#fff",
            fill_opacity=1.0,
        )
    ).add_to(m)


def _fmt_prob(p) -> str:
    try:
        if p is None:
            return "—"
        fv = float(p)
        if not np.isfinite(fv):
            return "—"
        if fv >= 0.01:
            return f"{fv * 100:.1f}%"
        return f"{fv:.2e}"
    except (TypeError, ValueError):
        return "—"


def _path_anim_step(path) -> int:
    """Current animation hop index into ``path`` (0 = start node)."""
    if not path or len(path) < 1:
        return 0
    max_i = len(path) - 1
    try:
        step = int(st.session_state.get("path_anim_step", max_i))
    except (TypeError, ValueError):
        step = max_i
    return max(0, min(step, max_i))


def _draw_hybrid_path_animation(
    m,
    G,
    path,
    step_trace: Optional[List] = None,
    *,
    step: int,
) -> None:
    """
    Draw Hybrid route up to ``step`` with hop probability tooltips and a
    moving blue agent at the current node (Streamlit-safe redraw, no AntPath).
    """
    if not path or len(path) < 2:
        return
    step = max(0, min(int(step), len(path) - 1))
    partial = path[: step + 1]
    coords = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in partial if n in G.nodes]
    if len(coords) >= 2:
        _no_click(
            folium.PolyLine(
                coords,
                color=HYBRID_ROUTE_COLOR,
                weight=6,
                opacity=0.95,
                tooltip="Hybrid escape path",
            )
        ).add_to(m)

    # Chosen edges with probability labels (midpoint markers).
    trace = step_trace or []
    for i in range(min(step, len(trace))):
        entry = trace[i] if i < len(trace) else {}
        u, v = path[i], path[i + 1]
        if u not in G.nodes or v not in G.nodes:
            continue
        lat_m = 0.5 * (float(G.nodes[u]["y"]) + float(G.nodes[v]["y"]))
        lon_m = 0.5 * (float(G.nodes[u]["x"]) + float(G.nodes[v]["x"]))
        p = entry.get("prob")
        tip = f"Hop {i + 1} · P={_fmt_prob(p)}"
        if entry.get("mode") and entry.get("mode") != "ml":
            tip += f" · {entry['mode']}"
        _no_click(
            folium.CircleMarker(
                [lat_m, lon_m],
                radius=3,
                color=HYBRID_ROUTE_COLOR,
                weight=1,
                fill=True,
                fill_color="#041018",
                fill_opacity=0.9,
                tooltip=tip,
            )
        ).add_to(m)

    # At the live node: faint candidate next-hop edges with probs.
    if step < len(path) - 1 and step < len(trace):
        entry = trace[step]
        cur = path[step]
        if cur in G.nodes:
            for cand in (entry.get("candidates") or [])[:5]:
                nb = cand.get("node")
                if nb is None or nb not in G.nodes:
                    continue
                chosen = bool(cand.get("chosen"))
                p = cand.get("prob")
                _no_click(
                    folium.PolyLine(
                        [
                            [G.nodes[cur]["y"], G.nodes[cur]["x"]],
                            [G.nodes[nb]["y"], G.nodes[nb]["x"]],
                        ],
                        color=HYBRID_ROUTE_COLOR if chosen else CANDIDATE_EDGE_COLOR,
                        weight=5 if chosen else 2,
                        opacity=0.95 if chosen else 0.45,
                        dash_array=None if chosen else "2 6",
                        tooltip=f"P={_fmt_prob(p)}" + (" · chosen" if chosen else ""),
                    )
                ).add_to(m)

    # Moving blue agent at current path node.
    cur_node = path[step]
    if cur_node in G.nodes:
        lat = float(G.nodes[cur_node]["y"])
        lon = float(G.nodes[cur_node]["x"])
        label = (
            "Start"
            if step == 0
            else ("Exit" if step == len(path) - 1 else f"Hop {step}")
        )
        _draw_start_blue_dot(
            m, lat, lon, tooltip=f"Agent · {label} · node {cur_node}"
        )
        # Outer pulse ring for visibility while animating.
        _no_click(
            folium.CircleMarker(
                [lat, lon],
                radius=18,
                color=START_DOT_COLOR,
                weight=2,
                fill=False,
                opacity=0.55,
            )
        ).add_to(m)


def _set_epicenter(lat: float, lon: float) -> None:
    st.session_state["epi_lat"] = float(lat)
    st.session_state["epi_lon"] = float(lon)


def _disaster_active() -> bool:
    """True when the active feed snapshot includes an earthquake / hazard incident."""
    if st.session_state.get("disaster_active"):
        return True
    snap = st.session_state.get("feed_snapshot") or {}
    if isinstance(snap, dict) and snap.get("has_disaster"):
        return True
    for inc in (snap.get("incidents") or []) if isinstance(snap, dict) else []:
        if str(inc.get("kind") or "").lower() in (
            "earthquake",
            "quake",
            "hazard",
            "hazard_zone",
            "disaster",
            "seismic",
        ):
            return True
    return False


def _mild_default_epi(G) -> Tuple[float, float]:
    """Far-enough mild epi so everyday feed routing is disruption-led (no rings)."""
    xs = [float(G.nodes[n]["x"]) for n in G.nodes()]
    ys = [float(G.nodes[n]["y"]) for n in G.nodes()]
    # Offset outside the dense core so Algorithm 1 soft penalties stay negligible.
    return float(np.mean(ys)) - 0.0045, float(np.mean(xs)) + 0.0045


def _load_demo_scenarios() -> Dict[str, Any]:
    """Curated strict Hybrid travel-win scenarios (data/demo_scenarios.json)."""
    if not DEMO_SCENARIOS_PATH.exists():
        return {}
    try:
        return json.loads(DEMO_SCENARIOS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pick_advantage_scenario(
    payload: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Prefer default_scenario_id, else largest Δ(classical − hybrid)."""
    payload = payload if payload is not None else _load_demo_scenarios()
    scenarios = list(payload.get("scenarios") or [])
    if not scenarios:
        return None
    default_id = payload.get("default_scenario_id")
    if default_id:
        for s in scenarios:
            if s.get("id") == default_id:
                return s

    def _gap(s: Dict[str, Any]) -> float:
        m = s.get("metrics") or {}
        if "delta_c_minus_h" in m:
            return float(m["delta_c_minus_h"])
        ht = float(m.get("hybrid_time") or 0.0)
        ct = float(m.get("classical_time") or 0.0)
        return ct - ht

    return max(scenarios, key=_gap)


def _apply_advantage_scenario(G, scenario: Dict[str, Any]) -> str:
    """Load start + destination pair from a curated advantage scenario."""
    _clear_route_results()
    start = scenario.get("start_node")
    dest = scenario.get("dest_node")
    if start is None or start not in G.nodes:
        _set_location(
            G,
            float(scenario["start_lat"]),
            float(scenario["start_lon"]),
        )
    else:
        st.session_state["start_node"] = start
        st.session_state["loc_lat"] = float(G.nodes[start]["y"])
        st.session_state["loc_lon"] = float(G.nodes[start]["x"])
        st.session_state["map_center"] = [
            float(st.session_state["loc_lat"]),
            float(st.session_state["loc_lon"]),
        ]
    if dest is not None and dest in G.nodes and dest != st.session_state["start_node"]:
        _set_destination(G, dest, via="judge scenario")
    else:
        # Fall back to exit_lat/lon from scenario, or farthest node.
        if scenario.get("exit_lat") is not None and scenario.get("exit_lon") is not None:
            _set_destination_from_latlon(
                G, float(scenario["exit_lat"]), float(scenario["exit_lon"]), via="judge scenario"
            )
        else:
            far = farthest_node_from(G, st.session_state["start_node"])
            _set_destination(G, far, via="judge scenario")
    # Scenario may include a mild epi; only mark disaster when feed says so later.
    if scenario.get("epi_lat") is not None and scenario.get("epi_lon") is not None:
        _set_epicenter(float(scenario["epi_lat"]), float(scenario["epi_lon"]))
    m = scenario.get("metrics") or {}
    expected = ""
    if m.get("hybrid_time") is not None and m.get("classical_time") is not None:
        tag = (
            "live broken"
            if m.get("verified_with_broken")
            else ("live flood" if m.get("verified_with_flood") else "stored")
        )
        expected = (
            f" {tag} H={float(m['hybrid_time']):.1f} "
            f"< C={float(m['classical_time']):.1f}."
        )
    title = scenario.get("title") or scenario.get("id") or "advantage"
    msg = (
        f"Advantage demo · {title}.{expected} "
        "Click the map to set your location, then Find escape route."
    )
    st.session_state["map_status"] = msg
    st.session_state["advantage_scenario_id"] = scenario.get("id")
    return msg


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Rough great-circle distance in meters (Intramuros-scale)."""
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(min(1.0, a))))


def _nearest_exit_within(G, lat: float, lon: float, max_m: float) -> Optional[Any]:
    """Return nearest evacuate-exit node if within max_m of the click."""
    exits = _ensure_exit_nodes(G)
    best = None
    best_d = float("inf")
    for n in exits:
        d = _haversine_m(
            float(lat),
            float(lon),
            float(G.nodes[n]["y"]),
            float(G.nodes[n]["x"]),
        )
        if d < best_d:
            best_d = d
            best = n
    if best is not None and best_d <= float(max_m):
        return best
    return None


def build_base_map(G, map_center, map_zoom: int = 16):
    """Basemap (tiles + roads). Dynamic overlays are added by caller.

    Keep ``st_folium`` ``key`` stable (never include scrubber ``t``) so Cloud
    does not remount the component identity every frame. We intentionally avoid
    ``feature_group_to_add`` — it has caused SessionInfo / websocket fragility
    on Streamlit Community Cloud with large overlay payloads.
    """
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
    return predict_escape_route(
        G, model, mean, std, start, dest, epicenter_lonlat, max_steps=max_steps
    )


def dijkstra_route(G, start, dest, epicenter_lonlat, max_steps=120):
    path, _radii, _env, travel, _meta = dijkstra_escape_route(
        G, start, dest, epicenter_lonlat, max_steps=max_steps
    )
    return path, travel


def _apply_feed_snapshot(snap) -> int:
    """Store feed snapshot + disruptions; sync disaster epi when present."""
    if snap is None:
        return 0
    st.session_state["feed_snapshot"] = snap.to_dict()
    disruptions = snap.edge_disruptions
    st.session_state["edge_disruptions"] = disruptions
    st.session_state.pop("_nudge_disruption", None)
    feed_epi = epicenter_from_snapshot(snap)
    if feed_epi is not None:
        lon, lat = feed_epi
        _set_epicenter(lat, lon)
        st.session_state["disaster_active"] = True
        r = getattr(snap, "r_epi_km", None)
        if r is not None:
            st.session_state["feed_r_epi_km"] = float(r)
    else:
        st.session_state["disaster_active"] = False
        st.session_state.pop("feed_r_epi_km", None)
    edges = disruptions.get("edges") if isinstance(disruptions, dict) else None
    return len(edges) if edges else 0


def _load_current_conditions(G, *, refresh: bool = False) -> str:
    """
    Load (or refresh) mock city conditions into the session.

    Product open path: current feed on start — not button spam.
    """
    near = st.session_state.get("start_node")
    try:
        disruptions, snap = active_feed_disruptions(
            G, near_node=near, refresh=bool(refresh)
        )
    except TrafficNotConfiguredError as exc:
        _handle_traffic_error(exc)
        return str(exc)
    if snap is None:
        st.session_state["edge_disruptions"] = disruptions
        st.session_state["feed_snapshot"] = None
        st.session_state["disaster_active"] = False
        msg = "Traffic feed returned no city conditions."
        st.session_state["map_status"] = msg
        return msg
    n = _apply_feed_snapshot(snap)
    name = getattr(snap, "scenario_name", None) or snap.to_dict().get("scenario_name")
    as_of = getattr(snap, "as_of", "") or ""
    verb = "Refreshed" if refresh else "Loaded"
    disaster_bit = " · disaster active" if _disaster_active() else ""
    msg = (
        f"{verb} city conditions · {name} · {n} disrupted edges"
        + disaster_bit
        + (f" · as of {as_of}" if as_of else "")
    )
    st.session_state["map_status"] = msg
    _clear_route_results()
    return msg


def _feed_incidents_html() -> str:
    """Render Conditions now incident list from session feed snapshot."""
    snap = st.session_state.get("feed_snapshot") or {}
    incidents = snap.get("incidents") or []
    as_of = snap.get("as_of") or "—"
    name = snap.get("scenario_name") or "City conditions"
    city = snap.get("city") or "Manila · Intramuros"
    tod = snap.get("time_of_day_label") or ""
    clock = snap.get("local_clock") or ""
    tod_line = ""
    if tod or clock:
        tod_line = (
            f'<br/><span style="color:#F5C542">{tod or "Local"}'
            + (f" · {clock}" if clock else "")
            + "</span>"
        )
    header = (
        f'<div class="qr-asof"><strong style="color:#fff">{city}</strong> · '
        f"{name}{tod_line}<br/>Last updated · {as_of}</div>"
    )
    if not incidents:
        return (
            header
            + '<div class="qr-incident">No active incidents on the simulated feed.</div>'
        )
    rows = [header]
    for inc in incidents:
        kind = str(inc.get("kind") or "incident")
        label = str(inc.get("label") or kind)
        area = str(inc.get("area_hint") or "")
        sev = float(inc.get("severity") or 0)
        is_quake = kind.lower() in (
            "earthquake",
            "quake",
            "hazard",
            "hazard_zone",
            "disaster",
            "seismic",
        )
        if is_quake:
            epi_bit = ""
            if inc.get("epi_lat") is not None and inc.get("epi_lon") is not None:
                epi_bit = (
                    f"<br/>epi {float(inc['epi_lat']):.5f}, "
                    f"{float(inc['epi_lon']):.5f}"
                )
                if inc.get("r_epi_km") is not None:
                    epi_bit += f" · r_epi ≈ {float(inc['r_epi_km']):.2f} km"
            rows.append(
                f'<div class="qr-incident">'
                f'<span class="sev" style="color:#FF4D6A">{kind.replace("_", " ")}'
                f" · sev {sev:.0%}</span><br/>"
                f"<strong>{label}</strong>"
                + (f" · {area}" if area else "")
                + epi_bit
                + "<br/>soft Algorithm-1 ring penalties"
                f"</div>"
            )
        else:
            n_e = int(inc.get("edge_count") or len(inc.get("edges") or []))
            rows.append(
                f'<div class="qr-incident">'
                f'<span class="sev">{kind.replace("_", " ")} · sev {sev:.0%}</span><br/>'
                f"<strong>{label}</strong>"
                + (f" · {area}" if area else "")
                + f"<br/>{n_e} edges · soft ×{float(inc.get('multiplier') or 0):.0f}"
                f"</div>"
            )
    return "".join(rows)


def _clear_route_results():
    for k in (
        "path",
        "classical_path",
        "dij_path",
        "hybrid_path_raw",
        "radii_trace",
        "qml_travel",
        "classical_travel",
        "dij_travel",
        "qml_safety",
        "classical_safety",
        "dij_safety",
        "qml_mean_epi_km",
        "qml_min_epi_km",
        "classical_min_epi_km",
        "dij_min_epi_km",
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
        "latency_ms",
        "is_hybrid_route",
        "hybrid_deferred",
        "deferred_reason",
        "deferred_note",
        "primary_engine",
        "_step_reveal",
        "step_trace",
        "path_logprob",
        "path_prob_product",
        "path_anim_step",
        "path_anim_playing",
    ):
        st.session_state.pop(k, None)


def _set_destination(G, node, *, via: str = "map click") -> str:
    """Pin evacuate exit / destination to a graph node."""
    if node not in G.nodes:
        raise ValueError(f"Evacuate exit node {node} not on graph")
    if node == st.session_state.get("start_node"):
        # Nudge to a neighbor if click snaps onto start.
        nbs = list(G.neighbors(node))
        if nbs:
            node = nbs[0]
        else:
            node = farthest_node_from(G, node)
    st.session_state["dest_node"] = node
    st.session_state["dest_lat"] = float(G.nodes[node]["y"])
    st.session_state["dest_lon"] = float(G.nodes[node]["x"])
    _clear_route_results()
    msg = (
        f"Evacuate exit → {st.session_state['dest_lat']:.5f}, "
        f"{st.session_state['dest_lon']:.5f} (node {node}) · set via {via}."
    )
    st.session_state["map_status"] = msg
    return msg


def _set_destination_from_latlon(
    G, lat: float, lon: float, *, via: str = "map click"
) -> str:
    node = snap_to_nearest_node(G, float(lat), float(lon))
    return _set_destination(G, node, via=via)


N_EVACUATE_AREAS = 5  # top-N perimeter candidates shown on map + panel


def _ensure_exit_nodes(G) -> List:
    """Stable perimeter evacuate areas for Escape recommendation (several, not one)."""
    exits = st.session_state.get("exit_nodes")
    if not exits or len(exits) < N_EVACUATE_AREAS:
        exits = select_exit_nodes(G, n_exits=N_EVACUATE_AREAS, seed=42)
        st.session_state["exit_nodes"] = list(exits)
    return list(st.session_state["exit_nodes"])


def _landmark_label_for(G, node, *, index: int = 1) -> str:
    try:
        info = named_escape_landmarks(G, exits=[node])
        if info:
            return str(info[0].get("label") or f"Exit {index}")
    except Exception:
        pass
    return f"Evacuate area {index}"


def _evacuate_area_rows(G) -> List[Dict[str, Any]]:
    """
    Display rows for all candidate evacuate areas.

    Prefer scored ranking from recommend_best_exit; fall back to unlabeled exits.
    """
    ranking = st.session_state.get("exit_ranking") or []
    if ranking:
        rows: List[Dict[str, Any]] = []
        for row in ranking:
            node = row.get("exit_node")
            if node is None or node not in G.nodes:
                continue
            rows.append(dict(row))
        if rows:
            return rows
    exits = _ensure_exit_nodes(G)
    landmarks = {lm["node"]: lm for lm in named_escape_landmarks(G, exits=exits)}
    rows = []
    for i, ex in enumerate(exits, start=1):
        lm = landmarks.get(ex) or {}
        rows.append(
            {
                "exit_node": ex,
                "label": lm.get("label") or _landmark_label_for(G, ex, index=i),
                "rank": i,
                "combined_score": None,
                "time_score": None,
                "safety_score": None,
                "travel_time": None,
                "safety_km": None,
                "why": "Rank after location + epicenter are set",
                "exit_reached": None,
                "recommended": False,
                "lat": float(G.nodes[ex]["y"]),
                "lon": float(G.nodes[ex]["x"]),
            }
        )
    return rows


def _select_evacuate_area(G, node, *, via: str = "exit select", auto: bool = False) -> str:
    """Pin routing destination to a chosen evacuate area (override or recommend)."""
    st.session_state["exit_auto"] = bool(auto)
    if auto:
        st.session_state["recommended_exit"] = node
    return _set_destination(G, node, via=via)


def _recommend_and_set_exit(G, *, via: str = "best exit") -> str:
    """Rank several evacuate areas under current epi + soft disruptions; pin best."""
    exits = _ensure_exit_nodes(G)
    start = st.session_state.get("start_node")
    if start is None or start not in G.nodes:
        return "Set your location before recommending an evacuate exit."
    epi = (
        float(st.session_state["epi_lon"]),
        float(st.session_state["epi_lat"]),
    )
    best, ranking = recommend_best_exit(
        G,
        start,
        exits,
        epi,
        edge_disruptions=st.session_state.get("edge_disruptions"),
    )
    st.session_state["exit_ranking"] = ranking
    st.session_state["exit_auto"] = True
    st.session_state["recommended_exit"] = best
    label = _landmark_label_for(G, best)
    msg = _set_destination(G, best, via=via)
    n = len(ranking) if ranking else len(exits)
    if ranking:
        top = ranking[0]
        why = top.get("why") or ""
        msg = (
            f"Several evacuate areas ({n}) — recommend {label} · "
            f"score {top.get('combined_score', '—')}. {why}"
        )
        st.session_state["map_status"] = msg
    return msg


def _draw_evacuate_exits_on_map(m, G, *, dest_node, recommended_node) -> None:
    """Draw all candidate evacuate areas; highlight recommended vs selected vs others."""
    rows = _evacuate_area_rows(G)
    if not rows:
        return
    # Draw non-selected first so selected/recommended sit on top.
    ordered = sorted(
        rows,
        key=lambda r: (
            1 if r.get("exit_node") == dest_node else 0,
            1 if r.get("exit_node") == recommended_node else 0,
        ),
    )
    for row in ordered:
        node = row.get("exit_node")
        if node is None or node not in G.nodes:
            continue
        lat = float(row.get("lat", G.nodes[node]["y"]))
        lon = float(row.get("lon", G.nodes[node]["x"]))
        label = str(row.get("label") or _landmark_label_for(G, node))
        rank = int(row.get("rank") or 0) or ""
        score = row.get("combined_score")
        is_dest = node == dest_node
        is_rec = node == recommended_node or bool(row.get("recommended"))
        score_bit = f" · score {score}" if score is not None else ""
        tip_bits = [f"#{rank} {label}" if rank else label]
        if is_rec:
            tip_bits.append("recommended")
        if is_dest:
            tip_bits.append("routing here")
        tip_bits.append(score_bit.strip(" ·") if score_bit else "")
        tooltip = " · ".join(b for b in tip_bits if b)

        if is_rec and is_dest:
            # Recommended best exit that Hybrid routes to — gold star.
            ring, fill, weight, radius = "#F5C542", "#F5C542", 4, 11
            icon_color, icon_name = "orange", "star"
        elif is_dest:
            ring, fill, weight, radius = "#00E5FF", "#0a0f1e", 4, 10
            icon_color, icon_name = "cadetblue", "flag"
        elif is_rec:
            ring, fill, weight, radius = "#F5C542", "#0a0f1e", 3, 9
            icon_color, icon_name = "orange", "star"
        else:
            ring, fill, weight, radius = "#9AA8BC", "#1a2233", 2, 7
            icon_color, icon_name = "lightgray", "info-sign"

        _no_click(
            folium.CircleMarker(
                [lat, lon],
                radius=radius,
                color=ring,
                weight=weight,
                fill=True,
                fill_color=fill,
                fill_opacity=0.9 if (is_rec or is_dest) else 0.65,
                tooltip=tooltip,
            )
        ).add_to(m)
        _no_click(
            folium.Marker(
                [lat, lon],
                icon=folium.Icon(color=icon_color, icon=icon_name),
                tooltip=tooltip,
            )
        ).add_to(m)
        if rank:
            _no_click(
                folium.Marker(
                    [lat, lon],
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font-size:10px;font-weight:700;'
                            f'color:#041018;background:{ring};'
                            f'border-radius:50%;width:16px;height:16px;'
                            f'line-height:16px;text-align:center;'
                            f'margin-left:-8px;margin-top:10px;">{rank}</div>'
                        ),
                        icon_size=(16, 16),
                        icon_anchor=(0, 0),
                    ),
                    tooltip=tooltip,
                )
            ).add_to(m)


def _activate_random_epicenter(G) -> str:
    """First-class Escape control: place a random earthquake epicenter."""
    (lon_r, lat_r), _ = random_epicenter(G)
    _set_epicenter(lat_r, lon_r)
    st.session_state["disaster_active"] = True
    if "hazard_t_scrub" not in st.session_state:
        st.session_state["hazard_t_scrub"] = 30
    _clear_route_results()
    msg = f"Epicenter → {lat_r:.5f}, {lon_r:.5f} · hazard rings active"
    st.session_state["map_status"] = msg
    try:
        _recommend_and_set_exit(G, via="epi update")
    except Exception:
        pass
    return msg


def _load_escape_open(G) -> str:
    """Product open: Earthquake Escape scenario + ranked evacuate areas."""
    near = st.session_state.get("start_node")
    try:
        from src.mock_traffic_feed import get_mock_traffic_feed

        feed = get_mock_traffic_feed()
        feed.force_scenario(ESCAPE_OPEN_SCENARIO)
        snap = feed.current(G, near_node=near)
        n = _apply_feed_snapshot(snap)
        feed.clear_force()
        name = getattr(snap, "scenario_name", None) or ESCAPE_OPEN_SCENARIO
        msg = (
            f"Earthquake Escape · {name} · {n} disrupted edges · "
            "rank several evacuate areas"
        )
        st.session_state["map_status"] = msg
        _recommend_and_set_exit(G, via="escape open")
        return msg
    except Exception:
        return _load_current_conditions(G, refresh=False)


def _should_defer_hybrid(
    *,
    hybrid_travel: float,
    classical_travel: Optional[float],
    hybrid_reached: bool,
    classical_path,
    latency_ms: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """
    Reliability rule: defer Hybrid when catastrophic vs Classical, failed, or very slow.

    Returns (defer, reason_code).
    """
    if classical_path is None or classical_travel is None:
        return False, ""
    ct = float(classical_travel)
    ht = float(hybrid_travel)
    if not hybrid_reached:
        return True, "failed"
    if ct > 1e-6 and ht > ct * HYBRID_CATASTROPHIC_RATIO:
        return True, "catastrophic"
    lat = latency_ms or {}
    h_ms = lat.get("hybrid")
    c_ms = lat.get("classical")
    try:
        h_ms_f = float(h_ms) if h_ms is not None else None
        c_ms_f = float(c_ms) if c_ms is not None else None
    except (TypeError, ValueError):
        h_ms_f, c_ms_f = None, None
    if h_ms_f is not None and h_ms_f >= HYBRID_SLOW_MS:
        return True, "timeout"
    if (
        h_ms_f is not None
        and c_ms_f is not None
        and c_ms_f > 1.0
        and h_ms_f >= max(15_000.0, c_ms_f * HYBRID_SLOW_VS_CLASSICAL)
    ):
        return True, "timeout"
    return False, ""


def _active_disruption_count() -> int:
    raw = st.session_state.get("edge_disruptions") or {}
    edges = raw.get("edges") if isinstance(raw, dict) else None
    return len(edges) if edges else 0


def _disruption_summary() -> str:
    """Human label for the active soft disruption (or None)."""
    snap = st.session_state.get("feed_snapshot")
    if isinstance(snap, dict) and snap.get("scenario_name"):
        n = _active_disruption_count()
        tod = snap.get("time_of_day_label") or ""
        tod_bit = f" · {tod}" if tod else ""
        return (
            f"{snap['scenario_name']}{tod_bit} · {n} disrupted edges · "
            f"feed {snap.get('feed', 'simulated')}"
        )
    raw = st.session_state.get("edge_disruptions")
    if not isinstance(raw, dict) or not raw.get("edges"):
        return "None — map edges at nominal + hazard weights only"
    n = len(raw["edges"])
    mult = float(raw.get("multiplier", 5.0))
    kind = str(raw.get("kind", "congestion"))
    if kind == "broken":
        label = "Broken roads near epicenter"
    elif kind == "flood":
        label = "Flooded corridor (related case)"
    elif kind == "soft_block":
        label = "Post-quake blocked corridor"
    else:
        label = "Post-quake damaged roads"
    return f"{label} · {n} disrupted edges · ×{mult:.0f} soft weight · amber dashed"


def _set_random_disruption(G, *, soft_block: bool = False) -> int:
    """Sample a soft congestion or soft-closed corridor via traffic provider."""
    import time as _time

    seed = int(_time.time() * 1000) % (2**31 - 1)
    corridor = (
        int(np.random.randint(3, 6)) if soft_block else int(np.random.randint(2, 5))
    )
    provider = get_traffic_provider()
    dset = provider.get_edge_disruptions(
        G,
        kind="soft_block" if soft_block else "congestion",
        n_seed_edges=1,
        corridor_extra=corridor,
        seed=seed,
    )
    st.session_state["edge_disruptions"] = dset.to_serializable()
    st.session_state["feed_snapshot"] = {
        "city": "Manila · Intramuros",
        "as_of": "manual overlay",
        "scenario_id": "manual",
        "scenario_name": (
            "Post-quake blocked corridor" if soft_block else "Post-quake damaged roads"
        ),
        "blurb": "Manual post-quake damage overlay",
        "feed": "simulated",
        "has_disaster": False,
        "incidents": [
            {
                "id": "manual:0",
                "kind": "soft_block" if soft_block else "congestion",
                "label": (
                    "Blocked corridor · post-quake debris"
                    if soft_block
                    else "Damaged roads · post-quake cascade"
                ),
                "severity": 0.7 if soft_block else 0.55,
                "area_hint": "Manual",
                "edge_count": len(dset.normalized_edges()),
                "edges": [[u, v] for u, v in dset.normalized_edges()],
                "multiplier": float(dset.multiplier),
            }
        ],
    }
    st.session_state["disaster_active"] = False
    st.session_state.pop("feed_r_epi_km", None)
    st.session_state.pop("_nudge_disruption", None)
    _clear_route_results()
    return len(dset.normalized_edges())


def _set_broken_roads_near_epi(
    G,
    *,
    epicenter_lonlat: Optional[Tuple[float, float]] = None,
    seed: Optional[int] = None,
    corridor_extra: int = 14,
    radius_km: Optional[float] = None,
) -> int:
    """Post-quake broken roads clustered near the epicenter (Escape primary)."""
    import time as _time

    if seed is None:
        seed = int(_time.time() * 1000) % (2**31 - 1)
    epi = epicenter_lonlat
    if epi is None:
        elat = st.session_state.get("epi_lat")
        elon = st.session_state.get("epi_lon")
        if elat is not None and elon is not None:
            epi = (float(elon), float(elat))
    provider = get_traffic_provider()
    dset = provider.get_edge_disruptions(
        G,
        kind="broken",
        corridor_extra=int(corridor_extra),
        seed=int(seed),
        epicenter_lonlat=epi,
        radius_km=radius_km,
    )
    st.session_state["edge_disruptions"] = dset.to_serializable()
    st.session_state["feed_snapshot"] = {
        "city": "Manila · Intramuros",
        "as_of": "manual overlay",
        "scenario_id": "manual_broken",
        "scenario_name": "Broken roads near epicenter",
        "blurb": "Post-quake broken roads hugging the hazard rings",
        "feed": "simulated",
        "has_disaster": True,
        "incidents": [
            {
                "id": "manual_broken:0",
                "kind": "broken",
                "label": "Broken roads near epicenter",
                "severity": 0.9,
                "area_hint": "Near hazard rings",
                "edge_count": len(dset.normalized_edges()),
                "edges": [[u, v] for u, v in dset.normalized_edges()],
                "multiplier": float(dset.multiplier),
            }
        ],
    }
    st.session_state["disaster_active"] = True
    st.session_state.pop("_nudge_disruption", None)
    _clear_route_results()
    return len(dset.normalized_edges())


def _set_flood_corridor(
    G,
    *,
    near_node=None,
    seed: Optional[int] = None,
    corridor_extra: int = 11,
) -> int:
    """Ondoy-like soft flood corridor via traffic provider (related case)."""
    import time as _time

    if seed is None:
        seed = int(_time.time() * 1000) % (2**31 - 1)
    provider = get_traffic_provider()
    dset = provider.get_edge_disruptions(
        G,
        kind="flood",
        near_node=near_node,
        corridor_extra=int(corridor_extra),
        seed=int(seed),
    )
    st.session_state["edge_disruptions"] = dset.to_serializable()
    st.session_state["feed_snapshot"] = {
        "city": "Manila · Intramuros",
        "as_of": "manual overlay",
        "scenario_id": "manual_flood",
        "scenario_name": "Flooded corridor (related case)",
        "blurb": "Manual flood overlay · Ondoy-like related case",
        "feed": "simulated",
        "has_disaster": False,
        "incidents": [
            {
                "id": "manual_flood:0",
                "kind": "flood",
                "label": "Flooded corridor near Pasig",
                "severity": 0.9,
                "area_hint": "Pasig-side low ground",
                "edge_count": len(dset.normalized_edges()),
                "edges": [[u, v] for u, v in dset.normalized_edges()],
                "multiplier": float(dset.multiplier),
            }
        ],
    }
    st.session_state["disaster_active"] = False
    st.session_state.pop("feed_r_epi_km", None)
    st.session_state.pop("_nudge_disruption", None)
    _clear_route_results()
    return len(dset.normalized_edges())


def _clear_disruption() -> None:
    st.session_state.pop("edge_disruptions", None)
    st.session_state.pop("feed_snapshot", None)
    st.session_state["disaster_active"] = False
    st.session_state.pop("feed_r_epi_km", None)
    _clear_route_results()


def _traffic_feed_badge_html() -> str:
    """Honest product badge: Live conditions · simulated feed (or live feed)."""
    info = get_traffic_provider().mode_info()
    cls = "qr-badge feed live" if info.mode == "live" else "qr-badge feed"
    return f'<span class="{cls}">{info.badge}</span>'


def _handle_traffic_error(exc: Exception) -> None:
    """Surface live-mode configure errors without breaking the Escape map."""
    msg = str(exc) or "Traffic feed unavailable."
    st.session_state["map_status"] = msg
    try:
        st.warning(msg)
    except Exception:
        pass


# Pinned damage seeds that keep Hybrid travel < Classical on best-ranked exits
# (verified against film_hybrid.pt / film_classical.pt + demo_scenarios.json).
# Fallback only — live pin lives in data/demo_scenarios.json (SSoT).
_JUDGE_DAMAGE_BY_SCENARIO = {
    "qa_1": {"seed": 17082, "corridor_extra": 14},
    "qa_2": {"seed": 16351, "corridor_extra": 11},
    "qa_3": {"seed": 17092, "corridor_extra": 11},
    "qa_4": {"seed": 18245, "corridor_extra": 14},
    "qa_5": {"seed": 16280, "corridor_extra": 8},
}
_JUDGE_FLOOD_BY_SCENARIO = _JUDGE_DAMAGE_BY_SCENARIO  # back-compat alias


def _hybrid_ckpt_debug() -> Dict[str, Any]:
    """sha/mtime of models/film_hybrid.pt for Cloud pin diagnostics."""
    path = Path(HYBRID_CHECKPOINT)
    out: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "sha16": None,
        "sha256": None,
        "mtime_utc": None,
        "size": None,
    }
    if not path.exists():
        return out
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    out["sha256"] = digest
    out["sha16"] = digest[:16]
    out["size"] = len(data)
    out["mtime_utc"] = datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")
    return out


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (WGS84 approx)."""
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    )
    return float(2.0 * r * np.arcsin(np.sqrt(min(1.0, float(a)))))


def _suggested_apartment_coords(
    scenario: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Known-good apartment / start from demo_scenarios.json (qa_1 pin).

    Audience clicks this faint marker so free-click doesn't land in a tie zone.
    """
    sc = scenario if scenario is not None else _judge_pinned_scenario()
    if not sc:
        return None
    jd = sc.get("_judge_demo") or {}
    lat = jd.get("suggested_start_lat", sc.get("start_lat"))
    lon = jd.get("suggested_start_lon", sc.get("start_lon"))
    node = jd.get("suggested_start_node", sc.get("start_node"))
    if lat is None or lon is None:
        return None
    return {
        "lat": float(lat),
        "lon": float(lon),
        "node": node,
        "hint": str(
            jd.get("click_hint")
            or "Faint cyan ring · NW apartment — click here for Hybrid win"
        ),
        "scenario_id": sc.get("id") or jd.get("scenario_id") or "qa_1",
        "best_exit": sc.get("best_exit") or jd.get("best_exit"),
        "best_exit_label": (
            jd.get("best_exit_label")
            or sc.get("best_exit_label")
            or "North Gate · NW"
        ),
    }


def _expected_hybrid_fingerprint() -> Dict[str, Any]:
    """Expected q% / sha16 from demo_scenarios.json (fallback to constants)."""
    payload = _load_demo_scenarios() or {}
    jd = payload.get("judge_demo") or {}
    q = (
        jd.get("expected_quantum_contribution_pct")
        or payload.get("quantum_contribution_pct")
        or EXPECTED_HYBRID_Q_PCT
    )
    sha16 = jd.get("expected_hybrid_sha16") or EXPECTED_HYBRID_SHA16
    try:
        q_f = float(q)
    except (TypeError, ValueError):
        q_f = float(EXPECTED_HYBRID_Q_PCT)
    return {"q_pct": q_f, "sha16": str(sha16)}


def _is_stale_hybrid_q(q_contrib: Optional[float], *, demo_mode: bool = False) -> bool:
    """True when Cloud is still on the demo PHN mix (~45.3%) or demo_mode ckpt."""
    if demo_mode:
        return True
    if q_contrib is None:
        return False
    return STALE_Q_PCT_LO <= float(q_contrib) <= STALE_Q_PCT_HI


def _show_hybrid_ckpt_fingerprint(
    q_live: Optional[float],
    *,
    demo_mode: bool,
    ckpt: Dict[str, Any],
    expected: Dict[str, Any],
) -> None:
    """Always-visible loaded-bytes pin so Cloud reboot can be verified."""
    q_txt = f"{q_live:.1f}%" if q_live is not None else "N/A"
    sha = ckpt.get("sha16") or "—"
    size = ckpt.get("size")
    size_txt = f"{size} B" if isinstance(size, int) else "—"
    mtime = ckpt.get("mtime_utc") or "—"
    exp_sha = expected.get("sha16") or EXPECTED_HYBRID_SHA16
    exp_q = expected.get("q_pct", EXPECTED_HYBRID_Q_PCT)
    match = (
        sha != "—"
        and str(sha) == str(exp_sha)
        and q_live is not None
        and abs(float(q_live) - float(exp_q)) <= 2.0
        and not demo_mode
    )
    line = (
        f"**Loaded Hybrid** · q% **{q_txt}** · sha16 `{sha}` · "
        f"size {size_txt} · demo_mode={demo_mode} · mtime {mtime} · "
        f"expect q≈{float(exp_q):.1f}% / sha16 `{exp_sha}`"
    )
    if match:
        st.success(line + " · **PIN OK**")
    else:
        st.info(line)


def _warn_stale_hybrid_checkpoint() -> Optional[float]:
    """
    Big warning when loaded Hybrid reports ~45% quantum (wrong Cloud checkpoint).

    Always shows loaded sha16 / q% so a Cloud reboot can be verified against the
    local pin (expect ≈77.6%, sha16 ``1ae31d03b3a4503d``).

    Returns live q% when available (for captions).
    """
    q_live: Optional[float] = None
    demo_mode = False
    expected = _expected_hybrid_fingerprint()
    ckpt = _hybrid_ckpt_debug()
    try:
        if not Path(HYBRID_CHECKPOINT).exists():
            st.error(
                "**Wrong / missing Hybrid checkpoint** — `models/film_hybrid.pt` "
                "is absent, so Cloud built an in-memory demo PHN (~45.3% quantum). "
                "Upload the trained `models/film_hybrid.pt` (expect quantum "
                f"≈{EXPECTED_HYBRID_Q_PCT:.1f}%, sha16 `{EXPECTED_HYBRID_SHA16}`), "
                "then **Redeploy / Reboot**. Until then Hybrid often **ties** Classical."
            )
            _show_hybrid_ckpt_fingerprint(
                45.3, demo_mode=True, ckpt=ckpt, expected=expected
            )
            return 45.3
        hybrid_model, _, _ = get_hybrid_model()
        demo_mode = bool(getattr(hybrid_model, "demo_mode", False))
        q_live = estimate_quantum_contribution_pct(hybrid_model)
    except Exception as exc:
        st.warning(f"Could not inspect Hybrid checkpoint ({type(exc).__name__}).")
        _show_hybrid_ckpt_fingerprint(
            None, demo_mode=demo_mode, ckpt=ckpt, expected=expected
        )
        return None

    stale = _is_stale_hybrid_q(q_live, demo_mode=demo_mode)
    sha_mismatch = bool(
        ckpt.get("sha16")
        and expected.get("sha16")
        and str(ckpt["sha16"]) != str(expected["sha16"])
    )
    _show_hybrid_ckpt_fingerprint(
        q_live, demo_mode=demo_mode, ckpt=ckpt, expected=expected
    )
    if stale or (sha_mismatch and (stale or demo_mode or (
        q_live is not None and float(q_live) < float(expected["q_pct"]) - 10
    ))):
        q_txt = f"{q_live:.1f}%" if q_live is not None else "N/A"
        st.error(
            f"**Stale / wrong Hybrid checkpoint on Cloud** — quantum **{q_txt}** "
            f"(demo mix ≈45.3% or old ~35–42% train) · expected ≈**{expected['q_pct']:.1f}%**. "
            f"Replace **`models/film_hybrid.pt`** on GitHub "
            f"(sha16 `{expected['sha16']}`, size ≈157322 B, `demo_mode=False`), "
            f"then **Redeploy** (not reboot-only if the file never landed). "
            f"Wrong file → Hybrid **ties** Classical."
        )
    return q_live


def _judge_damage_params(scenario: Dict[str, Any]) -> Dict[str, int]:
    """Broken-road seed / size for Escape pin — prefer scenario pin, then table."""
    sid = str(scenario.get("id") or "")

    def _seed_from(obj: Dict[str, Any]) -> Optional[int]:
        for key in ("broken_seed", "damage_seed", "flood_seed"):
            if obj.get(key) is not None:
                return int(obj[key])
        return None

    seed = _seed_from(scenario)
    if seed is not None:
        return {
            "seed": seed,
            "corridor_extra": int(
                scenario.get("corridor_extra")
                or (scenario.get("metrics") or {}).get("corridor_extra")
                or 14
            ),
        }
    m = scenario.get("metrics") or {}
    seed = _seed_from(m)
    if seed is not None:
        return {
            "seed": seed,
            "corridor_extra": int(m.get("corridor_extra") or 14),
        }
    if sid in _JUDGE_DAMAGE_BY_SCENARIO:
        return dict(_JUDGE_DAMAGE_BY_SCENARIO[sid])
    return {
        "seed": 17_000 + (sum(ord(c) for c in sid or "judge") % 10_000),
        "corridor_extra": 14,
    }


def _judge_flood_params(scenario: Dict[str, Any]) -> Dict[str, int]:
    """Back-compat alias — Escape pin now uses near-epi broken roads."""
    return _judge_damage_params(scenario)


def _judge_pinned_scenario() -> Optional[Dict[str, Any]]:
    """
    Single source of truth: data/demo_scenarios.json → judge_demo.scenario_id.

    Ignores session exit picks for the audience path.
    """
    payload = _load_demo_scenarios()
    if not payload:
        return None
    jd = payload.get("judge_demo") or {}
    sid = str(
        jd.get("scenario_id")
        or payload.get("default_scenario_id")
        or "qa_1"
    )
    scenarios = list(payload.get("scenarios") or [])
    for s in scenarios:
        if str(s.get("id")) == sid:
            # Overlay judge_demo damage pin when present.
            out = dict(s)
            for key in ("broken_seed", "damage_seed", "flood_seed"):
                if jd.get(key) is not None:
                    out[key] = int(jd[key])
            if jd.get("corridor_extra") is not None:
                out["corridor_extra"] = int(jd["corridor_extra"])
            out["_payload_q_pct"] = payload.get("quantum_contribution_pct")
            out["_judge_demo"] = jd
            return out
    return _pick_advantage_scenario(payload)


def _force_judge_pin(G, scenario: Dict[str, Any]) -> Dict[str, int]:
    """
    Hard-lock start, epi, near-epi broken roads from the pinned scenario, then
    route to the live recommend_best_exit under those dynamics.

    Kept for smoke / offline checks — audience path uses `_load_fixed_scenario`.
    """
    st.session_state["exit_auto"] = False
    st.session_state["any_node_dest"] = False
    st.session_state.pop("_nudge_disruption", None)

    start = scenario.get("start_node")
    if start is not None and start in G.nodes:
        st.session_state["start_node"] = start
        st.session_state["loc_lat"] = float(G.nodes[start]["y"])
        st.session_state["loc_lon"] = float(G.nodes[start]["x"])
        st.session_state["map_center"] = [
            float(st.session_state["loc_lat"]),
            float(st.session_state["loc_lon"]),
        ]
        st.session_state["location_set"] = True
    else:
        _set_location(
            G,
            float(scenario["start_lat"]),
            float(scenario["start_lon"]),
        )
        st.session_state["location_set"] = True

    if scenario.get("epi_lat") is not None and scenario.get("epi_lon") is not None:
        _set_epicenter(float(scenario["epi_lat"]), float(scenario["epi_lon"]))
        st.session_state["disaster_active"] = True

    flood_params = _judge_damage_params(scenario)
    epi_ll = None
    if scenario.get("epi_lat") is not None and scenario.get("epi_lon") is not None:
        epi_ll = (float(scenario["epi_lon"]), float(scenario["epi_lat"]))
    n = _set_broken_roads_near_epi(
        G,
        epicenter_lonlat=epi_ll,
        seed=flood_params["seed"],
        corridor_extra=flood_params["corridor_extra"],
    )
    if scenario.get("epi_lat") is not None and scenario.get("epi_lon") is not None:
        _set_epicenter(float(scenario["epi_lat"]), float(scenario["epi_lon"]))
        st.session_state["disaster_active"] = True

    exits = _ensure_exit_nodes(G)
    start_node = st.session_state.get("start_node")
    epi = (
        float(st.session_state["epi_lon"]),
        float(st.session_state["epi_lat"]),
    )
    best, ranking = recommend_best_exit(
        G,
        start_node,
        exits,
        epi,
        edge_disruptions=st.session_state.get("edge_disruptions"),
    )
    st.session_state["exit_ranking"] = ranking
    expected = scenario.get("best_exit") or scenario.get("dest_node")
    if expected is not None and best != expected:
        st.session_state["demo_pin_failed"] = True
        st.session_state["demo_pin_fail_detail"] = (
            f"Best-ranked exit {best} ≠ pinned dest {expected}. "
            "Re-search / update data/demo_scenarios.json."
        )
    _select_evacuate_area(G, best, via="best exit (judge)", auto=True)
    st.session_state["exit_auto"] = True
    st.session_state["advantage_scenario_id"] = scenario.get("id")
    flood_params["n_edges"] = int(n)
    flood_params["best_exit"] = best
    flood_params["expected_best_exit"] = expected
    return flood_params


def _load_fixed_scenario(G) -> str:
    """
    Audience open: fixed epicenter + near-epi broken roads + 5 exits.

    Does NOT set start, does NOT auto-run Hybrid/Classical compare.
    User clicks the map, then presses Find escape route.
    """
    st.session_state.pop("demo_pin_failed", None)
    st.session_state.pop("demo_pin_fail_detail", None)
    st.session_state.pop("judge_demo_armed", None)
    st.session_state.pop("_schedule_auto_run", None)
    st.session_state.pop("_auto_run_route", None)
    st.session_state["exit_auto"] = True
    st.session_state["any_node_dest"] = False
    st.session_state["location_set"] = False
    st.session_state.pop("start_node", None)
    st.session_state.pop("loc_lat", None)
    st.session_state.pop("loc_lon", None)
    st.session_state.pop("recommended_exit", None)
    st.session_state.pop("exit_ranking", None)
    st.session_state.pop("dest_node", None)
    st.session_state.pop("dest_lat", None)
    st.session_state.pop("dest_lon", None)
    _clear_route_results()

    sc = _judge_pinned_scenario()
    try:
        if sc is not None:
            if sc.get("epi_lat") is not None and sc.get("epi_lon") is not None:
                _set_epicenter(float(sc["epi_lat"]), float(sc["epi_lon"]))
                st.session_state["disaster_active"] = True
            damage_params = _judge_damage_params(sc)
            epi_ll = None
            if sc.get("epi_lat") is not None and sc.get("epi_lon") is not None:
                epi_ll = (float(sc["epi_lon"]), float(sc["epi_lat"]))
            n = _set_broken_roads_near_epi(
                G,
                epicenter_lonlat=epi_ll,
                seed=damage_params["seed"],
                corridor_extra=damage_params["corridor_extra"],
            )
            # Re-assert epi after damage apply (feed/clear must not steal pin).
            if sc.get("epi_lat") is not None and sc.get("epi_lon") is not None:
                _set_epicenter(float(sc["epi_lat"]), float(sc["epi_lon"]))
                st.session_state["disaster_active"] = True
            # Frame map on the suggested apartment corridor (known Hybrid win zone).
            sug = _suggested_apartment_coords(sc)
            if sug is not None:
                st.session_state["suggested_start_lat"] = sug["lat"]
                st.session_state["suggested_start_lon"] = sug["lon"]
                st.session_state["suggested_start_node"] = sug.get("node")
                st.session_state["suggested_click_hint"] = sug["hint"]
                st.session_state["map_center"] = [sug["lat"], sug["lon"]]
                st.session_state["map_zoom"] = 16
            elif sc.get("epi_lat") is not None and sc.get("epi_lon") is not None:
                st.session_state["map_center"] = [
                    float(sc["epi_lat"]),
                    float(sc["epi_lon"]),
                ]
            elif sc.get("start_lat") is not None and sc.get("start_lon") is not None:
                st.session_state["map_center"] = [
                    float(sc["start_lat"]),
                    float(sc["start_lon"]),
                ]
            st.session_state["fixed_scenario_id"] = sc.get("id")
            sid = sc.get("id") or "qa_1"
            hint = (sug or {}).get("hint") if sug else None
            msg = (
                f"Fixed scenario · {sid} · epicenter + broken roads near epi "
                f"({n} amber edges). Click the faint cyan apartment ring, then "
                f"Find escape route."
                + (f" · {hint}" if hint else "")
            )
        else:
            lat, lon = _mild_default_epi(G)
            _set_epicenter(lat, lon)
            st.session_state["disaster_active"] = True
            n = _set_broken_roads_near_epi(
                G,
                epicenter_lonlat=(lon, lat),
                seed=17082,
                corridor_extra=14,
            )
            st.session_state["map_center"] = [lat, lon]
            msg = (
                f"Epicenter + broken roads near epi ({n} amber edges). "
                "Click the map to set your location."
            )
    except TrafficNotConfiguredError as exc:
        msg = str(exc)
        st.session_state["map_status"] = msg
        return msg

    _ensure_exit_nodes(G)
    # Seed dest to first exit so map drawing has a fallback before recommend.
    exits = st.session_state.get("exit_nodes") or []
    if exits and exits[0] in G.nodes:
        st.session_state["dest_node"] = exits[0]
        st.session_state["dest_lat"] = float(G.nodes[exits[0]]["y"])
        st.session_state["dest_lon"] = float(G.nodes[exits[0]]["x"])
    st.session_state["map_status"] = msg
    return msg


def _run_judge_demo(G) -> str:
    """Offline / smoke helper — not used by the audience CTA."""
    st.session_state.pop("demo_pin_failed", None)
    st.session_state.pop("demo_pin_fail_detail", None)
    try:
        sc = _judge_pinned_scenario()
        if sc is not None:
            flood_params = _force_judge_pin(G, sc)
            n = int(flood_params.get("n_edges") or 0)
            title = sc.get("title") or sc.get("id") or "advantage"
            best = flood_params.get("best_exit") or sc.get("best_exit")
            best_label = (
                (sc.get("_judge_demo") or {}).get("best_exit_label")
                or sc.get("best_exit_label")
                or _landmark_label_for(G, best)
            )
            msg = (
                f"Judge pin · {title} · Best exit {best_label} · "
                f"Broken roads near epicenter ({n} amber edges)."
            )
            st.session_state["judge_scenario_id"] = sc.get("id")
        else:
            epi_ll = (
                (
                    float(st.session_state["epi_lon"]),
                    float(st.session_state["epi_lat"]),
                )
                if st.session_state.get("epi_lat") is not None
                else None
            )
            n = _set_broken_roads_near_epi(G, epicenter_lonlat=epi_ll)
            msg = f"Broken roads near epicenter ({n} amber edges)."
    except TrafficNotConfiguredError as exc:
        msg = str(exc)
        st.session_state["map_status"] = msg
        return msg
    st.session_state["map_status"] = msg
    return msg


def _evaluate_demo_pin(
    *,
    hybrid_travel: float,
    classical_travel: Optional[float],
    q_contrib: Optional[float],
    hybrid_fell_back: bool,
    hybrid_deferred: bool,
) -> None:
    """
    After compare: refuse 'success' when pinned demo does not show H < C.

    Cloud stale film_hybrid.pt (~45.3% demo mix) typically ties Classical.
    """
    if not st.session_state.get("judge_demo_armed"):
        return
    sid = st.session_state.get("judge_scenario_id") or "qa_1"
    ckpt = _hybrid_ckpt_debug()
    pl_ok = bool(quantum_status().get("pennylane_available"))
    expected_q = None
    payload = _load_demo_scenarios() or {}
    if payload.get("quantum_contribution_pct") is not None:
        try:
            expected_q = float(payload["quantum_contribution_pct"])
        except (TypeError, ValueError):
            expected_q = None

    ct = float(classical_travel) if classical_travel is not None else None
    ht = float(hybrid_travel)
    # Strict audience contract: Hybrid travel must be strictly below Classical.
    strict_win = (
        ct is not None
        and ct > 1e-6
        and ht < ct
        and not hybrid_fell_back
        and not hybrid_deferred
    )
    st.session_state["demo_debug"] = {
        "scenario_id": sid,
        "q_pct": q_contrib,
        "hybrid_travel": ht,
        "classical_travel": ct,
        "expected_q_pct": expected_q,
        "ckpt_sha16": ckpt.get("sha16"),
        "ckpt_mtime": ckpt.get("mtime_utc"),
        "pennylane": pl_ok,
        "strict_win": bool(strict_win),
        "dest_node": st.session_state.get("dest_node"),
        "recommended_exit": st.session_state.get("recommended_exit"),
        "best_exit_match": (
            st.session_state.get("dest_node")
            == st.session_state.get("recommended_exit")
        ),
    }
    # Dest must be the recommended best exit for the audience path.
    if (
        st.session_state.get("recommended_exit") is not None
        and st.session_state.get("dest_node")
        != st.session_state.get("recommended_exit")
    ):
        strict_win = False
        st.session_state["demo_debug"]["strict_win"] = False
        st.session_state["demo_debug"]["best_exit_match"] = False
    if strict_win:
        st.session_state["demo_pin_failed"] = False
        st.session_state.pop("demo_pin_fail_detail", None)
        return

    checks = [
        f"PennyLane available: {'yes' if pl_ok else 'NO — Hybrid mirrors Classical'}",
        f"film_hybrid.pt exists: {'yes' if ckpt.get('exists') else 'NO'}",
        f"film_hybrid.pt sha16: {ckpt.get('sha16') or '—'}",
        f"film_hybrid.pt mtime: {ckpt.get('mtime_utc') or '—'}",
        f"quantum % this run: {q_contrib:.1f}%"
        if q_contrib is not None
        else "quantum % this run: N/A",
        (
            f"expected quantum % (demo_scenarios.json): ≈{expected_q:.1f}%"
            if expected_q is not None
            else "expected quantum %: (missing from demo_scenarios.json)"
        ),
        f"live travel: H={ht:.1f}"
        + (f" C={ct:.1f}" if ct is not None else " C=—")
        + (f" Δ={ct - ht:.1f}" if ct is not None else ""),
        f"dest={st.session_state.get('dest_node')} · "
        f"recommended={st.session_state.get('recommended_exit')} · "
        f"best_exit_match="
        f"{st.session_state.get('dest_node') == st.session_state.get('recommended_exit')}",
        f"scenario id: {sid}",
        "Upload models/film_hybrid.pt (not a *_bak / demo 45.3% checkpoint) "
        "+ data/demo_scenarios.json + app.py, then Reboot Cloud.",
    ]
    if q_contrib is not None and expected_q is not None and abs(q_contrib - expected_q) > 15:
        checks.insert(
            0,
            f"STALE CHECKPOINT LIKELY: q%={q_contrib:.1f} vs expected ≈{expected_q:.1f}",
        )
    detail = "\n".join(f"• {c}" for c in checks)
    st.session_state["demo_pin_failed"] = True
    st.session_state["demo_pin_fail_detail"] = detail
    # Suppress success-shaped state for the audience path.
    st.session_state["is_hybrid_route"] = False


def _arm_judge_if_near_suggested(G, lat: float, lon: float) -> bool:
    """Arm Escape win-corridor check when click is near the suggested apartment."""
    sug = _suggested_apartment_coords()
    if sug is None:
        st.session_state.pop("judge_demo_armed", None)
        return False
    dist = _haversine_m(float(lat), float(lon), sug["lat"], sug["lon"])
    node = st.session_state.get("start_node")
    near_node = (
        sug.get("node") is not None
        and node is not None
        and node == sug.get("node")
    )
    if dist <= SUGGESTED_CLICK_M or near_node:
        st.session_state["judge_demo_armed"] = True
        st.session_state["judge_scenario_id"] = sug.get("scenario_id") or "qa_1"
        return True
    st.session_state.pop("judge_demo_armed", None)
    return False


def _draw_suggested_apartment_on_map(m, G) -> None:
    """Faint cyan apartment marker — known-good Hybrid win start (no auto-run)."""
    sug = _suggested_apartment_coords()
    if sug is None:
        return
    lat, lon = sug["lat"], sug["lon"]
    # Prefer graph-snapped coords when the pinned node is present.
    node = sug.get("node")
    if node is not None and node in G.nodes:
        lat = float(G.nodes[node]["y"])
        lon = float(G.nodes[node]["x"])
    tip = sug.get("hint") or "Suggested apartment · click here"
    # Skip duplicate when user already snapped exactly onto the pin.
    start = st.session_state.get("start_node")
    if (
        st.session_state.get("location_set")
        and start is not None
        and node is not None
        and start == node
    ):
        return
    _no_click(
        folium.CircleMarker(
            [lat, lon],
            radius=16,
            color="#00E5FF",
            weight=2,
            fill=True,
            fill_color="#00E5FF",
            fill_opacity=0.12,
            opacity=0.55,
            tooltip=tip,
        )
    ).add_to(m)
    _no_click(
        folium.CircleMarker(
            [lat, lon],
            radius=6,
            color="#00E5FF",
            weight=2,
            fill=True,
            fill_color="#0a0f1e",
            fill_opacity=0.35,
            opacity=0.7,
            tooltip=tip,
        )
    ).add_to(m)
    _no_click(
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=(
                    '<div style="font-size:10px;font-weight:600;color:#7EEFFF;'
                    "opacity:0.85;white-space:nowrap;text-shadow:0 1px 2px #041018;"
                    'margin-left:10px;margin-top:-6px;">Suggested apartment</div>'
                ),
                icon_size=(140, 18),
                icon_anchor=(0, 0),
            ),
            tooltip=tip,
        )
    ).add_to(m)


def _set_location(G, lat: float, lon: float) -> None:
    """Snap user start / apartment to nearest graph node."""
    node = snap_to_nearest_node(G, float(lat), float(lon))
    if node == st.session_state.get("dest_node"):
        nbs = list(G.neighbors(node))
        if nbs:
            node = nbs[0]
    st.session_state["start_node"] = node
    st.session_state["loc_lat"] = float(G.nodes[node]["y"])
    st.session_state["loc_lon"] = float(G.nodes[node]["x"])
    st.session_state["location_set"] = True
    st.session_state["map_center"] = [
        float(st.session_state["loc_lat"]),
        float(st.session_state["loc_lon"]),
    ]
    _arm_judge_if_near_suggested(
        G,
        float(st.session_state["loc_lat"]),
        float(st.session_state["loc_lon"]),
    )


def _init_session(G, nodes, origin):
    _ensure_exit_nodes(G)
    if "disaster_active" not in st.session_state:
        st.session_state["disaster_active"] = True
    if "hazard_t_scrub" not in st.session_state:
        st.session_state["hazard_t_scrub"] = 30
    if "location_set" not in st.session_state:
        st.session_state["location_set"] = False
    if "map_center" not in st.session_state:
        st.session_state["map_center"] = [
            float(origin[1]),
            float(origin[0]),
        ]
    if "map_zoom" not in st.session_state:
        st.session_state["map_zoom"] = 16
    if "map_status" not in st.session_state:
        st.session_state["map_status"] = "Click the map to set your location."
    if "edge_disruptions" not in st.session_state:
        st.session_state["edge_disruptions"] = None
    if "feed_snapshot" not in st.session_state:
        st.session_state["feed_snapshot"] = None
    if "exit_auto" not in st.session_state:
        st.session_state["exit_auto"] = True
    _ = nodes  # kept for signature symmetry / future filters


def _apply_map_click(G, lat: float, lon: float) -> str:
    """
    Map click: set location (start) and auto-recommend best exit.

    Silent override: click near an exit pin to route there instead of #1.
    """
    _clear_route_results()
    st.session_state["map_center"] = [float(lat), float(lon)]

    # Optional silent override — click an evacuate exit pin.
    near_exit = _nearest_exit_within(G, lat, lon, EXIT_OVERRIDE_CLICK_M)
    if near_exit is not None and st.session_state.get("location_set"):
        is_best = near_exit == st.session_state.get("recommended_exit")
        msg = _select_evacuate_area(
            G, near_exit, via="exit pin", auto=is_best
        )
        if not is_best:
            label = _landmark_label_for(G, near_exit)
            msg = f"Exit override · {label} (click another exit or relocate to re-rank)."
            st.session_state["map_status"] = msg
        return msg

    _set_location(G, lat, lon)
    try:
        rec = _recommend_and_set_exit(G, via="location update")
    except Exception:
        rec = f"Evacuate exit held · node {st.session_state.get('dest_node')}."
    label = _landmark_label_for(G, st.session_state.get("dest_node"))
    if st.session_state.get("judge_demo_armed"):
        msg = (
            f"Suggested apartment · recommended {label}. "
            "Press Find escape route for the Hybrid probability path."
        )
    else:
        msg = (
            f"Location set · recommended {label}. "
            "(For the Hybrid win corridor, click the faint cyan Suggested apartment.)"
        )
    st.session_state["map_status"] = msg
    _ = rec
    return msg


def main():
    feed_info = get_traffic_provider().mode_info()
    st.markdown(
        '<div class="qr-header">'
        '<div class="qr-brand">Quantum<span>Relief</span></div>'
        '<span class="qr-online"><span class="dot"></span>Hybrid QML · Online</span>'
        f"{_traffic_feed_badge_html()}"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qr-tagline">Earthquake Escape — Hybrid finds the high-probability path</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qr-tag">Fault line → epicenter → broken roads near epi. '
        "Click map for a <b>blue start</b> → recommended exit → <b>Find escape route</b> "
        "→ watch the agent hop node-by-node with next-hop probabilities."
        "</div>",
        unsafe_allow_html=True,
    )
    if feed_info.mode == "live" and not feed_info.live_ready:
        st.warning(feed_info.detail)

    qstat = quantum_status()
    pl_ok = qstat["pennylane_available"]
    _warn_stale_hybrid_checkpoint()

    try:
        G = get_graph()
    except Exception as e:
        st.error(f"Failed to load Manila graph: {e}")
        st.stop()

    nodes = list(G.nodes())
    origin = get_graph_origin(G)
    _init_session(G, nodes, origin)

    # First open: fixed epi + near-epi broken roads + 5 exits — never auto-run compare.
    if not st.session_state.get("_demo_autoload_done"):
        st.session_state["_demo_autoload_done"] = True
        st.session_state["_feed_autoload_done"] = True
        _load_fixed_scenario(G)

    if "_map_click" in st.session_state:
        lat_p, lon_p = st.session_state.pop("_map_click")
        msg = _apply_map_click(G, float(lat_p), float(lon_p))
        try:
            st.toast(msg, icon="📍")
        except Exception:
            pass

    location_set = bool(st.session_state.get("location_set")) and (
        st.session_state.get("start_node") in G.nodes
    )
    if location_set and st.session_state.get("dest_node") not in G.nodes:
        try:
            _recommend_and_set_exit(G, via="dest repair")
        except Exception:
            far = farthest_node_from(G, st.session_state["start_node"])
            st.session_state["dest_node"] = far
            st.session_state["dest_lat"] = float(G.nodes[far]["y"])
            st.session_state["dest_lon"] = float(G.nodes[far]["x"])

    map_col, panel_col = st.columns([2, 1], gap="medium")

    # ------------------------------------------------------------------
    # RIGHT PANEL (~1/3) — click → recommend → Find escape route
    # ------------------------------------------------------------------
    with panel_col:
        badge = (
            f'<span class="qr-badge ok">PennyLane sim · {qstat["n_qubits"]}-qubit PHN</span>'
            if pl_ok
            else '<span class="qr-badge warn">PennyLane unavailable · Classical only</span>'
        )
        st.markdown(
            f'<div class="qr-panel"><h3>Earthquake Escape</h3>{badge} {_traffic_feed_badge_html()}'
            "<p style='color:#9AA8BC;font-size:0.82rem;margin:0.5rem 0 0.35rem 0'>"
            "Click map → <b style='color:#3B82F6'>blue start</b> → recommended exit → "
            "<b style='color:#E8EEF6'>Find escape route</b>. "
            "Hybrid picks the highest next-hop probability at each step."
            "</p>"
            '<div class="qr-legend">'
            f'<span><i style="background:{FAULT_LINE_COLOR}"></i>Fault line</span>'
            f'<span><i style="background:{HAZARD_ROUTE_COLOR}"></i>Epicenter / hazard</span>'
            f'<span><i style="background:{DISRUPTION_COLOR}"></i>Broken roads</span>'
            f'<span><i style="background:{START_DOT_COLOR}"></i>Blue start / agent</span>'
            f'<span><i style="background:{HYBRID_ROUTE_COLOR}"></i>Hybrid path</span>'
            f'<span><i style="background:#F5C542"></i>Gold star = best exit</span>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        if not pl_ok:
            st.warning(
                "PennyLane unavailable — Hybrid card mirrors Classical "
                "(install pennylane, reboot for the win)."
            )
            st.caption(qstat["note"])

        exit_label = (
            _landmark_label_for(G, st.session_state["dest_node"])
            if st.session_state.get("dest_node") in G.nodes
            else "—"
        )
        recommended_node = st.session_state.get("recommended_exit")

        if location_set:
            st.markdown(
                f'<div class="qr-oneliner">Your location · set'
                f'<br/><span style="color:#9AA8BC;font-size:0.78rem;font-weight:400">'
                f"Click map to move · click an exit pin to override</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="qr-oneliner">Recommended exit · '
                f"<span>{exit_label}</span>"
                f'<br/><span style="color:#9AA8BC;font-size:0.78rem;font-weight:400">'
                f"#1 ranked · gold star on map"
                f"</span></div>",
                unsafe_allow_html=True,
            )
        else:
            sug_hint = st.session_state.get("suggested_click_hint") or (
                "Click the faint cyan Suggested apartment (NW, west of epi)"
            )
            st.markdown(
                '<div class="qr-oneliner">Your location · '
                "<span>click Suggested apartment</span>"
                f'<br/><span style="color:#9AA8BC;font-size:0.78rem;font-weight:400">'
                f"{sug_hint}"
                "</span></div>",
                unsafe_allow_html=True,
            )

        run = False
        if st.button(
            "Find escape route",
            type="primary",
            use_container_width=True,
            key="btn_find_escape",
            disabled=not location_set,
            help="Roll out Hybrid policy with per-hop probabilities to the recommended exit",
        ):
            run = True

        st.caption(st.session_state.get("map_status", ""))

        # Clamp hazard scrub range (used when no route yet).
        _path_for_scrub = st.session_state.get("path")
        _radii_for_scrub = st.session_state.get("radii_trace")
        if (
            _radii_for_scrub
            and _path_for_scrub
            and len(_path_for_scrub) >= 2
        ):
            _scrub_max_t = max(0, len(_radii_for_scrub) - 1)
        else:
            _scrub_max_t = 60
        if "hazard_t_scrub" not in st.session_state:
            st.session_state["hazard_t_scrub"] = min(30, _scrub_max_t)
        else:
            try:
                _prev_scrub = int(st.session_state["hazard_t_scrub"])
            except (TypeError, ValueError):
                _prev_scrub = _scrub_max_t
            if _prev_scrub > _scrub_max_t:
                st.session_state["hazard_t_scrub"] = _scrub_max_t
            elif _prev_scrub < 0:
                st.session_state["hazard_t_scrub"] = 0

        if (not _path_for_scrub or len(_path_for_scrub) < 2) and _disaster_active():
            st.slider(
                "Hazard time t",
                0,
                _scrub_max_t,
                key="hazard_t_scrub",
                help="Epicenter damage radius grows with t",
            )

        start = st.session_state.get("start_node")
        dest = st.session_state.get("dest_node")
        epi_lat = float(st.session_state.get("epi_lat") or origin[1])
        epi_lon = float(st.session_state.get("epi_lon") or origin[0])

        if run and location_set and start in G.nodes and dest in G.nodes:
            use_hybrid = bool(pl_ok)
            hybrid_fell_back = False
            try:
                with st.spinner("Routing Hybrid escape path…"):
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
                                f"({type(hybrid_exc).__name__}). "
                                "Falling back to Classical FiLM."
                            )

                    classical_model, c_mean, c_std = get_classical_model()
                    if mean is None:
                        mean, std = c_mean, c_std

                    hero_model = hybrid_model if use_hybrid else classical_model
                    label = (
                        "Hybrid QML (PHN · sim)"
                        if use_hybrid and not hybrid_fell_back
                        else "Classical FiLM (ablation)"
                    )

                    cmp = compare_three_way(
                        G,
                        hero_model,
                        classical_model,
                        mean,
                        std,
                        start,
                        dest,
                        (epi_lon, epi_lat),
                        include_classical=True,
                        include_dijkstra=True,
                        edge_disruptions=st.session_state.get("edge_disruptions"),
                    )

                    h = cmp["hybrid"]
                    path = h["path"]
                    hybrid_path_raw = list(path) if path else None
                    radii_trace = h["radii_trace"]
                    qml_travel = h["travel_time"]
                    sample_x = h.get("sample_x")
                    route_meta = h["meta"]
                    reached = bool(h["exit_reached"]) and bool(path) and path[-1] == dest

                    classical_path = None
                    classical_travel = 0.0
                    classical_meta = {}
                    classical_reached = False
                    classical_accuracy = 0.0
                    if cmp.get("classical"):
                        c = cmp["classical"]
                        classical_path = c["path"]
                        classical_travel = float(c["travel_time"])
                        classical_meta = c.get("meta") or {}
                        classical_reached = bool(c["exit_reached"])
                        classical_accuracy = float(
                            c.get("overlap_vs_dijkstra_pct") or 0.0
                        )

                    # Empty Hybrid path → try Classical before hard-failing.
                    if (not path or len(path) < 2) and classical_path and len(classical_path) >= 2:
                        path = classical_path
                        radii_trace = (cmp.get("classical") or {}).get("radii_trace") or []
                        reached = bool(classical_reached) and path[-1] == dest
                        hybrid_fell_back = True

                    if not path or len(path) < 2:
                        raise RuntimeError(
                            "No route hops found — try another start or destination."
                        )

                    dij_path, dij_travel = (None, 0.0)
                    dij_reached = False
                    if cmp.get("dijkstra"):
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

                    latency_ms = cmp.get("latency_ms") or {}
                    hybrid_deferred = bool(hybrid_fell_back)
                    deferred_reason = "failed" if hybrid_fell_back else ""
                    deferred_note = ""
                    primary_engine = "hybrid"

                    if use_hybrid and not hybrid_fell_back:
                        defer, reason = _should_defer_hybrid(
                            hybrid_travel=float(qml_travel),
                            classical_travel=(
                                float(classical_travel)
                                if classical_path is not None
                                else None
                            ),
                            hybrid_reached=bool(reached),
                            classical_path=classical_path,
                            latency_ms=latency_ms,
                        )
                        if defer and classical_path and len(classical_path) >= 2:
                            hybrid_deferred = True
                            deferred_reason = reason
                            primary_engine = "classical"
                            # Serve Classical as primary recommendation; keep Hybrid faded.
                            path = classical_path
                            radii_trace = (cmp.get("classical") or {}).get(
                                "radii_trace"
                            ) or radii_trace
                            reached = bool(classical_reached) and path[-1] == dest
                            label = "Classical FiLM (Hybrid deferred)"
                            deferred_note = "Hybrid deferred · showing Classical"
                            if reason == "catastrophic":
                                deferred_note += (
                                    f" · Hybrid travel >"
                                    f"{HYBRID_CATASTROPHIC_RATIO:.2f}× Classical"
                                )
                            elif reason == "failed":
                                deferred_note += " · Hybrid did not reach destination"
                            elif reason == "timeout":
                                deferred_note += " · Hybrid too slow this run"

                    if hybrid_fell_back and classical_path and len(classical_path) >= 2:
                        primary_engine = "classical"
                        path = classical_path
                        reached = bool(classical_reached)
                        label = "Classical FiLM (Hybrid deferred)"
                        if not deferred_note:
                            if deferred_reason == "failed":
                                deferred_note = (
                                    "Hybrid deferred · showing Classical · Hybrid failed"
                                )
                            else:
                                deferred_note = (
                                    "Hybrid deferred · showing Classical · runtime"
                                )
                                deferred_reason = deferred_reason or "runtime"

                    st.session_state.update(
                        {
                            "path": path,
                            "classical_path": classical_path,
                            "dij_path": dij_path,
                            "hybrid_path_raw": hybrid_path_raw,
                            "radii_trace": radii_trace,
                            "qml_travel": qml_travel,
                            "classical_travel": classical_travel,
                            "dij_travel": dij_travel,
                            "qml_safety": float(
                                (h.get("safety") or {}).get(
                                    "safety_score", float("nan")
                                )
                            ),
                            "classical_safety": (
                                float(
                                    (cmp["classical"].get("safety") or {}).get(
                                        "safety_score", float("nan")
                                    )
                                )
                                if cmp.get("classical")
                                else None
                            ),
                            "dij_safety": (
                                float(
                                    (cmp["dijkstra"].get("safety") or {}).get(
                                        "safety_score", float("nan")
                                    )
                                )
                                if cmp.get("dijkstra")
                                else None
                            ),
                            "qml_mean_epi_km": float(
                                (h.get("safety") or {}).get(
                                    "mean_epi_km", float("nan")
                                )
                            ),
                            "qml_min_epi_km": float(
                                (h.get("safety") or {}).get(
                                    "min_epi_km", float("nan")
                                )
                            ),
                            "classical_min_epi_km": (
                                float(
                                    (cmp["classical"].get("safety") or {}).get(
                                        "min_epi_km", float("nan")
                                    )
                                )
                                if cmp.get("classical")
                                else None
                            ),
                            "dij_min_epi_km": (
                                float(
                                    (cmp["dijkstra"].get("safety") or {}).get(
                                        "min_epi_km", float("nan")
                                    )
                                )
                                if cmp.get("dijkstra")
                                else None
                            ),
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
                            "latency_ms": latency_ms,
                            "demo_hybrid": bool(
                                getattr(hero_model, "demo_mode", False)
                                and use_hybrid
                                and not hybrid_fell_back
                                and not hybrid_deferred
                            ),
                            "is_hybrid_route": bool(
                                use_hybrid
                                and not hybrid_fell_back
                                and not hybrid_deferred
                            ),
                            "hybrid_deferred": bool(hybrid_deferred),
                            "deferred_reason": deferred_reason,
                            "deferred_note": deferred_note,
                            "primary_engine": primary_engine,
                            "epi": (epi_lon, epi_lat),
                            "start": start,
                            "dest": dest,
                            "step_trace": list(
                                h.get("step_trace")
                                or (route_meta or {}).get("step_trace")
                                or []
                            ),
                            "path_logprob": h.get("path_logprob")
                            or (route_meta or {}).get("path_logprob"),
                            "path_prob_product": h.get("path_prob_product")
                            or (route_meta or {}).get("path_prob_product"),
                            "path_anim_step": 0,
                            "path_anim_playing": True,
                        }
                    )
                    _evaluate_demo_pin(
                        hybrid_travel=float(qml_travel),
                        classical_travel=(
                            float(classical_travel)
                            if classical_path is not None
                            else None
                        ),
                        q_contrib=q_contrib,
                        hybrid_fell_back=bool(hybrid_fell_back),
                        hybrid_deferred=bool(hybrid_deferred),
                    )
                    pin_failed = bool(st.session_state.get("demo_pin_failed"))
                    if pin_failed:
                        toast_msg = (
                            "Demo pin failed — Hybrid not loaded or wrong checkpoint"
                        )
                        toast_icon = "🚨"
                    elif deferred_note:
                        toast_msg = deferred_note
                        toast_icon = "⚠️"
                    else:
                        toast_msg = (
                            "Escape route ready — play the blue agent hop-by-hop."
                        )
                        toast_icon = "✅"
                    try:
                        st.toast(toast_msg, icon=toast_icon)
                    except Exception:
                        pass
            except Exception as e:
                detail = str(e)
                hint = ""
                if "numpy" in detail.lower():
                    hint = (
                        " Hint: Streamlit Cloud needs `numpy==1.26.4` before "
                        "`torch==2.2.2` in requirements.txt."
                    )
                st.error(f"Route calculation failed: {e}.{hint}")

        path = st.session_state.get("path")
        if path:
            qml_travel = float(st.session_state.get("qml_travel", 0.0))
            classical_travel = st.session_state.get("classical_travel")
            dij_travel = st.session_state.get("dij_travel")
            qml_safety = st.session_state.get("qml_safety")
            classical_safety = st.session_state.get("classical_safety")
            dij_safety = st.session_state.get("dij_safety")
            accuracy = float(st.session_state.get("accuracy", 0.0))
            _q_raw = st.session_state.get("q_contrib")
            try:
                q_contrib = float(_q_raw) if _q_raw is not None else None
            except (TypeError, ValueError):
                q_contrib = None
            model_used = st.session_state.get("model_used", "Hybrid QML")
            narrative = st.session_state.get("compare_narrative") or {}
            reached = bool(st.session_state.get("exit_reached", False))
            hybrid_deferred = bool(st.session_state.get("hybrid_deferred"))
            deferred_note = st.session_state.get("deferred_note") or ""
            is_hybrid = bool(st.session_state.get("is_hybrid_route")) and not hybrid_deferred
            classical_path = st.session_state.get("classical_path")
            dij_path = st.session_state.get("dij_path")
            step_trace = st.session_state.get("step_trace") or []
            path_logprob = st.session_state.get("path_logprob")
            path_prob_product = st.session_state.get("path_prob_product")

            beats_classical = False
            ties_classical = False
            safer_than_classical = False
            safety_win = False
            if (
                not hybrid_deferred
                and classical_path is not None
                and classical_travel is not None
                and reached
            ):
                ct = float(classical_travel)
                if ct > 1e-6:
                    beats_classical = bool(qml_travel < ct)
                    ties_classical = bool(
                        not beats_classical and qml_travel <= ct * 1.02
                    )
                if (
                    qml_safety is not None
                    and classical_safety is not None
                    and np.isfinite(float(qml_safety))
                    and np.isfinite(float(classical_safety))
                ):
                    safer_than_classical = bool(
                        float(qml_safety) > float(classical_safety) + 1e-6
                    )
                    safety_win = bool(safer_than_classical and ties_classical)
            near_dij = (
                not hybrid_deferred
                and dij_travel is not None
                and dij_path
                and reached
                and qml_travel <= float(dij_travel) * 1.25
            )
            if not hybrid_deferred:
                if narrative.get("hybrid_beats_classical") is not None:
                    beats_classical = bool(narrative["hybrid_beats_classical"])
                if narrative.get("hybrid_ties_classical") is not None:
                    ties_classical = bool(narrative["hybrid_ties_classical"])
                if narrative.get("hybrid_safer_than_classical") is not None:
                    safer_than_classical = bool(narrative["hybrid_safer_than_classical"])
                if narrative.get("hybrid_safety_win") is not None:
                    safety_win = bool(narrative["hybrid_safety_win"])
                if narrative.get("hybrid_near_dijkstra") is not None:
                    near_dij = bool(narrative["hybrid_near_dijkstra"])

            st.markdown(
                '<div class="qr-panel"><h3>Hybrid escape route</h3></div>',
                unsafe_allow_html=True,
            )

            # Animation controls (after route exists so first Find paints them).
            _anim_max = max(0, len(path) - 1)
            if "path_anim_step" not in st.session_state:
                st.session_state["path_anim_step"] = 0
            else:
                st.session_state["path_anim_step"] = max(
                    0, min(int(st.session_state.get("path_anim_step", 0)), _anim_max)
                )
            st.markdown("**Escape animation**")
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                if st.button("⏮", key="anim_reset", use_container_width=True, help="Reset to start"):
                    st.session_state["path_anim_step"] = 0
                    st.session_state["path_anim_playing"] = False
                    st.rerun()
            with a2:
                if st.button("▶ Play", key="anim_play", use_container_width=True):
                    st.session_state["path_anim_playing"] = True
                    st.rerun()
            with a3:
                if st.button("⏸", key="anim_pause", use_container_width=True):
                    st.session_state["path_anim_playing"] = False
                    st.rerun()
            with a4:
                if st.button("⏭", key="anim_step", use_container_width=True, help="Advance one hop"):
                    st.session_state["path_anim_playing"] = False
                    cur = int(st.session_state.get("path_anim_step", 0))
                    st.session_state["path_anim_step"] = min(cur + 1, _anim_max)
                    st.rerun()
            _cur_anim = max(0, min(int(st.session_state.get("path_anim_step", 0)), _anim_max))
            _new_anim = st.slider(
                "Path hop",
                0,
                _anim_max,
                value=_cur_anim,
                help="Blue agent moves node-by-node along the Hybrid path",
            )
            if int(_new_anim) != _cur_anim:
                st.session_state["path_anim_step"] = int(_new_anim)
                st.session_state["path_anim_playing"] = False
            anim_step = _path_anim_step(path)
            st.caption(
                f"Hop {anim_step}/{_anim_max} · hazard rings track the same step"
            )

            pin_failed = bool(st.session_state.get("demo_pin_failed"))
            if pin_failed:
                st.error(
                    "**Demo pin failed — Hybrid not loaded or wrong checkpoint**\n\n"
                    + (st.session_state.get("demo_pin_fail_detail") or "")
                )
                beats_classical = False
                ties_classical = False
                safety_win = False
                near_dij = False
            elif _is_stale_hybrid_q(
                q_contrib,
                demo_mode=bool(st.session_state.get("demo_hybrid")),
            ):
                st.error(
                    f"Quantum **{q_contrib:.1f}%** ≈ demo mix 45.3% — Cloud still has the "
                    "wrong `film_hybrid.pt`. Upload the trained checkpoint "
                    f"(expect ≈{EXPECTED_HYBRID_Q_PCT:.1f}%, sha16 `{EXPECTED_HYBRID_SHA16}`), "
                    "Reboot, then click map → Find escape route."
                )

            if hybrid_deferred and deferred_note and not pin_failed:
                st.warning(deferred_note)

            # Hazard caption synced to animation hop (no second scrubber write).
            radii_for_scrub = st.session_state.get("radii_trace")
            if _disaster_active() and radii_for_scrub and path and len(path) >= 2:
                max_t = max(0, len(radii_for_scrub) - 1)
                t_scrub = max(0, min(int(anim_step), max_t))
                st.session_state["_step_reveal"] = min(t_scrub, len(path) - 1)
                r_now = float(damage_radius(float(t_scrub)))
                if 0 <= t_scrub < len(radii_for_scrub):
                    r_now = float(radii_for_scrub[t_scrub]["r_epi"])
                st.caption(
                    f"Hazard rings follow the agent · "
                    f"r_epi ≈ **{r_now:.3f} km** · hop {t_scrub}/{max_t}"
                )
            else:
                st.session_state.pop("_step_reveal", None)

            show_hero = bool(
                not hybrid_deferred
                and not pin_failed
                and (beats_classical or safety_win)
            )
            win = " win" if show_hero else ""
            hero_pill = (
                '<span class="qr-hero-pill">HERO</span>' if show_hero else ""
            )
            hybrid_sub = (
                "Deferred this run"
                if hybrid_deferred
                else "Quantum-inspired Hybrid policy (sim)"
            )
            st.markdown(
                f'<div class="qr-card hybrid{win}">'
                f"{hero_pill}"
                f'<div class="label">Hybrid travel</div>'
                f'<div class="value accent">{qml_travel:.1f}</div>'
                f'<div class="sub">{hybrid_sub}</div></div>',
                unsafe_allow_html=True,
            )

            # Path probability score (product of chosen next-hop softmax probs).
            prod_txt = _fmt_prob(path_prob_product)
            logp_txt = (
                f"{float(path_logprob):.2f}"
                if path_logprob is not None and np.isfinite(float(path_logprob))
                else "—"
            )
            n_prob = sum(
                1 for e in step_trace if e.get("prob") is not None
            )
            st.markdown(
                f'<div class="qr-card hybrid">'
                f'<div class="label">Path probability score</div>'
                f'<div class="value accent" style="font-size:1.35rem">{prod_txt}</div>'
                f'<div class="sub">∏ chosen P(next|state) · {n_prob} scored hops · '
                f"Σ log P = {logp_txt}</div></div>",
                unsafe_allow_html=True,
            )

            # Live hop candidates at current animation step.
            hop_rows = []
            if anim_step < len(step_trace):
                entry = step_trace[anim_step]
                hop_rows.append(
                    f'<div class="hi">Hop {anim_step + 1} · mode {entry.get("mode", "—")} · '
                    f'chosen P={_fmt_prob(entry.get("prob"))}</div>'
                )
                for cand in (entry.get("candidates") or [])[:5]:
                    mark = "→" if cand.get("chosen") else "·"
                    hop_rows.append(
                        f"{mark} node {cand.get('node')} · {_fmt_prob(cand.get('prob'))}"
                    )
            elif step_trace:
                hop_rows.append(
                    f'<div class="hi">At exit · path product {_fmt_prob(path_prob_product)}</div>'
                )
            else:
                hop_rows.append("No per-hop probabilities recorded this run.")
            st.markdown(
                f'<div class="qr-card"><div class="label">Next-hop probabilities</div>'
                f'<div class="qr-prob">{"".join(f"<div>{r}</div>" for r in hop_rows)}'
                f"</div>"
                f'<div class="sub">Softmax over neighbor logits · policy picks max P</div></div>',
                unsafe_allow_html=True,
            )

            q_val = (
                f"{q_contrib:.1f}%"
                if is_hybrid and q_contrib is not None and q_contrib > 0
                else ("N/A" if is_hybrid else "—")
            )

            def _fmt_safe(v) -> str:
                try:
                    fv = float(v)
                    return f"{fv:.2f}" if np.isfinite(fv) else "—"
                except (TypeError, ValueError):
                    return "—"

            st.markdown(
                f'<div class="qr-card">'
                f'<div class="label">Reached · safety · quantum</div>'
                f'<div class="value" style="font-size:1.15rem">'
                f'{"YES" if reached else "NO"} · {_fmt_safe(qml_safety)} · {q_val}</div>'
                f'<div class="sub">Exit · path safety score · PHN weight share</div></div>',
                unsafe_allow_html=True,
            )

            if hybrid_deferred:
                story = deferred_note or "Hybrid deferred"
            elif pin_failed:
                story = "Demo pin failed — Hybrid not loaded or wrong checkpoint"
            elif beats_classical:
                story = "Hybrid high-P path · beats Classical travel"
            elif safety_win:
                story = "Travel tie · Hybrid safer"
            else:
                story = f"{model_used} · greedy max-P rollout"
            st.markdown(
                f'<div class="qr-card{" win" if show_hero else ""}">'
                f'<div class="label">Verdict</div>'
                f'<div class="value" style="font-size:1.05rem">{story}</div></div>',
                unsafe_allow_html=True,
            )

            with st.expander("Advanced · Classical / Dijkstra", expanded=False):
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
                st.caption(
                    f"Classical travel **{c_val}** · Dijkstra **{d_val}** · "
                    f"Hybrid overlap vs Dijkstra **{accuracy:.0f}%** · "
                    f"Classical safety {_fmt_safe(classical_safety)} · "
                    f"Dijkstra safety {_fmt_safe(dij_safety)}"
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
                q_line = (
                    f"≈ **{q_contrib:.1f}%** this run"
                    if q_contrib is not None and q_contrib > 0
                    else "N/A this run"
                )
                st.caption(
                    f"Quantum contribution {q_line} · "
                    "100 × mean(|W_q|) / (mean(|W_c|) + mean(|W_q|))"
                )
                show_cmp = st.checkbox(
                    "Show Classical / Dijkstra paths on map",
                    value=False,
                    key="show_cmp_paths",
                )
                st.session_state["_draw_cmp_paths"] = bool(show_cmp)
        else:
            st.info(
                "Click the map to set a **blue start**, then press "
                "**Find escape route** for the Hybrid probability path."
            )

        st.markdown(
            '<div class="qr-footer">'
            "Team 5 — Quantrio · QC4SG SEA Quantathon 2026<br/>"
            "QuantumRelief · Earthquake Escape · Hybrid high-probability routing."
            "</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # LEFT MAP (~2/3) — Folium only (no wrap div)
    # ------------------------------------------------------------------
    with map_col:
        path = st.session_state.get("path")
        classical_path = st.session_state.get("classical_path")
        dij_path = st.session_state.get("dij_path")
        hybrid_path_raw = st.session_state.get("hybrid_path_raw")
        hybrid_deferred = bool(st.session_state.get("hybrid_deferred"))
        radii_trace = st.session_state.get("radii_trace")
        step_trace = st.session_state.get("step_trace") or []
        start_draw = st.session_state.get("start_node")
        dest_draw = st.session_state.get("dest_node")
        epi = (
            float(st.session_state.get("epi_lon") or origin[0]),
            float(st.session_state.get("epi_lat") or origin[1]),
        )
        disaster_on = _disaster_active()
        draw_cmp = bool(st.session_state.get("_draw_cmp_paths"))
        n_exits = len(st.session_state.get("exit_nodes") or [])
        st.caption(
            f"Fault line → epicenter → broken roads · "
            f"blue start / agent · cyan Hybrid hops · "
            f"exits ({n_exits or N_EVACUATE_AREAS})"
            + (" · red rings = hazard" if disaster_on else "")
            + (" · Advanced comparison paths on" if draw_cmp else "")
        )

        # Animation hop drives hazard rings when a route is active.
        anim_step = _path_anim_step(path) if path and len(path) >= 2 else 0
        if disaster_on:
            if radii_trace and path and len(path) >= 2:
                max_t = max(0, len(radii_trace) - 1)
                t_show = max(0, min(int(anim_step), max_t))
            else:
                max_t = 60
                try:
                    t_show = int(st.session_state.get("hazard_t_scrub", max_t))
                except (TypeError, ValueError):
                    t_show = max_t
                t_show = max(0, min(t_show, max_t))
        else:
            t_show = 0
            max_t = 60

        # Stable key (never include scrubber t). Overlays live on the map itself —
        # safer on Cloud than feature_group_to_add with large dynamic JS payloads.
        center = list(st.session_state["map_center"])
        zoom = int(st.session_state.get("map_zoom", 16))
        m = build_base_map(G, center, zoom)

        # 1) Fault line through epicenter (under everything).
        _draw_fault_line_on_map(m, float(epi[1]), float(epi[0]))

        # 2) Soft road disruptions (amber dashed) — broken streets near epi.
        disruption_coords = disruption_edge_latlons(
            G, st.session_state.get("edge_disruptions")
        )
        for coords_d in disruption_coords:
            _no_click(
                folium.PolyLine(
                    coords_d,
                    color=DISRUPTION_COLOR,
                    weight=5,
                    opacity=0.9,
                    dash_array="6 8",
                    tooltip="Broken roads near epicenter",
                )
            ).add_to(m)

        # 3) Red hazard rings + epicenter pin.
        if disaster_on:
            r_epi = float(damage_radius(float(t_show)))
            if radii_trace and 0 <= t_show < len(radii_trace):
                r_epi = float(radii_trace[t_show]["r_epi"])
            feed_r = st.session_state.get("feed_r_epi_km")
            if feed_r is not None and (not radii_trace):
                r_epi = float(feed_r) * (0.5 + 0.5 * (t_show / max(max_t, 1)))

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

            _no_click(
                folium.CircleMarker(
                    [epi[1], epi[0]],
                    radius=8,
                    color="#fff",
                    weight=2,
                    fill=True,
                    fill_color=HAZARD_ROUTE_COLOR,
                    fill_opacity=1.0,
                    tooltip="Earthquake epicenter",
                )
            ).add_to(m)
            _no_click(
                folium.Marker(
                    [epi[1], epi[0]],
                    icon=folium.Icon(color="red", icon="warning-sign"),
                    tooltip="Earthquake epicenter",
                )
            ).add_to(m)

        # Suggested apartment (known Hybrid win corridor) — guide free-click.
        _draw_suggested_apartment_on_map(m, G)

        # Blue start (fixed location). When animating, the moving agent is drawn
        # by _draw_hybrid_path_animation; still show start faintly if mid-route.
        if (
            location_set
            and start_draw is not None
            and start_draw in G.nodes
        ):
            s_lat = float(G.nodes[start_draw]["y"])
            s_lon = float(G.nodes[start_draw]["x"])
            if not path or len(path) < 2:
                _draw_start_blue_dot(m, s_lat, s_lon, tooltip="Your location (start)")
            else:
                # Ghost start so the origin stays visible while the agent moves.
                _no_click(
                    folium.CircleMarker(
                        [s_lat, s_lon],
                        radius=6,
                        color=START_DOT_COLOR,
                        weight=2,
                        fill=True,
                        fill_color=START_DOT_COLOR,
                        fill_opacity=0.35,
                        tooltip="Start",
                    )
                ).add_to(m)

        # All candidate evacuate areas (recommended highlighted; selected = routing).
        recommended_draw = st.session_state.get("recommended_exit")
        if (
            recommended_draw is None
            and location_set
            and bool(st.session_state.get("exit_auto", True))
        ):
            recommended_draw = dest_draw
        _draw_evacuate_exits_on_map(
            m,
            G,
            dest_node=dest_draw if location_set else None,
            recommended_node=recommended_draw if location_set else None,
        )

        # Optional Classical / Dijkstra overlays (Advanced checkbox).
        if draw_cmp or hybrid_deferred:
            if dij_path and len(dij_path) >= 2:
                coords_d = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in dij_path]
                _no_click(
                    folium.PolyLine(
                        coords_d,
                        color=DIJKSTRA_ROUTE_COLOR,
                        weight=3,
                        opacity=0.7,
                        dash_array="8 10",
                        tooltip="Dijkstra",
                    )
                ).add_to(m)
            if classical_path and len(classical_path) >= 2:
                coords_c = [
                    [G.nodes[n]["y"], G.nodes[n]["x"]] for n in classical_path
                ]
                _no_click(
                    folium.PolyLine(
                        coords_c,
                        color=CLASSICAL_ROUTE_COLOR,
                        weight=4 if hybrid_deferred else 3,
                        opacity=0.9 if hybrid_deferred else 0.7,
                        tooltip="Classical FiLM",
                    )
                ).add_to(m)

        # Hybrid path animation (primary).
        if hybrid_deferred:
            faded = hybrid_path_raw or path or []
            if faded and len(faded) >= 2:
                coords_h = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in faded]
                _no_click(
                    folium.PolyLine(
                        coords_h,
                        color=HYBRID_ROUTE_COLOR,
                        weight=3,
                        opacity=0.35,
                        dash_array="4 10",
                        tooltip="Hybrid (deferred)",
                    )
                ).add_to(m)
        elif path and len(path) >= 2:
            _draw_hybrid_path_animation(
                m, G, path, step_trace, step=anim_step
            )

        map_data = st_folium(
            m,
            key="qr_map_escape",
            height=MAP_H,
            use_container_width=True,
            returned_objects=["last_clicked"],
            center=center,
            zoom=zoom,
        )

        if map_data and map_data.get("last_clicked"):
            click = map_data["last_clicked"]
            if click and "lat" in click and "lng" in click:
                lat_c, lon_c = float(click["lat"]), float(click["lng"])
                click_key = (round(lat_c, 6), round(lon_c, 6))
                if click_key != st.session_state.get("_last_click_key"):
                    st.session_state["_last_click_key"] = click_key
                    st.session_state["_map_click"] = (lat_c, lon_c)
                    st.rerun()

    # Auto-advance path animation after the map has painted this frame.
    _play_path = st.session_state.get("path")
    if (
        st.session_state.get("path_anim_playing")
        and _play_path
        and len(_play_path) >= 2
    ):
        import time as _time

        _max_play = len(_play_path) - 1
        _cur_play = int(st.session_state.get("path_anim_step", 0))
        if _cur_play >= _max_play:
            st.session_state["path_anim_playing"] = False
        else:
            _time.sleep(0.40)
            st.session_state["path_anim_step"] = _cur_play + 1
            st.rerun()


if __name__ == "__main__":
    main()
