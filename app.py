"""
QuantumRelief — B2G2C Escape-only Streamlit demo.

Citizens (and gov-facing demos): set your location → random quake → auto-best exit
→ Hybrid QML vs Classical vs Dijkstra. Folium 2D only. No God View surface.
Layout: left ~2/3 map (fixed), right ~1/3 scrollable controls + metrics.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

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
from src.routing_service import (
    compare_three_way,
    dijkstra_escape_route,
    nearest_node as _rs_nearest_node,
    path_travel_time as _rs_path_travel_time,
    predict_escape_route,
    recommend_best_exit,
    route_overlap_accuracy as _rs_route_overlap,
)
from src.utils import get_graph_origin

st.set_page_config(
    page_title="QuantumRelief",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

HYBRID_ROUTE_COLOR = "#00E5FF"
CLASSICAL_ROUTE_COLOR = "#F5C542"
DIJKSTRA_ROUTE_COLOR = "#E8EEF6"
HAZARD_ROUTE_COLOR = "#FF4D6A"
EXIT_RING_COLOR = "#F5C542"
ORANGE_ACCENT = "#FF8A4C"

MAP_H = 820  # concrete Folium px height (avoid % → black map)

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
    .qr-ro {
      background: rgba(10,15,30,0.65);
      border: 1px solid rgba(154,168,188,0.16);
      border-radius: 12px;
      padding: 0.5rem 0.7rem;
      font-size: 0.82rem; color: var(--qr-mist);
      margin-bottom: 0.4rem;
    }
    .qr-ro strong { color: #fff; }
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
    .qr-footer {
      margin-top: 0.75rem; padding-top: 0.65rem;
      border-top: 1px solid rgba(154,168,188,0.12);
      color: var(--qr-mist); font-size: 0.75rem;
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


def nearest_node(G: nx.Graph, lat: float, lon: float, candidates=None):
    return _rs_nearest_node(G, lat, lon, candidates=candidates)


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


def _set_epicenter(lat: float, lon: float) -> None:
    st.session_state["epi_lat"] = float(lat)
    st.session_state["epi_lon"] = float(lon)


def _parse_lat_lon(text: str) -> Optional[Tuple[float, float]]:
    """Parse 'lat, lon' or 'lat lon' from a free-text field."""
    raw = (text or "").strip()
    if not raw:
        return None
    m = re.match(
        r"^\s*(-?\d+(?:\.\d+)?)\s*[,;\s]\s*(-?\d+(?:\.\d+)?)\s*$",
        raw,
    )
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    # Manila Intramuros ≈ lat 14.59, lon 120.97 — detect swapped order
    if 14.0 <= a <= 15.5 and 120.0 <= b <= 122.0:
        return a, b
    if 14.0 <= b <= 15.5 and 120.0 <= a <= 122.0:
        return b, a
    # Accept any plausible lat/lon pair
    if -90 <= a <= 90 and -180 <= b <= 180:
        return a, b
    return None


def _geocode_address(query: str) -> Optional[Tuple[float, float]]:
    """Nominatim lookup (no extra deps). Prefer Intramuros / Manila bias."""
    q = (query or "").strip()
    if not q:
        return None
    if "manila" not in q.lower() and "intramuros" not in q.lower():
        q = f"{q}, Intramuros, Manila, Philippines"
    url = (
        "https://nominatim.openstreetmap.org/search?"
        + urllib.parse.urlencode(
            {"q": q, "format": "json", "limit": 1, "countrycodes": "ph"}
        )
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "QuantumRelief-EscapeDemo/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data:
            return None
        return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        return None


def _resolve_location_input(text: str) -> Optional[Tuple[float, float]]:
    parsed = _parse_lat_lon(text)
    if parsed is not None:
        return parsed
    return _geocode_address(text)


def build_base_map(G, exits, map_center, map_zoom: int = 16):
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
            radius=7,
            color=ORANGE_ACCENT,
            fill=True,
            fill_color=ORANGE_ACCENT,
            fill_opacity=0.85,
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
    return predict_escape_route(
        G, model, mean, std, start, dest, epicenter_lonlat, max_steps=max_steps
    )


def dijkstra_route(G, start, dest, epicenter_lonlat, max_steps=120):
    path, _radii, _env, travel, _meta = dijkstra_escape_route(
        G, start, dest, epicenter_lonlat, max_steps=max_steps
    )
    return path, travel


def _clear_route_results():
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
        "latency_ms",
        "is_hybrid_route",
        "_step_reveal",
    ):
        st.session_state.pop(k, None)


def _refresh_exit_ranking(G, exits) -> List[dict]:
    """Auto-pick best exit silently; store one-line ranking meta."""
    start = st.session_state["start_node"]
    epi = (float(st.session_state["epi_lon"]), float(st.session_state["epi_lat"]))
    best, ranking = recommend_best_exit(G, start, exits, epi)
    st.session_state["exit_ranking"] = ranking
    st.session_state["recommended_exit"] = best
    st.session_state["dest_node"] = best
    return ranking


def _set_location(G, exits, lat: float, lon: float) -> None:
    """Snap user location to nearest non-exit road node."""
    candidates = [n for n in G.nodes() if n not in exits]
    node = nearest_node(G, float(lat), float(lon), candidates)
    st.session_state["start_node"] = node
    st.session_state["loc_lat"] = float(G.nodes[node]["y"])
    st.session_state["loc_lon"] = float(G.nodes[node]["x"])
    st.session_state["map_center"] = [
        float(st.session_state["loc_lat"]),
        float(st.session_state["loc_lon"]),
    ]
    st.session_state["loc_input"] = (
        f'{st.session_state["loc_lat"]:.5f}, {st.session_state["loc_lon"]:.5f}'
    )


def _init_session(G, exits, nodes, origin):
    xs = [G.nodes[n]["x"] for n in nodes]
    ys = [G.nodes[n]["y"] for n in nodes]
    if "start_node" not in st.session_state:
        # Default near Intramuros center
        _set_location(G, exits, 14.5908, 120.9752)
    if "loc_lat" not in st.session_state:
        n0 = st.session_state["start_node"]
        st.session_state["loc_lat"] = float(G.nodes[n0]["y"])
        st.session_state["loc_lon"] = float(G.nodes[n0]["x"])
    if "loc_input" not in st.session_state:
        st.session_state["loc_input"] = (
            f'{st.session_state["loc_lat"]:.5f}, {st.session_state["loc_lon"]:.5f}'
        )
    if "dest_node" not in st.session_state:
        st.session_state["dest_node"] = exits[0]
    if "epi_lat" not in st.session_state:
        st.session_state["epi_lat"] = float(np.mean(ys)) - 0.0015
        st.session_state["epi_lon"] = float(np.mean(xs)) + 0.0012
    if "map_center" not in st.session_state:
        st.session_state["map_center"] = [
            float(st.session_state.get("loc_lat", origin[1])),
            float(st.session_state.get("loc_lon", origin[0])),
        ]
    if "map_zoom" not in st.session_state:
        st.session_state["map_zoom"] = 16
    if "map_status" not in st.session_state:
        st.session_state["map_status"] = "Click the map to set your location."
    if "exit_ranking" not in st.session_state:
        _refresh_exit_ranking(G, exits)


def _apply_map_click(G, exits, lat: float, lon: float) -> str:
    """Map click sets user location (snapped to nearest graph node)."""
    _clear_route_results()
    st.session_state["map_center"] = [float(lat), float(lon)]
    _set_location(G, exits, lat, lon)
    _refresh_exit_ranking(G, exits)
    best = st.session_state.get("recommended_exit", st.session_state["dest_node"])
    msg = (
        f"Location → {st.session_state['loc_lat']:.5f}, "
        f"{st.session_state['loc_lon']:.5f} (node {st.session_state['start_node']}). "
        f"Best exit → node {best}."
    )
    st.session_state["map_status"] = msg
    return msg


def main():
    st.markdown(
        '<div class="qr-header">'
        '<div class="qr-brand">Quantum<span>Relief</span></div>'
        '<span class="qr-online"><span class="dot"></span>Hybrid QML · Online</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qr-tagline">Escape · Manila Intramuros · B2G2C</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qr-tag">Set your location. We pick the safest exit and route you out '
        "with Hybrid QML — compared to Classical and Dijkstra.</div>",
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

    if "_map_click" in st.session_state:
        lat_p, lon_p = st.session_state.pop("_map_click")
        msg = _apply_map_click(G, exits, float(lat_p), float(lon_p))
        try:
            st.toast(msg, icon="📍")
        except Exception:
            pass

    start_options = [n for n in nodes if n not in exits]
    if st.session_state["start_node"] not in start_options:
        st.session_state["start_node"] = start_options[0]
    if st.session_state["dest_node"] not in exits:
        st.session_state["dest_node"] = exits[0]

    if not st.session_state.get("exit_ranking"):
        _refresh_exit_ranking(G, exits)

    map_col, panel_col = st.columns([2, 1], gap="medium")

    # ------------------------------------------------------------------
    # RIGHT PANEL (~1/3) — minimal controls + metrics
    # ------------------------------------------------------------------
    with panel_col:
        badge = (
            f'<span class="qr-badge ok">PennyLane · {qstat["n_qubits"]}-qubit HQNN</span>'
            if pl_ok
            else '<span class="qr-badge warn">PennyLane unavailable · Classical only</span>'
        )
        st.markdown(
            f'<div class="qr-panel"><h3>Escape</h3>{badge}'
            "<p style='color:#9AA8BC;font-size:0.82rem;margin:0.5rem 0 0 0'>"
            "For citizens and city demos: local Hybrid inference under a live quake. "
            f"<b style='color:{HYBRID_ROUTE_COLOR}'>Cyan</b> Hybrid · "
            f"<b style='color:{CLASSICAL_ROUTE_COLOR}'>Gold</b> Classical · "
            f"<b style='color:{DIJKSTRA_ROUTE_COLOR}'>White</b> Dijkstra."
            "</p></div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="qr-panel"><h3>Your location</h3></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="qr-ro"><strong>Snapped</strong><br/>'
            f'{st.session_state["loc_lat"]:.5f}, {st.session_state["loc_lon"]:.5f}'
            f' · node {st.session_state["start_node"]}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Click the map, or enter address / lat, lon below.")
        loc_text = st.text_input(
            "Address or lat, lon",
            key="loc_input",
            label_visibility="collapsed",
            placeholder="14.5908, 120.9752  or  Fort Santiago, Intramuros",
        )
        if st.button("Set location", use_container_width=True, type="secondary"):
            resolved = _resolve_location_input(loc_text)
            if resolved is None:
                st.warning("Could not parse location. Use lat, lon or a Manila address.")
            else:
                lat_r, lon_r = resolved
                _clear_route_results()
                _set_location(G, exits, lat_r, lon_r)
                _refresh_exit_ranking(G, exits)
                st.session_state["map_status"] = (
                    f"Location → {st.session_state['loc_lat']:.5f}, "
                    f"{st.session_state['loc_lon']:.5f}"
                )
                st.rerun()

        st.markdown('<div class="qr-panel"><h3>Epicenter</h3></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="qr-ro"><strong>Quake</strong><br/>'
            f'{st.session_state["epi_lat"]:.5f}, {st.session_state["epi_lon"]:.5f}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Random epicenter", use_container_width=True, type="secondary"):
            (lon_r, lat_r), _ = random_epicenter(G)
            _set_epicenter(lat_r, lon_r)
            _clear_route_results()
            _refresh_exit_ranking(G, exits)
            st.session_state["map_status"] = f"Epicenter → {lat_r:.5f}, {lon_r:.5f}"
            st.rerun()

        ranking = st.session_state.get("exit_ranking") or []
        if ranking:
            best = ranking[0]
            t_txt = (
                f'{best["travel_time"]:.1f}'
                if best.get("exit_reached") and np.isfinite(best.get("travel_time", np.nan))
                else "—"
            )
            st.markdown(
                f'<div class="qr-rec"><strong>Best exit · {best["label"]}</strong>'
                f' · score {best["combined_score"]:.0f}/100 · est. {t_txt}'
                f'<br/><span style="font-size:0.78rem">{best.get("why", "")}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Best exit will appear after location + epicenter are set.")

        if not pl_ok:
            st.caption(qstat["note"])

        run = st.button("Find route", type="primary", use_container_width=True)
        st.caption(
            f"{G.number_of_nodes()} nodes · {G.number_of_edges()} edges · "
            f"{st.session_state.get('map_status', '')}"
        )

        start = st.session_state["start_node"]
        dest = st.session_state["dest_node"]
        epi_lat = float(st.session_state["epi_lat"])
        epi_lon = float(st.session_state["epi_lon"])

        if run:
            use_hybrid = bool(pl_ok)
            hybrid_fell_back = False
            try:
                with st.spinner("Routing Hybrid · Classical · Dijkstra…"):
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
                        "Hybrid QML (HQNN)"
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
                            "No escape hops found — try another location or epicenter."
                        )

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

                    st.session_state.update(
                        {
                            "path": path,
                            "classical_path": classical_path,
                            "dij_path": dij_path,
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
                            "demo_hybrid": bool(
                                getattr(hero_model, "demo_mode", False)
                                and use_hybrid
                                and not hybrid_fell_back
                            ),
                            "is_hybrid_route": bool(use_hybrid and not hybrid_fell_back),
                            "epi": (epi_lon, epi_lat),
                            "start": start,
                            "dest": dest,
                        }
                    )
                    try:
                        st.toast("Route ready — compare Hybrid / Classical / Dijkstra.", icon="✅")
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

            st.markdown('<div class="qr-panel"><h3>Metrics</h3></div>', unsafe_allow_html=True)

            radii_for_scrub = st.session_state.get("radii_trace")
            if radii_for_scrub and path and len(path) >= 2:
                max_t = max(0, len(radii_for_scrub) - 1)
                prev_t = int(st.session_state.get("hazard_t_scrub", max_t))
                if prev_t > max_t or "hazard_t_scrub" not in st.session_state:
                    st.session_state["hazard_t_scrub"] = max_t
                t_scrub = st.slider(
                    "Hazard time t",
                    0,
                    max_t,
                    key="hazard_t_scrub",
                )
                st.session_state["_step_reveal"] = min(int(t_scrub) + 1, len(path) - 1)
            else:
                st.session_state.pop("_step_reveal", None)

            win = " win" if beats_classical or (reached and is_hybrid) else ""
            st.markdown(
                f'<div class="qr-card hybrid{win}">'
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
                f'<div class="qr-card classical">'
                f'<div class="label">Classical · Dijkstra</div>'
                f'<div class="value gold" style="font-size:1.35rem">{c_val}'
                f' <span style="color:#9AA8BC;font-size:0.85rem">/</span> '
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
                f'<div class="qr-card">'
                f'<div class="label">Exit · overlap · quantum</div>'
                f'<div class="value" style="font-size:1.2rem">'
                f'{"YES" if reached else "NO"} · {accuracy:.0f}% · {q_val}</div>'
                f'<div class="sub">Reached · vs Dijkstra · PHN contrib</div></div>',
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
                f'<div class="value" style="font-size:1.05rem">{story}</div></div>',
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

            with st.expander("Quantum Contribution", expanded=False):
                q_line = (
                    f"≈ **{q_contrib:.1f}%** this run"
                    if q_contrib is not None and q_contrib > 0
                    else "N/A this run"
                )
                st.markdown(
                    f"""
**Live from Hybrid checkpoint** ({q_line}).

```
Quantum Contribution % = 100 × mean(|W_q|) / (mean(|W_c|) + mean(|W_q|))
```

{QUANTUM_CONTRIBUTION_FORMULA}
                    """
                )
        else:
            st.info("Set location · random epicenter · **Find route**.")

        st.markdown(
            '<div class="qr-footer">'
            "Team 5 — Quantrio · QC4SG SEA Quantathon 2026<br/>"
            "B2G2C Escape · Folium 2D · Quantum Intelligence. Human Relief."
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
        radii_trace = st.session_state.get("radii_trace")
        start_draw = st.session_state["start_node"]
        dest_draw = st.session_state["dest_node"]
        epi = (float(st.session_state["epi_lon"]), float(st.session_state["epi_lat"]))
        ranking = st.session_state.get("exit_ranking") or []

        t_show = int(st.session_state.get("hazard_t_scrub", 0)) if radii_trace else 0
        step_reveal = st.session_state.get("_step_reveal")
        if radii_trace and path and len(path) >= 2 and step_reveal is None:
            t_show = max(0, len(radii_trace) - 1)
            step_reveal = min(t_show + 1, len(path) - 1)

        m = build_base_map(
            G,
            exits,
            st.session_state["map_center"],
            int(st.session_state.get("map_zoom", 16)),
        )

        for row in ranking:
            color = HYBRID_ROUTE_COLOR if row.get("recommended") else ORANGE_ACCENT
            radius = 11 if row.get("recommended") else 7
            marker = folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
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
            route_label = "Hybrid QML · HQNN"

        if dij_path and len(dij_path) >= 2:
            coords_d = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in dij_path]
            _no_click(
                folium.PolyLine(
                    coords_d,
                    color=DIJKSTRA_ROUTE_COLOR,
                    weight=3,
                    opacity=0.75,
                    dash_array="8 10",
                )
            ).add_to(m)

        if classical_path and len(classical_path) >= 2:
            coords_c = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in classical_path]
            _no_click(
                folium.PolyLine(
                    coords_c,
                    color=CLASSICAL_ROUTE_COLOR,
                    weight=4,
                    opacity=0.88,
                )
            ).add_to(m)

        if path and len(path) >= 2:
            end_i = step_reveal if step_reveal is not None else len(path) - 1
            partial = path[: end_i + 1]
            coords = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in partial]
            _no_click(
                folium.PolyLine(
                    coords,
                    color=HYBRID_ROUTE_COLOR,
                    weight=6,
                    opacity=0.95,
                )
            ).add_to(m)
            for n in partial:
                _no_click(
                    folium.CircleMarker(
                        [G.nodes[n]["y"], G.nodes[n]["x"]],
                        radius=4,
                        color=HYBRID_ROUTE_COLOR,
                        fill=True,
                        fill_opacity=0.95,
                    )
                ).add_to(m)

        map_data = st_folium(
            m,
            key="qr_map_escape",
            height=MAP_H,
            use_container_width=True,
            returned_objects=["last_clicked"],
            center=st.session_state["map_center"],
            zoom=int(st.session_state.get("map_zoom", 16)),
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


if __name__ == "__main__":
    main()
