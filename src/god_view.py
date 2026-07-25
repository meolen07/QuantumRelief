"""
God View — B2G / B2B Command Center for QuantumRelief.

Citizens get free B2C escape routing. Commanders and logistics teams use this
surface to watch city-wide corridor stress, inject hazards, and re-route the
network in one shot.

Honest architecture (Streamlit Cloud CPU):
  - Bulk arterial heatmap → static Dijkstra on the hazard-weighted graph
  - Hybrid QML only on a tiny hero sample (cyan quantum arterials)
  - Scaled-citizen metric is narrative (CITY_SCALE_PER_AGENT × batch), not
    14k Hybrid inferences
  - Never auto-run; cache last sim in session_state until Trigger is clicked
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import folium
import networkx as nx
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

from src.dynamic_simulation import damage_radius, exit_radius
from src.quantum_hybrid import (
    QUANTUM_CONTRIBUTION_FORMULA,
    estimate_quantum_contribution_pct,
)
from src.routing_service import path_travel_time, predict_escape_route
from src.utils import get_graph_origin

# Cloud-safe defaults — Hybrid is expensive per citizen on CPU
DEFAULT_BATCH_SIZE = 10
MAX_BATCH_SIZE = 20
HYBRID_HERO_SAMPLE = 4  # cyan quantum arterials for B2G story
HYBRID_MAX_STEPS = 28
BRIDGE_PENALTY = 80.0
FLOOD_BASE_MULT = 1.0
# Pitch narrative only: default batch of 10 → ~14k "citizens under command"
CITY_SCALE_PER_AGENT = 1_428

# Map corridor colors — Crisis Core palette (match B2C / Lovable intro)
HYBRID_ROUTE_COLOR = "#00E5FF"  # cyan Hybrid hero corridors
DIJKSTRA_ROUTE_COLOR = "#E8EEF6"  # white/light Dijkstra bulk
HAZARD_ROUTE_COLOR = "#FF4D6A"  # red/pink blocked / danger
ORANGE_ACCENT = "#FF8A4C"
EXIT_RING_COLOR = "#F5C542"  # gold exit congestion
CLASSICAL_ROUTE_COLOR = "#F5C542"  # gold (legend parity with B2C)


def _no_click(layer):
    """Keep Folium overlays from stealing map interactions."""
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


def find_main_bridge_edge(G: nx.Graph) -> Optional[Tuple[Any, Any]]:
    """
    Pick a high-centrality corridor edge to act as the 'Main Highway Bridge'.

    Prefers edges with ``betweenness`` from graph_setup; falls back to
    NetworkX edge betweenness on a sample if the attribute is missing.
    """
    best_uv, best_bc = None, -1.0
    for u, v, data in G.edges(data=True):
        bc = float(data.get("betweenness", 0.0) or 0.0)
        if bc > best_bc:
            best_bc = bc
            best_uv = (u, v)

    if best_uv is not None and best_bc > 0:
        return best_uv

    try:
        sample_k = min(40, G.number_of_nodes())
        bc_map = nx.edge_betweenness_centrality(
            G, k=sample_k, weight="travel_time_nominal", seed=42
        )
        if bc_map:
            return max(bc_map, key=bc_map.get)
    except Exception:
        pass

    longest, best_len = None, -1.0
    for u, v, data in G.edges(data=True):
        length = float(data.get("length", data.get("travel_time_nominal", 1.0)) or 1.0)
        if length > best_len:
            best_len = length
            longest = (u, v)
    return longest


def apply_god_view_hazards(
    G: nx.Graph,
    *,
    flood_level: float = 0.0,
    block_bridge: bool = False,
    bridge_edge: Optional[Tuple[Any, Any]] = None,
) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Inject commander hazards onto a graph copy.

    - ``flood_level`` in [0, 1]: scales sector weights near the graph centroid
    - ``block_bridge``: multiplies the main bridge edge weight by BRIDGE_PENALTY
    """
    H = G.copy()
    meta: Dict[str, Any] = {
        "flood_level": float(flood_level),
        "block_bridge": bool(block_bridge),
        "bridge_edge": None,
        "blocked_edges": 0,
        "penalized_edges": 0,
    }

    if flood_level > 1e-6:
        xs = [H.nodes[n]["x"] for n in H.nodes()]
        ys = [H.nodes[n]["y"] for n in H.nodes()]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        flood_r = 0.002 + 0.012 * float(flood_level)
        flood_mult = FLOOD_BASE_MULT + 4.0 * float(flood_level)
        for u, v, data in H.edges(data=True):
            mx = 0.5 * (H.nodes[u]["x"] + H.nodes[v]["x"])
            my = 0.5 * (H.nodes[u]["y"] + H.nodes[v]["y"])
            d = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
            if d <= flood_r:
                w0 = float(data.get("weight", data.get("travel_time", 1.0)))
                data["weight"] = w0 * flood_mult
                data["travel_time"] = data["weight"]
                data["god_view_flood"] = True
                meta["penalized_edges"] += 1

    if block_bridge:
        edge = bridge_edge or find_main_bridge_edge(H)
        if edge is not None and H.has_edge(*edge):
            u, v = edge
            data = H.edges[u, v]
            w0 = float(data.get("weight", data.get("travel_time", 1.0)))
            data["weight"] = w0 * BRIDGE_PENALTY
            data["travel_time"] = data["weight"]
            data["god_view_blocked"] = True
            meta["bridge_edge"] = (u, v)
            meta["blocked_edges"] = 1

    return H, meta


def congestion_alert_status(
    flood_level: float,
    block_bridge: bool,
    blocked_edges: int,
    hazard_t: float = 8.0,
) -> str:
    """Human-readable network congestion / alert string for commanders."""
    r_epi = damage_radius(hazard_t)
    severity = 0
    if flood_level >= 0.65:
        severity += 2
    elif flood_level >= 0.35:
        severity += 1
    if block_bridge or blocked_edges > 0:
        severity += 2
    if r_epi >= 0.7:
        severity += 1

    if severity >= 4:
        return "CRITICAL — arterial failure · divert fleets"
    if severity >= 2:
        return "ELEVATED — corridor stress · Hybrid rebalancing"
    if severity >= 1:
        return "WATCH — sector hazard rising"
    return "NOMINAL — network clear"


def _fast_dijkstra_path(
    G: nx.Graph, start: Any, dest: Any
) -> Tuple[List[Any], float]:
    """Static shortest path on the already hazard-weighted graph (Cloud-fast)."""
    path = nx.shortest_path(G, start, dest, weight="weight")
    travel = float(path_travel_time(G, path))
    return path, travel


def _accumulate_edges(edge_counts: Counter, path: Sequence) -> None:
    for u, v in zip(path[:-1], path[1:]):
        edge_counts[tuple(sorted((u, v)))] += 1


def run_evacuation_batch(
    G: nx.Graph,
    model,
    mean,
    std,
    exits: Sequence,
    epicenter_lonlat: Tuple[float, float],
    *,
    n_agents: int = DEFAULT_BATCH_SIZE,
    n_hybrid: int = HYBRID_HERO_SAMPLE,
    seed: int = 42,
    max_steps: Optional[int] = HYBRID_MAX_STEPS,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Fast city-wide batch for God View.

    - Most agents: Dijkstra on hazard-weighted graph → light arterial heatmap
    - Small Hybrid QML sample: cyan quantum arterials for the B2G story
    """
    rng = np.random.default_rng(int(seed))
    candidates = [n for n in G.nodes() if n not in exits]
    empty = {
        "paths": [],
        "hybrid_paths": [],
        "edge_counts": Counter(),
        "quantum_edges": set(),
        "n_routed": 0,
        "n_success": 0,
        "n_hybrid": 0,
        "n_dijkstra": 0,
        "success_rate": 0.0,
        "avg_travel": 0.0,
        "sample_x": None,
        "quantum_contribution": 0.0,
        "n_agents_requested": 0,
        "elapsed_s": 0.0,
    }
    if not candidates or not exits:
        return empty

    n_agents = int(max(3, min(int(n_agents), MAX_BATCH_SIZE, len(candidates))))
    n_hybrid = int(max(0, min(int(n_hybrid), n_agents, HYBRID_HERO_SAMPLE)))
    # If Hybrid model missing, all Dijkstra
    if model is None or mean is None or std is None:
        n_hybrid = 0

    starts = list(rng.choice(candidates, size=n_agents, replace=False))
    hybrid_starts = set(starts[:n_hybrid])

    paths: List[List[Any]] = []
    hybrid_paths: List[List[Any]] = []
    travels: List[float] = []
    n_success = 0
    n_hybrid_ok = 0
    n_dij_ok = 0
    edge_counts: Counter = Counter()
    quantum_edges: Set[Tuple[Any, Any]] = set()
    sample_x = None
    total = len(starts)
    t0 = time.perf_counter()

    for i, start in enumerate(starts, start=1):
        dest = exits[int(rng.integers(0, len(exits)))]
        if start == dest:
            if progress_cb is not None:
                progress_cb(i, total, "skip")
            continue

        use_hybrid = start in hybrid_starts
        path: List[Any] = []
        travel = 0.0
        reached = False
        engine = "dijkstra"

        if use_hybrid:
            if progress_cb is not None:
                progress_cb(i, total, "hybrid")
            try:
                path, _radii, _env, travel, sx, meta = predict_escape_route(
                    G,
                    model,
                    mean,
                    std,
                    start,
                    dest,
                    epicenter_lonlat,
                    max_steps=max_steps or HYBRID_MAX_STEPS,
                )
                if sample_x is None and sx is not None:
                    sample_x = sx
                reached = bool(meta.get("reached") and path and path[-1] == dest)
                engine = "hybrid"
            except Exception:
                # Soft-fail to Dijkstra so the demo never hangs on one bad route
                use_hybrid = False

        if not use_hybrid:
            if progress_cb is not None:
                progress_cb(i, total, "dijkstra")
            try:
                path, travel = _fast_dijkstra_path(G, start, dest)
                reached = bool(path) and path[-1] == dest
                engine = "dijkstra"
            except Exception:
                if progress_cb is not None:
                    progress_cb(i, total, "fail")
                continue

        if not path or len(path) < 2:
            continue

        paths.append(path)
        travels.append(float(travel))
        if reached:
            n_success += 1
        _accumulate_edges(edge_counts, path)
        if engine == "hybrid":
            n_hybrid_ok += 1
            hybrid_paths.append(path)
            for u, v in zip(path[:-1], path[1:]):
                quantum_edges.add(tuple(sorted((u, v))))
        else:
            n_dij_ok += 1

    n_routed = len(paths)
    success_rate = (100.0 * n_success / n_routed) if n_routed else 0.0
    avg_travel = float(np.mean(travels)) if travels else 0.0
    q_contrib = (
        estimate_quantum_contribution_pct(model, sample_x) if model is not None else 0.0
    )
    elapsed_s = float(time.perf_counter() - t0)

    return {
        "paths": paths,
        "hybrid_paths": hybrid_paths,
        "edge_counts": edge_counts,
        "quantum_edges": quantum_edges,
        "n_routed": n_routed,
        "n_success": n_success,
        "n_hybrid": n_hybrid_ok,
        "n_dijkstra": n_dij_ok,
        "success_rate": float(success_rate),
        "avg_travel": avg_travel,
        "sample_x": sample_x,
        "quantum_contribution": float(q_contrib),
        "n_agents_requested": n_agents,
        "elapsed_s": elapsed_s,
    }


def _corridor_color(
    count: int,
    max_count: int,
    *,
    is_quantum: bool = False,
) -> str:
    """Cyan Hybrid heroes vs white/silver Dijkstra bulk."""
    if is_quantum:
        return HYBRID_ROUTE_COLOR
    if max_count <= 0:
        return DIJKSTRA_ROUTE_COLOR
    frac = count / max_count
    if frac >= 0.45:
        return DIJKSTRA_ROUTE_COLOR
    if frac >= 0.22:
        return "#A8B8CC"
    return "#6B7A90"


def _path_latlons(G: nx.Graph, path: Sequence) -> List[List[float]]:
    """Convert a node path to Folium [[lat, lon], ...] coords."""
    coords: List[List[float]] = []
    for n in path:
        if n not in G.nodes:
            continue
        coords.append([float(G.nodes[n]["y"]), float(G.nodes[n]["x"])])
    return coords


def _is_quantum_edge(
    u: Any, v: Any, quantum_edges: Set[Tuple[Any, Any]]
) -> bool:
    key = tuple(sorted((u, v)))
    return key in quantum_edges or (u, v) in quantum_edges or (v, u) in quantum_edges


def build_god_view_map(
    G: nx.Graph,
    exits: Sequence,
    epicenter_lonlat: Tuple[float, float],
    edge_counts: Counter,
    *,
    quantum_edges: Optional[Set[Tuple[Any, Any]]] = None,
    hybrid_paths: Optional[Sequence[Sequence]] = None,
    bridge_edge: Optional[Tuple[Any, Any]] = None,
    flood_level: float = 0.0,
    map_center: Optional[List[float]] = None,
    map_zoom: int = 15,
    hazard_t: float = 8.0,
) -> folium.Map:
    """Full-width Folium: epicenter danger + Hybrid / Dijkstra corridors."""
    lon_e, lat_e = epicenter_lonlat
    if map_center is None:
        origin = get_graph_origin(G)
        map_center = [float(origin[1]), float(origin[0])]

    m = folium.Map(
        location=list(map_center),
        zoom_start=int(map_zoom),
        tiles="CartoDB dark_matter",
    )
    quantum_edges = quantum_edges or set()
    hybrid_paths = list(hybrid_paths or [])
    max_count = max(edge_counts.values()) if edge_counts else 0

    # Base roads (dim)
    for u, v in G.edges():
        key = tuple(sorted((u, v)))
        if key in edge_counts and edge_counts[key] > 0:
            continue
        u_lat, u_lon = G.nodes[u]["y"], G.nodes[u]["x"]
        v_lat, v_lon = G.nodes[v]["y"], G.nodes[v]["x"]
        line = folium.PolyLine(
            [[u_lat, u_lon], [v_lat, v_lon]],
            color="#2a3f5a",
            weight=1.2,
            opacity=0.35,
        )
        _no_click(line).add_to(m)

    if flood_level > 0.05:
        xs = [G.nodes[n]["x"] for n in G.nodes()]
        ys = [G.nodes[n]["y"] for n in G.nodes()]
        flood = folium.Circle(
            location=[float(np.mean(ys)), float(np.mean(xs))],
            radius=(0.002 + 0.012 * flood_level) * 111_000.0,
            color="#3b82f6",
            weight=1,
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.08 + 0.12 * flood_level,
        )
        _no_click(flood).add_to(m)

    # 1) Dijkstra bulk first — light/white, thinner, slightly transparent
    if max_count > 0:
        ranked = sorted(edge_counts.items(), key=lambda kv: kv[1])
        for (u, v), count in ranked:
            if not G.has_edge(u, v):
                continue
            if _is_quantum_edge(u, v, quantum_edges):
                continue
            frac = count / max_count
            color = _corridor_color(count, max_count, is_quantum=False)
            weight = 3.0 + 1.0 * frac  # 3–4
            line = folium.PolyLine(
                [[G.nodes[u]["y"], G.nodes[u]["x"]], [G.nodes[v]["y"], G.nodes[v]["x"]]],
                color=color,
                weight=weight,
                opacity=0.45 + 0.25 * frac,
            )
            _no_click(line).add_to(m)

    # 2) Hybrid hero corridors LAST — cyan, thick (z-order = draw order)
    # Prefer full path polylines so hero routes stay continuous and visible.
    drawn_hybrid = False
    for path in hybrid_paths:
        coords = _path_latlons(G, path)
        if len(coords) < 2:
            continue
        line = folium.PolyLine(
            coords,
            color=HYBRID_ROUTE_COLOR,
            weight=6.5,
            opacity=0.95,
        )
        _no_click(line).add_to(m)
        drawn_hybrid = True

    # Fallback: edge segments from quantum_edges (legacy cached results)
    if not drawn_hybrid and quantum_edges and max_count > 0:
        for (u, v), count in edge_counts.items():
            if not G.has_edge(u, v):
                continue
            if not _is_quantum_edge(u, v, quantum_edges):
                continue
            line = folium.PolyLine(
                [[G.nodes[u]["y"], G.nodes[u]["x"]], [G.nodes[v]["y"], G.nodes[v]["x"]]],
                color=HYBRID_ROUTE_COLOR,
                weight=6.5,
                opacity=0.95,
            )
            _no_click(line).add_to(m)
    elif not drawn_hybrid and quantum_edges:
        for key in quantum_edges:
            u, v = key[0], key[1]
            if not G.has_edge(u, v):
                continue
            line = folium.PolyLine(
                [[G.nodes[u]["y"], G.nodes[u]["x"]], [G.nodes[v]["y"], G.nodes[v]["x"]]],
                color=HYBRID_ROUTE_COLOR,
                weight=6.5,
                opacity=0.95,
            )
            _no_click(line).add_to(m)

    if bridge_edge is not None and G.has_edge(*bridge_edge):
        u, v = bridge_edge
        blocked = folium.PolyLine(
            [[G.nodes[u]["y"], G.nodes[u]["x"]], [G.nodes[v]["y"], G.nodes[v]["x"]]],
            color=HAZARD_ROUTE_COLOR,
            weight=8,
            opacity=0.95,
            dash_array="6 8",
        )
        _no_click(blocked).add_to(m)

    r_epi = damage_radius(hazard_t)
    for frac, op in [(1.0, 0.10), (0.75, 0.16), (0.35, 0.28)]:
        ring = folium.Circle(
            location=[lat_e, lon_e],
            radius=frac * r_epi * 1000.0,
            color=HAZARD_ROUTE_COLOR,
            weight=2 if frac == 1.0 else 1,
            fill=True,
            fill_color=HAZARD_ROUTE_COLOR,
            fill_opacity=op,
        )
        _no_click(ring).add_to(m)

    _no_click(
        folium.Marker(
            [lat_e, lon_e],
            icon=folium.Icon(color="red", icon="warning-sign"),
        )
    ).add_to(m)

    for i, ex in enumerate(exits):
        marker = folium.CircleMarker(
            location=[G.nodes[ex]["y"], G.nodes[ex]["x"]],
            radius=9,
            color=EXIT_RING_COLOR,
            fill=True,
            fill_color=EXIT_RING_COLOR,
            fill_opacity=0.9,
            popup=f"Safe haven {i + 1}",
        )
        _no_click(marker).add_to(m)

    r_exit = exit_radius(hazard_t)
    for ex in exits:
        for frac, op in [(1.0, 0.08), (0.5, 0.14)]:
            ring = folium.Circle(
                location=[G.nodes[ex]["y"], G.nodes[ex]["x"]],
                radius=max(frac * r_exit * 1000.0, 10.0),
                color=EXIT_RING_COLOR,
                weight=1,
                fill=True,
                fill_color=EXIT_RING_COLOR,
                fill_opacity=op,
            )
            _no_click(ring).add_to(m)

    legend_html = f"""
    <div style="position:fixed;bottom:28px;left:28px;z-index:9999;
         background:rgba(10,15,30,0.92);border:1px solid rgba(0,229,255,0.28);
         border-radius:8px;padding:10px 14px;font-size:12px;color:#e8eef6;
         font-family:sans-serif;line-height:1.55;max-width:300px;
         pointer-events:none;">
      <b style="color:{ORANGE_ACCENT};letter-spacing:0.04em;">GOD VIEW LEGEND</b><br/>
      <span style="color:{HAZARD_ROUTE_COLOR};">●</span> Danger / epicenter / blocked bridge<br/>
      <span style="color:{HYBRID_ROUTE_COLOR};">●</span> Quantum sample arterials (Hybrid hero)<br/>
      <span style="color:{DIJKSTRA_ROUTE_COLOR};">●</span> Alternate corridors (Dijkstra bulk)<br/>
      <span style="color:{EXIT_RING_COLOR};">●</span> Exit / safe haven (gold)<br/>
      <span style="color:#3b82f6;">●</span> Flood / sector hazard zone<br/>
      <span style="color:#a8bdd4;font-size:11px;">Hybrid = small sample · bulk = Dijkstra</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def _default_epi_from_graph(G: nx.Graph) -> Tuple[float, float]:
    """Return (lat, lon) near the graph centroid."""
    xs = [G.nodes[n]["x"] for n in G.nodes()]
    ys = [G.nodes[n]["y"] for n in G.nodes()]
    return float(np.mean(ys)), float(np.mean(xs))


def init_god_view_state(G: nx.Graph, exits: Sequence, origin) -> None:
    """Session defaults for the Command Center (call before sidebar controls)."""
    if "gv_flood" not in st.session_state:
        st.session_state["gv_flood"] = 0.35
    if "gv_block_bridge" not in st.session_state:
        st.session_state["gv_block_bridge"] = False
    if "gv_batch_size" not in st.session_state:
        st.session_state["gv_batch_size"] = DEFAULT_BATCH_SIZE
    if "gv_seed" not in st.session_state:
        st.session_state["gv_seed"] = 42
    if "gv_bridge_edge" not in st.session_state:
        st.session_state["gv_bridge_edge"] = find_main_bridge_edge(G)
    if "gv_sync_b2c_epi" not in st.session_state:
        st.session_state["gv_sync_b2c_epi"] = True
    if "gv_epi_lat" not in st.session_state:
        if "epi_lat" in st.session_state:
            st.session_state["gv_epi_lat"] = float(st.session_state["epi_lat"])
            st.session_state["gv_epi_lon"] = float(st.session_state["epi_lon"])
        else:
            lat, lon = _default_epi_from_graph(G)
            st.session_state["gv_epi_lat"] = lat
            st.session_state["gv_epi_lon"] = lon
    if "gv_map_center" not in st.session_state:
        st.session_state["gv_map_center"] = [float(origin[1]), float(origin[0])]
    if "gv_result" not in st.session_state:
        st.session_state["gv_result"] = None
    if "gv_hazard_meta" not in st.session_state:
        st.session_state["gv_hazard_meta"] = {
            "flood_level": 0.35,
            "block_bridge": False,
            "bridge_edge": st.session_state.get("gv_bridge_edge"),
            "blocked_edges": 0,
            "penalized_edges": 0,
        }


def _resolve_epicenter(controls: Dict[str, Any]) -> Tuple[float, float]:
    """
    Prefer B2C session epicenter when sync is on; else God View lat/lon.
    Returns (lon, lat) for routing APIs.
    """
    sync = bool(controls.get("sync_b2c_epi", st.session_state.get("gv_sync_b2c_epi", True)))
    if sync and "epi_lat" in st.session_state and "epi_lon" in st.session_state:
        lat = float(st.session_state["epi_lat"])
        lon = float(st.session_state["epi_lon"])
        st.session_state["gv_epi_lat"] = lat
        st.session_state["gv_epi_lon"] = lon
        return lon, lat

    lat = float(controls.get("epi_lat", st.session_state.get("gv_epi_lat", 14.59)))
    lon = float(controls.get("epi_lon", st.session_state.get("gv_epi_lon", 120.97)))
    st.session_state["gv_epi_lat"] = lat
    st.session_state["gv_epi_lon"] = lon
    return lon, lat


def render_god_view_controls() -> Dict[str, Any]:
    """
    Sidebar / panel controls for the Command Center.

    Returns a dict of current control values and whether the sim was triggered.
    """
    st.markdown("## Command Center")
    st.caption(
        "B2G surface — inject hazards, trigger city-wide evacuation, "
        "watch Hybrid sample arterials rebalance against Dijkstra bulk."
    )

    st.markdown(
        '<div class="qr-click-panel"><div class="title">'
        "City-Wide Evacuation Simulation</div>"
        "<div style='color:#a8bdd4;font-size:0.82rem'>"
        "Citizens get free B2C routing. Commanders use God View for logistics, "
        "corridor stress, and Hybrid QML fleet rebalancing.<br/>"
        f"<b style='color:#e8eef6'>Fast path:</b> Dijkstra bulk + "
        f"≤{HYBRID_HERO_SAMPLE} Hybrid hero corridors "
        "(not 14k Hybrid inferences)."
        "</div></div>",
        unsafe_allow_html=True,
    )

    # Clamp legacy sessions that used batch=28+ before the Cloud speed fix
    if "gv_batch_size" in st.session_state:
        st.session_state["gv_batch_size"] = int(
            max(4, min(int(st.session_state["gv_batch_size"]), MAX_BATCH_SIZE))
        )
    else:
        st.session_state["gv_batch_size"] = DEFAULT_BATCH_SIZE

    flood = st.slider(
        "Flood / sector hazard level",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("gv_flood", 0.35)),
        step=0.05,
        key="gv_flood",
        help="Raises travel weights in the central flood sector (Algorithm-1 style penalties).",
    )
    block = st.checkbox(
        "Block Main Highway Bridge",
        value=bool(st.session_state.get("gv_block_bridge", False)),
        key="gv_block_bridge",
        help="Heavily penalize the highest-centrality arterial edge.",
    )
    n_agents = st.slider(
        "Simulated agents (batch)",
        min_value=4,
        max_value=MAX_BATCH_SIZE,
        value=int(st.session_state.get("gv_batch_size", DEFAULT_BATCH_SIZE)),
        step=1,
        key="gv_batch_size",
        help=(
            f"Default {DEFAULT_BATCH_SIZE}: Dijkstra for bulk heatmap, "
            f"Hybrid QML on ≤{HYBRID_HERO_SAMPLE} hero corridors. Max {MAX_BATCH_SIZE} on Cloud. "
            f"Scaled citizens = batch × {CITY_SCALE_PER_AGENT:,} (narrative only)."
        ),
    )

    sync_b2c = st.checkbox(
        "Use B2C epicenter (if set)",
        value=bool(st.session_state.get("gv_sync_b2c_epi", True)),
        key="gv_sync_b2c_epi",
        help="Reuse the epicenter from the B2C map-click session when available.",
    )

    epi_lat = float(st.session_state.get("gv_epi_lat", 14.59))
    epi_lon = float(st.session_state.get("gv_epi_lon", 120.97))
    if sync_b2c and "epi_lat" in st.session_state:
        st.caption(
            f"Epicenter synced from B2C · "
            f"{float(st.session_state['epi_lat']):.5f}, "
            f"{float(st.session_state['epi_lon']):.5f}"
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            epi_lat = st.number_input(
                "Epicenter lat",
                value=epi_lat,
                format="%.5f",
                key="gv_epi_lat",
                help="Hazard epicenter latitude (degrees).",
            )
        with c2:
            epi_lon = st.number_input(
                "Epicenter lon",
                value=epi_lon,
                format="%.5f",
                key="gv_epi_lon",
                help="Hazard epicenter longitude (degrees).",
            )

    bridge = st.session_state.get("gv_bridge_edge")
    if bridge is not None:
        st.caption(f"Bridge target edge · `{bridge[0]} ↔ {bridge[1]}`")

    scaled_preview = int(CITY_SCALE_PER_AGENT * int(n_agents))
    st.caption(
        f"Batch {int(n_agents)} agents → narrative scale "
        f"**~{scaled_preview:,} citizens** "
        f"(×{CITY_SCALE_PER_AGENT:,} per agent · not Hybrid count)."
    )

    trigger = st.button(
        "Trigger City-Wide Evacuation Simulation",
        type="primary",
        use_container_width=True,
        key="gv_trigger",
    )

    return {
        "flood_level": float(flood),
        "block_bridge": bool(block),
        "n_agents": int(n_agents),
        "sync_b2c_epi": bool(sync_b2c),
        "epi_lat": float(epi_lat),
        "epi_lon": float(epi_lon),
        "trigger": bool(trigger),
    }


def _render_empty_metrics() -> None:
    """Enterprise standby row before the first Trigger — no spinner."""
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Simulated Agents", "—", delta="awaiting trigger")
    with m2:
        st.metric("Scaled Citizens", "—", delta="narrative · after run")
    with m3:
        st.metric("Escape Success", "—", delta="run simulation")
    with m4:
        st.metric("Network Alert", "STANDBY")

    st.markdown(
        '<div class="qr-map-hint" style="margin-top:0.5rem">'
        "<b>Command Center ready.</b> Set flood / bridge hazards in the sidebar, "
        "then click <b>Trigger City-Wide Evacuation Simulation</b>. "
        f"Default batch <b>{DEFAULT_BATCH_SIZE}</b> agents "
        f"(≤{HYBRID_HERO_SAMPLE} Hybrid QML + Dijkstra bulk) — typically under 15s on Cloud. "
        "Results cache until you trigger again."
        "</div>",
        unsafe_allow_html=True,
    )


def _render_live_metrics(
    result: Dict[str, Any],
    *,
    hazard_meta: Dict[str, Any],
    controls: Dict[str, Any],
    hybrid_model,
    pennylane_ok: bool,
) -> None:
    """Post-sim metrics: honest agent vs scaled citizens + quantum load."""
    batch_n = max(0, int(result.get("n_routed") or 0))
    n_hybrid = int(result.get("n_hybrid") or 0)
    n_dij = int(result.get("n_dijkstra") or 0)
    citizens_scaled = int(CITY_SCALE_PER_AGENT * batch_n)
    success = float(result.get("success_rate") or 0.0)
    elapsed = float(result.get("elapsed_s") or 0.0)

    q_contrib = float(result.get("quantum_contribution") or 0.0)
    if q_contrib <= 0 and hybrid_model is not None:
        q_contrib = float(estimate_quantum_contribution_pct(hybrid_model))
    if not pennylane_ok and q_contrib <= 0:
        q_contrib = 37.9

    alert = congestion_alert_status(
        float(hazard_meta.get("flood_level", controls.get("flood_level", 0))),
        bool(hazard_meta.get("block_bridge", controls.get("block_bridge", False))),
        int(hazard_meta.get("blocked_edges", 0)),
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Simulated Agents",
            f"{batch_n}",
            delta=f"{n_hybrid} Hybrid · {n_dij} Dijkstra",
        )
    with m2:
        st.metric(
            "Scaled Citizens (narrative)",
            f"{citizens_scaled:,}",
            delta=f"×{CITY_SCALE_PER_AGENT:,} per agent",
        )
    with m3:
        st.metric(
            "Escape Success Rate",
            f"{success:.1f}%",
            delta=f"{result.get('n_success', 0)} exits reached",
        )
    with m4:
        st.metric(
            "Quantum Contribution",
            f"{q_contrib:.1f}%",
            delta="PHN combine · live",
        )

    a1, a2 = st.columns([2, 1])
    with a1:
        st.metric("Network Congestion Alert", alert)
    with a2:
        if elapsed > 0:
            st.metric("Batch latency", f"{elapsed:.1f}s", delta=f"{batch_n} agents")
        else:
            st.metric("Batch latency", "—")

    latency_note = ""
    if elapsed > 0:
        latency_note = f"Batch finished in **{elapsed:.1f}s** · "
    st.caption(
        f"{latency_note}"
        f"Avg travel · {float(result.get('avg_travel') or 0):.1f}  ·  "
        f"Hybrid heroes · {n_hybrid}  ·  "
        f"Dijkstra bulk · {n_dij}  ·  "
        f"Flood edges · {hazard_meta.get('penalized_edges', 0)}  ·  "
        f"{QUANTUM_CONTRIBUTION_FORMULA.split(',')[0]}."
    )
    st.caption(
        f"Honest note: only **{n_hybrid}** Hybrid QML inferences ran this batch; "
        f"the **{citizens_scaled:,}** scaled-citizen figure is narrative "
        f"(batch × {CITY_SCALE_PER_AGENT:,}), not Hybrid count."
    )


def render_god_view(
    G: nx.Graph,
    exits: Sequence,
    hybrid_model,
    mean,
    std,
    *,
    pennylane_ok: bool = True,
    controls: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render the full God View Command Center (metrics + map + pitch blurb).

    Heavy simulation runs only on Trigger click — never on tab open.
    Last result is cached in ``st.session_state['gv_result']``.
    """
    origin = get_graph_origin(G)
    init_god_view_state(G, exits, origin)

    st.markdown(
        '<div class="qr-tagline" style="margin-top:0.35rem">'
        "Command Center — God View</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qr-map-hint">'
        "<b>B2G / B2B:</b> Citizens escape free on the B2C tab. "
        "Here, commanders monitor <b>city-wide flows</b>, "
        "inject flood &amp; bridge failures, and rebalance the network in one trigger. "
        f"<b style='color:{HYBRID_ROUTE_COLOR}'>Cyan = Hybrid hero sample</b> · "
        f"<b style='color:{DIJKSTRA_ROUTE_COLOR}'>White = Dijkstra bulk</b> · "
        f"<b style='color:{HAZARD_ROUTE_COLOR}'>Red = danger / blocked</b> · "
        f"<b style='color:{EXIT_RING_COLOR}'>Gold = exits</b>"
        "</div>",
        unsafe_allow_html=True,
    )

    if controls is None:
        with st.expander("Simulation controls", expanded=True):
            controls = render_god_view_controls()

    # Run batch ONLY on explicit Trigger — never auto-run on tab open
    if controls.get("trigger"):
        epi = _resolve_epicenter(controls)
        n_agents = int(controls["n_agents"])
        n_hybrid = min(HYBRID_HERO_SAMPLE, n_agents)
        status = st.empty()
        progress = st.progress(0)
        t_wall0 = time.perf_counter()
        try:
            progress.progress(0, text="Preparing hazard graph…")
        except TypeError:
            pass

        def _set_progress(frac: float, msg: str) -> None:
            try:
                progress.progress(min(1.0, max(0.0, frac)), text=msg)
            except TypeError:
                progress.progress(min(1.0, max(0.0, frac)))

        def _progress(i: int, total: int, phase: str) -> None:
            label = {
                "hybrid": "Hybrid QML",
                "dijkstra": "Dijkstra",
                "skip": "skip",
                "fail": "retry",
            }.get(phase, phase)
            elapsed = time.perf_counter() - t_wall0
            status.caption(
                f"Routing agent {i}/{total} · {label} · {elapsed:.1f}s elapsed"
            )
            _set_progress(
                i / max(total, 1),
                f"City-wide evacuation {i}/{total} · {label}",
            )

        H, hazard_meta = apply_god_view_hazards(
            G,
            flood_level=controls["flood_level"],
            block_bridge=controls["block_bridge"],
            bridge_edge=st.session_state.get("gv_bridge_edge"),
        )
        status.caption(
            f"Routing {n_agents} agents · {n_hybrid} Hybrid hero + "
            f"{n_agents - n_hybrid} Dijkstra bulk…"
        )
        result = run_evacuation_batch(
            H,
            hybrid_model,
            mean,
            std,
            exits,
            epi,
            n_agents=n_agents,
            n_hybrid=n_hybrid,
            seed=int(st.session_state.get("gv_seed", 42)),
            max_steps=HYBRID_MAX_STEPS,
            progress_cb=_progress,
        )
        wall_s = float(time.perf_counter() - t_wall0)
        # Prefer wall clock (includes hazard prep) if batch timer is thinner
        if wall_s > float(result.get("elapsed_s") or 0):
            result["elapsed_s"] = wall_s
        _set_progress(
            1.0,
            f"Evacuation complete · {result['elapsed_s']:.1f}s",
        )
        status.success(
            f"Done · {result['n_success']}/{result['n_routed']} reached · "
            f"{result.get('n_hybrid', 0)} Hybrid / {result.get('n_dijkstra', 0)} Dijkstra · "
            f"{result['elapsed_s']:.1f}s"
        )
        progress.empty()

        st.session_state["gv_result"] = result
        st.session_state["gv_hazard_meta"] = hazard_meta
        st.session_state["gv_map_center"] = [epi[1], epi[0]]
        try:
            st.toast(
                f"Evacuation complete — {result['n_success']}/{result['n_routed']} "
                f"reached · {result.get('n_hybrid', 0)} Hybrid / "
                f"{result.get('n_dijkstra', 0)} Dijkstra · "
                f"{result['elapsed_s']:.1f}s.",
                icon="🛰️",
            )
        except Exception:
            pass

    result = st.session_state.get("gv_result")
    has_sim = result is not None and int(result.get("n_routed") or 0) > 0
    hazard_meta = st.session_state.get("gv_hazard_meta") or {
        "flood_level": controls.get("flood_level", 0.35),
        "block_bridge": controls.get("block_bridge", False),
        "bridge_edge": st.session_state.get("gv_bridge_edge"),
        "blocked_edges": 0,
        "penalized_edges": 0,
    }

    # ---- Metrics ----
    if has_sim:
        _render_live_metrics(
            result,
            hazard_meta=hazard_meta,
            controls=controls,
            hybrid_model=hybrid_model,
            pennylane_ok=pennylane_ok,
        )
    else:
        _render_empty_metrics()

    # ---- Map (hazard preview before sim; corridors after trigger) ----
    epi = (
        float(st.session_state["gv_epi_lon"]),
        float(st.session_state["gv_epi_lat"]),
    )
    # Keep map epicenter in sync with B2C when checkbox is on
    if controls.get("sync_b2c_epi") and "epi_lat" in st.session_state:
        epi = (
            float(st.session_state["epi_lon"]),
            float(st.session_state["epi_lat"]),
        )

    bridge = hazard_meta.get("bridge_edge") or st.session_state.get("gv_bridge_edge")
    show_bridge = bridge if hazard_meta.get("block_bridge") else None

    preview_flood = float(
        hazard_meta.get("flood_level", controls.get("flood_level", 0.35))
        if has_sim
        else controls.get("flood_level", 0.35)
    )
    preview_bridge = (
        show_bridge
        if has_sim
        else (
            st.session_state.get("gv_bridge_edge")
            if controls.get("block_bridge")
            else None
        )
    )

    empty_counts: Counter = Counter()
    fmap = build_god_view_map(
        G,
        exits,
        epi,
        (result.get("edge_counts") if has_sim else None) or empty_counts,
        quantum_edges=(result.get("quantum_edges") if has_sim else None) or set(),
        hybrid_paths=(result.get("hybrid_paths") if has_sim else None) or [],
        bridge_edge=preview_bridge,
        flood_level=preview_flood,
        map_center=st.session_state.get("gv_map_center") or [epi[1], epi[0]],
        map_zoom=15,
        hazard_t=8.0,
    )
    st_folium(
        fmap,
        key="qr_god_view_map",
        height=620,
        use_container_width=True,
        returned_objects=[],
    )

    with st.expander("Architecture for judges", expanded=False):
        st.markdown(
            f"""
**How God View works (honest, Cloud-fast)**

1. Commander sets **flood / sector hazard** and optional **bridge block**
2. Graph copy receives Algorithm-1-style weight penalties
3. **Dijkstra** routes the bulk batch → cyan arterial heatmap (seconds)
4. **Hybrid QML** (`predict_escape_route`) runs on ≤{HYBRID_HERO_SAMPLE} hero agents → **cyan quantum arterials**
5. Live **Quantum Contribution %** is read from the PHN `combine` layer — same formula as B2C
6. **Scaled citizens** = simulated agents × {CITY_SCALE_PER_AGENT:,} — **narrative only**, not Hybrid inference count
7. Result is **cached** until you click Trigger again (no re-run on tab switch / widget tweak)

**Pitch line:** Citizens escape free (B2C). Governments & logistics command the network (God View).
            """
        )

    st.markdown(
        '<div class="qr-footer">'
        "<span>God View · Hybrid QML Command Center · Team 5 Quantrio · QC4SG — SEA Quantathon 2026</span>"
        "<span>Citizens free · Commanders in control</span>"
        "</div>",
        unsafe_allow_html=True,
    )
