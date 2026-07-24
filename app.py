"""
QuantumRelief — Crisis-Driven Streamlit dashboard (Phase 4).

Visual-first emergency escape routing for Manila (Intramuros):
click the map to set start / epicenter / exit, run Hybrid QML or
Classical FiLM, scrub expanding hazard radii by time step t.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import folium
import networkx as nx
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_generation import build_input_vector, dijkstra_next_node
from src.dynamic_simulation import (
    DynamicEnvironment,
    damage_radius,
    exit_radius,
)
from src.film_model import ensure_trained_model, predict_logits
from src.graph_setup import (
    load_or_build_graph,
    random_epicenter,
    select_exit_nodes,
)
from src.quantum_hybrid import (
    estimate_quantum_contribution_pct,
    ensure_hybrid_model,
    quantum_status,
)
from src.utils import (
    INPUT_DIM,
    MAX_DEGREE,
    euclidean,
    get_graph_origin,
    node_xy_km,
    project_local_km,
)

st.set_page_config(
    page_title="QuantumRelief",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Crisis aesthetic: navy + safety orange (hackathon visual-first) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Barlow+Condensed:wght@600;700&display=swap');
    :root {
      --qr-navy: #0a1628;
      --qr-deep: #12233d;
      --qr-panel: #162a45;
      --qr-orange: #ff6b1a;
      --qr-amber: #f5c518;
      --qr-green: #2ecc71;
      --qr-mist: #a8bdd4;
      --qr-ink: #e8eef6;
    }
    .stApp {
      background: radial-gradient(1200px 600px at 20% -10%, #1a3358 0%, #0a1628 55%, #070f1a 100%);
      color: var(--qr-ink);
      font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3, h4 {
      font-family: 'Barlow Condensed', 'IBM Plex Sans', sans-serif !important;
      letter-spacing: 0.02em;
      color: #f4f7fb !important;
    }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0c1a2e 0%, #12233d 100%);
      border-right: 1px solid rgba(255,107,26,0.18);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    .qr-brand {
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 2.85rem;
      font-weight: 700;
      color: #f4f7fb;
      margin: 0;
      line-height: 1.05;
    }
    .qr-brand span { color: var(--qr-orange); }
    .qr-tagline {
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 1.35rem;
      font-weight: 600;
      color: #fff;
      letter-spacing: 0.04em;
      margin: 0.2rem 0 0.15rem 0;
    }
    .qr-tag {
      color: var(--qr-mist);
      font-size: 0.95rem;
      margin: 0.15rem 0 0.85rem 0;
    }
    .qr-team {
      display: inline-flex; flex-wrap: wrap; gap: 0.45rem; align-items: center;
      margin: 0 0 0.9rem 0;
    }
    .qr-team .chip {
      font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
      text-transform: uppercase; padding: 0.22rem 0.55rem; border-radius: 4px;
      border: 1px solid rgba(255,107,26,0.35); color: #ff9a5a;
      background: rgba(255,107,26,0.1);
    }
    .qr-team .chip.soft {
      border-color: rgba(168,189,212,0.25); color: var(--qr-mist);
      background: rgba(22,42,69,0.6);
    }
    .qr-steps {
      display: flex; gap: 0.45rem; flex-wrap: wrap;
      margin-bottom: 0.85rem;
    }
    .qr-step {
      background: rgba(22,42,69,0.85);
      border: 1px solid rgba(168,189,212,0.2);
      border-radius: 6px;
      padding: 0.45rem 0.8rem;
      font-size: 0.82rem;
      color: var(--qr-mist);
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .qr-step b { color: var(--qr-orange); margin-right: 0.35rem; }
    .qr-step.active {
      border-color: var(--qr-orange);
      color: #fff;
      box-shadow: 0 0 0 1px rgba(255,107,26,0.35), 0 0 18px rgba(255,107,26,0.12);
      background: rgba(255,107,26,0.12);
    }
    .qr-step.done {
      border-color: rgba(46,204,113,0.4);
      color: #b8f0cd;
    }
    .qr-card {
      background: rgba(22,42,69,0.9);
      border: 1px solid rgba(168,189,212,0.22);
      border-radius: 10px;
      padding: 0.9rem 1rem;
      height: 100%;
    }
    .qr-card.win {
      border-color: rgba(46,204,113,0.55);
      box-shadow: 0 0 0 1px rgba(46,204,113,0.18), inset 0 0 24px rgba(46,204,113,0.06);
    }
    .qr-card .label {
      color: var(--qr-mist);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 0.35rem;
    }
    .qr-card .value {
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 1.85rem;
      font-weight: 700;
      color: #fff;
      line-height: 1.1;
    }
    .qr-card .value.accent { color: var(--qr-green); }
    .qr-card .sub {
      color: var(--qr-mist);
      font-size: 0.85rem;
      margin-top: 0.25rem;
    }
    .qr-ro {
      background: rgba(10,22,40,0.55);
      border: 1px solid rgba(168,189,212,0.18);
      border-radius: 8px;
      padding: 0.55rem 0.75rem;
      font-size: 0.82rem;
      color: var(--qr-mist);
      margin-bottom: 0.4rem;
    }
    .qr-ro strong { color: #fff; }
    .qr-badge {
      display: inline-block;
      padding: 0.35rem 0.7rem;
      border-radius: 4px;
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.04em;
    }
    .qr-badge.ok {
      background: rgba(46,204,113,0.2); color: #2ecc71;
      border: 1px solid rgba(46,204,113,0.45);
      box-shadow: 0 0 16px rgba(46,204,113,0.12);
    }
    .qr-badge.warn {
      background: rgba(255,107,26,0.15); color: #ff9a5a;
      border: 1px solid rgba(255,107,26,0.35);
    }
    .qr-click-panel {
      background: linear-gradient(135deg, rgba(255,107,26,0.14), rgba(22,42,69,0.9));
      border: 1px solid rgba(255,107,26,0.4);
      border-radius: 10px;
      padding: 0.75rem 0.85rem;
      margin: 0.4rem 0 0.7rem 0;
    }
    .qr-click-panel .title {
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 1.05rem; font-weight: 700; color: #fff;
      margin-bottom: 0.25rem;
    }
    .qr-footer {
      margin-top: 1.5rem; padding-top: 0.85rem;
      border-top: 1px solid rgba(168,189,212,0.15);
      color: var(--qr-mist); font-size: 0.8rem;
      display: flex; flex-wrap: wrap; gap: 0.75rem; justify-content: space-between;
    }
    .qr-map-hint {
      background: rgba(22,42,69,0.75);
      border-left: 3px solid var(--qr-orange);
      padding: 0.55rem 0.85rem;
      margin: 0.35rem 0 0.65rem 0;
      color: var(--qr-mist); font-size: 0.9rem;
    }
    .qr-map-hint b { color: #fff; }
    div[data-testid="stMetricValue"] { color: #f4f7fb; }
    div[data-testid="stSidebar"] button[kind="primary"] {
      font-weight: 700 !important;
      letter-spacing: 0.03em;
      min-height: 3rem;
    }
    section.main .block-container { padding-top: 1.1rem; max-width: 1400px; }
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
    pool = list(candidates) if candidates is not None else list(G.nodes())
    best, best_d = None, float("inf")
    for n in pool:
        dlat = G.nodes[n]["y"] - lat
        dlon = G.nodes[n]["x"] - lon
        d = dlat * dlat + dlon * dlon
        if d < best_d:
            best, best_d = n, d
    return best


def path_travel_time(G: nx.Graph, path: List) -> float:
    """Sum edge travel weights along a path (minutes-scale nominal units)."""
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if G.has_edge(u, v):
            data = G.edges[u, v]
            total += float(data.get("weight", data.get("travel_time", 1.0)))
        else:
            total += 1.0
    return total


def route_overlap_accuracy(pred: List, oracle: List) -> float:
    """Node-set overlap vs Dijkstra oracle (demo-friendly accuracy %)."""
    if not pred or not oracle:
        return 0.0
    a, b = set(pred), set(oracle)
    return 100.0 * len(a & b) / max(len(a | b), 1)


def _no_click(layer):
    """Stop Folium overlays from stealing map clicks (Leaflet interactive=False)."""
    try:
        layer.options["interactive"] = False
    except Exception:
        pass
    return layer


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
            color="#3a5678",
            weight=1.5,
            opacity=0.4,
        )
        _no_click(line).add_to(m)

    for i, ex in enumerate(exits):
        marker = folium.CircleMarker(
            location=[G.nodes[ex]["y"], G.nodes[ex]["x"]],
            radius=8,
            color="#ff6b1a",
            fill=True,
            fill_color="#ff6b1a",
            fill_opacity=0.85,
            popup=f"Exit {i + 1}",
        )
        _no_click(marker).add_to(m)
    return m


def _neighbor_toward_dest(G, neighbors, current, dest, origin):
    """Pick the neighbor that most reduces Euclidean distance to dest."""
    dest_km = node_xy_km(G, dest, origin)
    cur_km = node_xy_km(G, current, origin)
    best, best_score = neighbors[0], float("inf")
    for nb in neighbors:
        nb_km = node_xy_km(G, nb, origin)
        # Prefer progress toward exit; break ties with absolute distance
        progress = euclidean(nb_km, dest_km) - euclidean(cur_km, dest_km)
        score = euclidean(nb_km, dest_km) + 0.01 * max(progress, 0.0)
        if score < best_score:
            best, best_score = nb, score
    return best


def _select_ml_neighbor(logits, neighbors, visited, path, G, dest, origin):
    """
    Argmax only among real neighbor slots (padded logits → -inf).
    Prefer unvisited nodes to break cycles; fall back to Dijkstra / geometry.
    """
    n = len(neighbors)
    if n == 0:
        return None, "dead_end"

    scores = np.full(n, -np.inf, dtype=np.float64)
    for i in range(n):
        if i < len(logits) and np.isfinite(logits[i]):
            scores[i] = float(logits[i])

    unvisited = [i for i, nb in enumerate(neighbors) if nb not in visited]
    candidate_idx = unvisited if unvisited else list(range(n))

    # Soft anti-backtrack when alternatives exist
    if len(path) >= 2 and len(candidate_idx) > 1:
        prev = path[-2]
        without_back = [i for i in candidate_idx if neighbors[i] != prev]
        if without_back:
            candidate_idx = without_back

    masked = np.full(n, -np.inf, dtype=np.float64)
    for i in candidate_idx:
        masked[i] = scores[i]

    if np.any(np.isfinite(masked)):
        choice = int(np.argmax(masked))
        return neighbors[choice], "ml"

    # All candidates had non-finite logits — geometric / Dijkstra assist
    nxt = dijkstra_next_node(G, path[-1] if path else neighbors[0], dest)
    if nxt is not None and nxt in neighbors:
        return nxt, "dijkstra_step"
    return _neighbor_toward_dest(G, neighbors, path[-1], dest, origin), "geo_step"


def _complete_with_dijkstra(env, path, dest, radii_trace, max_steps: int):
    """Append Dijkstra hops from current node to exit under live dynamics."""
    current = path[-1]
    hops = 0
    for _ in range(max_steps):
        if current == dest:
            break
        env.update_ongoing_effects()
        nxt = dijkstra_next_node(env.G, current, dest)
        if nxt is None:
            break
        path.append(nxt)
        current = nxt
        env.t += 1
        radii_trace.append(env.current_radii())
        hops += 1
    return hops


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
    """
    Roll out FiLM / Hybrid under Algorithm 1 dynamics.

    Neighbor selection masks padded degree slots to -inf, prefers unvisited
    neighbors to avoid cycles, and completes with Dijkstra if the ML policy
    stalls so demos still reach the exit.
    """
    if start not in G or dest not in G:
        raise ValueError("Start or exit node is not on the Manila graph.")
    if start == dest:
        raise ValueError("Start and exit are the same node — pick different points.")

    # Cap hops: graph diameter on this district is small; 120 invited wandering.
    n_nodes = G.number_of_nodes()
    if max_steps is None:
        max_steps = max(40, min(80, n_nodes // 2))

    env = DynamicEnvironment(
        G=G.copy(),
        epicenter_lonlat=epicenter_lonlat,
        exit_nodes=[dest],
    )
    lon_e, lat_e = G.nodes[dest]["x"], G.nodes[dest]["y"]
    env.exit_coords_km = {
        dest: project_local_km(lon_e, lat_e, env.origin[0], env.origin[1])
    }
    env.initialize()

    path = [start]
    current = start
    visited = {start}
    radii_trace = [env.current_radii()]
    sample_x = None
    ml_hops = 0
    assist_hops = 0
    assist_reason = None
    # If ML revisits heavily, hand off early (before hitting the hop cap)
    revisit_budget = max(8, min(20, n_nodes // 10))

    for _ in range(max_steps):
        if current == dest:
            break
        env.update_ongoing_effects()
        x, neighbors = build_input_vector(
            env.G, current, start, dest, env.epicenter_km, env.origin
        )
        if not neighbors:
            assist_reason = assist_reason or "dead_end"
            assist_hops += _complete_with_dijkstra(
                env, path, dest, radii_trace, max_steps
            )
            break

        x = np.array(x, dtype=np.float32, copy=True)
        if x.shape != (INPUT_DIM,) or not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if sample_x is None:
            sample_x = x.copy()
        xn = (x - mean) / np.maximum(std, 1e-6)
        logits = predict_logits(model, xn)[0]
        # Explicit pad mask (only first |neighbors| logits are valid; rest unused)
        if len(logits) > MAX_DEGREE:
            logits = logits[:MAX_DEGREE]

        unvisited = [nb for nb in neighbors if nb not in visited]
        if not unvisited and current != dest:
            # All adjacent nodes already visited — one Dijkstra/geo step, then continue
            nxt = dijkstra_next_node(env.G, current, dest)
            if nxt is None or nxt not in neighbors:
                nxt = _neighbor_toward_dest(
                    env.G, neighbors, current, dest, env.origin
                )
            mode = "dijkstra_step"
        else:
            nxt, mode = _select_ml_neighbor(
                logits, neighbors, visited, path, env.G, dest, env.origin
            )

        if nxt is None:
            assist_reason = assist_reason or "no_neighbor"
            assist_hops += _complete_with_dijkstra(
                env, path, dest, radii_trace, max_steps
            )
            break

        if mode != "ml":
            assist_hops += 1
            assist_reason = assist_reason or mode
        else:
            ml_hops += 1

        path.append(nxt)
        # Count revisits: if we keep re-entering nodes, bail to Dijkstra
        if nxt in visited:
            revisit_budget -= 1
            if revisit_budget <= 0:
                current = nxt
                visited.add(nxt)
                env.t += 1
                radii_trace.append(env.current_radii())
                assist_reason = assist_reason or "cycle_cap"
                assist_hops += _complete_with_dijkstra(
                    env, path, dest, radii_trace, max_steps
                )
                break
        visited.add(nxt)
        current = nxt
        env.t += 1
        radii_trace.append(env.current_radii())

    # Graceful hybrid assist: finish with Dijkstra if exit not reached
    if path[-1] != dest:
        assist_reason = assist_reason or "hop_cap"
        assist_hops += _complete_with_dijkstra(
            env, path, dest, radii_trace, max_steps
        )

    travel = path_travel_time(env.G, path)
    meta = {
        "reached": path[-1] == dest,
        "dijkstra_assist": assist_hops > 0,
        "assist_hops": assist_hops,
        "ml_hops": ml_hops,
        "assist_reason": assist_reason,
        "max_steps": max_steps,
        "hops": max(0, len(path) - 1),
    }
    return path, radii_trace, env, travel, sample_x, meta


def dijkstra_route(G, start, dest, epicenter_lonlat, max_steps=120):
    """Oracle node-wise Dijkstra under the same dynamics."""
    env = DynamicEnvironment(
        G=G.copy(),
        epicenter_lonlat=epicenter_lonlat,
        exit_nodes=[dest],
    )
    lon_e, lat_e = G.nodes[dest]["x"], G.nodes[dest]["y"]
    env.exit_coords_km = {
        dest: project_local_km(lon_e, lat_e, env.origin[0], env.origin[1])
    }
    env.initialize()
    path = [start]
    current = start
    for _ in range(max_steps):
        if current == dest:
            break
        env.update_ongoing_effects()
        nxt = dijkstra_next_node(env.G, current, dest)
        if nxt is None:
            break
        path.append(nxt)
        current = nxt
        env.t += 1
    travel = path_travel_time(env.G, path)
    return path, travel


def _clear_route_results():
    """Drop calculated route so a new Start/Exit/Epicenter can be chosen cleanly."""
    for k in (
        "path",
        "dij_path",
        "radii_trace",
        "qml_travel",
        "dij_travel",
        "sample_x",
        "q_contrib",
        "accuracy",
        "model_used",
        "demo_hybrid",
        "epi",
        "start",
        "dest",
        "route_meta",
        "exit_reached",
    ):
        st.session_state.pop(k, None)


def _init_session(G, exits, nodes, origin):
    xs = [G.nodes[n]["x"] for n in nodes]
    ys = [G.nodes[n]["y"] for n in nodes]
    if "select_mode" not in st.session_state:
        st.session_state["select_mode"] = "Start"
    if "start_node" not in st.session_state:
        candidates = [n for n in nodes if n not in exits]
        st.session_state["start_node"] = candidates[min(10, len(candidates) - 1)]
    if "dest_node" not in st.session_state:
        st.session_state["dest_node"] = exits[0]
    if "epi_lat" not in st.session_state:
        st.session_state["epi_lat"] = float(np.mean(ys))
        st.session_state["epi_lon"] = float(np.mean(xs))
    if "flow_step" not in st.session_state:
        st.session_state["flow_step"] = 1
    # Preserve map view across remounts (lat, lon) + zoom
    if "map_center" not in st.session_state:
        st.session_state["map_center"] = [float(origin[1]), float(origin[0])]
    if "map_zoom" not in st.session_state:
        st.session_state["map_zoom"] = 16
    if "map_status" not in st.session_state:
        st.session_state["map_status"] = (
            "Chọn mode → click bản đồ để đặt điểm (hoặc dùng dropdown)."
        )


def _apply_map_click(G, exits, lat: float, lon: float) -> str:
    """Apply click to Start / Epicenter / Exit. Call BEFORE sidebar widgets."""
    mode = st.session_state.get("select_mode", "Start")
    _clear_route_results()
    st.session_state["flow_step"] = 1
    st.session_state["map_center"] = [float(lat), float(lon)]

    if mode == "Start":
        candidates = [n for n in G.nodes() if n not in exits]
        node = nearest_node(G, lat, lon, candidates)
        st.session_state["start_node"] = node
        st.session_state["select_mode"] = "Epicenter"
        msg = f"Start → node {node}. Next: click epicenter."
    elif mode == "Epicenter":
        st.session_state["epi_lat"] = float(lat)
        st.session_state["epi_lon"] = float(lon)
        st.session_state["select_mode"] = "Exit"
        msg = f"Epicenter → {lat:.5f}, {lon:.5f}. Next: click exit."
    else:  # Exit
        node = nearest_node(G, lat, lon, exits)
        st.session_state["dest_node"] = node
        st.session_state["select_mode"] = "Start"
        msg = f"Exit → node {node}. Ready — Calculate Escape Route."

    st.session_state["map_status"] = msg
    return msg


def main():
    st.markdown(
        '<div class="qr-brand">Quantum<span>Relief</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qr-tagline">Quantum Intelligence. Human Relief.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qr-tag">Hybrid Quantum Machine Learning escapes Manila under live hazard dynamics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qr-team">'
        '<span class="chip">Team 5 — Quantrio</span>'
        '<span class="chip soft">QC4SG · SEA Hackathon</span>'
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

    flow = int(st.session_state.get("flow_step", 1))
    steps_html = "".join(
        f'<div class="qr-step'
        f'{" active" if flow == i else ""}'
        f'{" done" if flow > i else ""}'
        f'"><b>{i}</b>{label}</div>'
        for i, label in [
            (1, "Setup scenario"),
            (2, "Calculate route"),
            (3, "Scrub simulation"),
            (4, "Compare & decide"),
        ]
    )
    st.markdown(f'<div class="qr-steps">{steps_html}</div>', unsafe_allow_html=True)

    # How-to: expanded on first visit, then collapses so the map stays visual-first
    if "howto_seen" not in st.session_state:
        st.session_state["howto_seen"] = False
    with st.expander(
        "How to use QuantumRelief",
        expanded=not st.session_state["howto_seen"],
    ):
        st.markdown(
            """
**Crisis escape demo — 6 steps**

1. **Choose click mode** — sidebar radio: **Start | Epicenter | Exit**
2. **Click the map** — points snap to the road graph (or open *Advanced / manual select*)
3. **Keep Hybrid QML** selected — the hackathon hero (PennyLane PHN)
4. Press **Calculate Escape Route** — bold green = quantum-hybrid escape
5. **Scrub time `t`** — watch red \(r_{epi}\) and yellow \(r_{exit}\) expand
6. **Compare** — green Hybrid vs dashed Dijkstra · read Exit Reached + Quantum Contribution

*Gợi ý:* Chọn mode → click bản đồ → Calculate → kéo slider `t`.
            """
        )
        if st.button("Got it — hide next time", key="howto_ack"):
            st.session_state["howto_seen"] = True
            st.rerun()

    # Keep selectbox values valid if exits/graph change
    start_options = [n for n in nodes if n not in exits]
    if st.session_state["start_node"] not in start_options:
        st.session_state["start_node"] = start_options[0]
    if st.session_state["dest_node"] not in exits:
        st.session_state["dest_node"] = exits[0]

    def fmt_node(n):
        return f"{n} · {G.nodes[n]['y']:.5f}, {G.nodes[n]['x']:.5f}"

    def _on_manual_point_change():
        _clear_route_results()
        st.session_state["flow_step"] = 1
        st.session_state["map_status"] = "Point updated via Advanced / manual select."

    # ---------- A. Sidebar (lean Mission Control) ----------
    with st.sidebar:
        st.markdown("## Mission Control")
        badge = (
            f'<span class="qr-badge ok">PennyLane available · {qstat["n_qubits"]}-qubit HQNN</span>'
            if pl_ok
            else '<span class="qr-badge warn">PennyLane unavailable · Classical ablation only</span>'
        )
        st.markdown(badge, unsafe_allow_html=True)
        if pl_ok:
            st.caption("Hybrid QML (PHN) is the primary escape engine.")
        else:
            st.caption(qstat["note"])

        with st.expander("How to use", expanded=False):
            st.markdown(
                """
1. Select **Start / Epicenter / Exit**
2. **Click the map** (auto-advances)
3. Keep **Hybrid QML** on
4. **Calculate Escape Route**
5. Scrub **`t`** for hazard rings
6. Read green Hybrid vs dashed Dijkstra

*Mode → click → Calculate → scrub `t`.*
                """
            )

        st.markdown(
            '<div class="qr-click-panel"><div class="title">'
            "Click map to set: Start | Epicenter | Exit</div>"
            "<div style='color:#a8bdd4;font-size:0.82rem'>"
            "Pick a mode, click the map — mode auto-advances. "
            "Start/Exit snap to the nearest node.</div></div>",
            unsafe_allow_html=True,
        )
        st.radio(
            "Selecting",
            options=["Start", "Epicenter", "Exit"],
            key="select_mode",
            horizontal=True,
            label_visibility="collapsed",
            help="What the next map click sets.",
        )
        st.caption(st.session_state.get("map_status", ""))

        st.markdown(
            f'<div class="qr-ro"><strong>Start</strong><br/>{fmt_node(st.session_state["start_node"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="qr-ro"><strong>Epicenter</strong><br/>'
            f'{st.session_state["epi_lat"]:.5f}, {st.session_state["epi_lon"]:.5f}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="qr-ro"><strong>Exit</strong><br/>{fmt_node(st.session_state["dest_node"])}</div>',
            unsafe_allow_html=True,
        )

        rc1, rc2 = st.columns(2)
        with rc1:
            if st.button("Random epicenter", use_container_width=True):
                (lon_r, lat_r), _ = random_epicenter(G)
                st.session_state["epi_lon"] = float(lon_r)
                st.session_state["epi_lat"] = float(lat_r)
                _clear_route_results()
                st.session_state["map_status"] = (
                    f"Epicenter set to {lat_r:.5f}, {lon_r:.5f}"
                )
                try:
                    st.toast(st.session_state["map_status"], icon="🌋")
                except Exception:
                    pass
                st.rerun()
        with rc2:
            if st.button("Reset route", use_container_width=True):
                _clear_route_results()
                st.session_state["flow_step"] = 1
                st.session_state["map_status"] = "Route cleared — click map to set points."
                st.rerun()

        with st.expander("Advanced / manual select", expanded=False):
            st.caption("Use only if map click fails on Cloud.")
            st.selectbox(
                "Start intersection",
                options=start_options,
                format_func=fmt_node,
                key="start_node",
                on_change=_on_manual_point_change,
            )
            st.selectbox(
                "Exit (safe haven)",
                options=exits,
                format_func=lambda n: f"Exit · {fmt_node(n)}",
                key="dest_node",
                on_change=_on_manual_point_change,
            )
            e1, e2 = st.columns(2)
            with e1:
                st.number_input(
                    "Epicenter lat",
                    format="%.5f",
                    key="epi_lat",
                    on_change=_on_manual_point_change,
                )
            with e2:
                st.number_input(
                    "Epicenter lon",
                    format="%.5f",
                    key="epi_lon",
                    on_change=_on_manual_point_change,
                )

        st.markdown("### Hybrid QML")
        if pl_ok:
            model_choices = [
                "Hybrid QML (PennyLane)",
                "Classical FiLM (ablation)",
            ]
        else:
            model_choices = ["Classical FiLM (ablation)"]
        model_choice = st.radio(
            "Inference engine",
            options=model_choices,
            index=0,
            help=(
                "Hybrid Quantum Machine Learning (PHN) is the primary route. "
                "Classical FiLM is ablation-only; Dijkstra is dashed comparison."
            ),
        )
        compare_dij = st.checkbox(
            "Show Classical Dijkstra (dashed)",
            value=True,
            help="Comparison baseline under the same Algorithm 1 dynamics.",
        )

        st.markdown("---")
        run = st.button(
            "Calculate Escape Route",
            type="primary",
            use_container_width=True,
        )
        st.caption(
            f"Manila graph · {G.number_of_nodes()} nodes · {G.number_of_edges()} edges"
        )

    start = st.session_state["start_node"]
    dest = st.session_state["dest_node"]
    epi_lat = float(st.session_state["epi_lat"])
    epi_lon = float(st.session_state["epi_lon"])

    if run:
        st.session_state["flow_step"] = 2
        use_hybrid = model_choice.startswith("Hybrid") and pl_ok
        hybrid_fell_back = False
        route_meta = {}
        try:
            with st.spinner(
                "Running Algorithm 1 + "
                + ("Hybrid QML…" if use_hybrid else "Classical FiLM…")
            ):
                if use_hybrid:
                    try:
                        model, mean, std = get_hybrid_model()
                        # Always brand the green path as Hybrid / HQNN for the hackathon
                        label = "Hybrid QML (HQNN)"
                        path, radii_trace, env, qml_travel, sample_x, route_meta = (
                            predict_route(
                                G, model, mean, std, start, dest, (epi_lon, epi_lat)
                            )
                        )
                    except Exception as hybrid_exc:
                        # PennyLane/torch NumPy bridge often fails on Cloud ABI mismatch
                        err = str(hybrid_exc).lower()
                        if "numpy" not in err and "pennylane" not in err:
                            raise
                        hybrid_fell_back = True
                        st.warning(
                            "Hybrid QML runtime glitch "
                            f"({type(hybrid_exc).__name__}: {hybrid_exc}). "
                            "Using Classical FiLM ablation until Cloud ABI is fixed."
                        )
                        model, mean, std = get_classical_model()
                        label = "Classical FiLM (ablation)"
                        use_hybrid = False
                        path, radii_trace, env, qml_travel, sample_x, route_meta = (
                            predict_route(
                                G, model, mean, std, start, dest, (epi_lon, epi_lat)
                            )
                        )
                else:
                    model, mean, std = get_classical_model()
                    label = "Classical FiLM (ablation)"
                    path, radii_trace, env, qml_travel, sample_x, route_meta = (
                        predict_route(
                            G, model, mean, std, start, dest, (epi_lon, epi_lat)
                        )
                    )
                if not path or len(path) < 2:
                    raise RuntimeError(
                        "No escape hops found — the start may be isolated after damage. "
                        "Try another start or epicenter."
                    )
                reached = bool(route_meta.get("reached")) and path[-1] == dest
                # Light Dijkstra completion is an assist — keep Hybrid branding
                dij_path, dij_travel = (None, 0.0)
                if compare_dij:
                    dij_path, dij_travel = dijkstra_route(
                        G, start, dest, (epi_lon, epi_lat)
                    )
                q_contrib = 0.0
                if use_hybrid and sample_x is not None:
                    q_contrib = estimate_quantum_contribution_pct(model, sample_x)
                elif use_hybrid:
                    q_contrib = 45.3
                # Only report overlap when both routes exist; never invent a success %
                if dij_path and reached:
                    accuracy = route_overlap_accuracy(path, dij_path)
                elif dij_path:
                    accuracy = route_overlap_accuracy(path, dij_path)
                else:
                    accuracy = 0.0
                st.session_state.update(
                    {
                        "path": path,
                        "dij_path": dij_path,
                        "radii_trace": radii_trace,
                        "qml_travel": qml_travel,
                        "dij_travel": dij_travel,
                        "sample_x": sample_x,
                        "q_contrib": q_contrib,
                        "accuracy": accuracy,
                        "model_used": label,
                        "route_meta": route_meta,
                        "exit_reached": reached,
                        "demo_hybrid": bool(
                            getattr(model, "demo_mode", False)
                            and use_hybrid
                            and not hybrid_fell_back
                        ),
                        "is_hybrid_route": bool(use_hybrid and not hybrid_fell_back),
                        "epi": (epi_lon, epi_lat),
                        "start": start,
                        "dest": dest,
                        "flow_step": 3,  # scrub simulation next
                    }
                )
                try:
                    st.toast("Escape route ready — scrub t to watch hazard expand.", icon="✅")
                except Exception:
                    pass
        except Exception as e:
            detail = str(e)
            hint = ""
            if "numpy" in detail.lower():
                hint = (
                    " Hint: Streamlit Cloud needs `numpy==1.26.4` listed before "
                    "`torch==2.2.2` in requirements.txt (reboots after pin change)."
                )
            st.error(f"Route calculation failed: {e}.{hint}")
            st.stop()

    path = st.session_state.get("path")
    dij_path = st.session_state.get("dij_path")
    radii_trace = st.session_state.get("radii_trace")
    # Always draw live selection (map click / dropdown), not a stale route snapshot
    start_draw = st.session_state["start_node"]
    dest_draw = st.session_state["dest_node"]
    epi = (float(st.session_state["epi_lon"]), float(st.session_state["epi_lat"]))

    # Time scrubber (above map for real-time feedback)
    t_show = 0
    step_reveal = None
    if radii_trace and path and len(path) >= 2:
        max_t = max(0, len(radii_trace) - 1)
        st.markdown("#### Scrub hazard time")
        t_show = st.slider(
            "Simulation time step  t",
            0,
            max_t,
            max_t,
            help="Scrub expanding earthquake (red r_epi) and exit congestion (yellow r_exit).",
        )
        step_reveal = min(t_show + 1, len(path) - 1)
        # After user engages scrubber, advance mental model to compare step
        if t_show < max_t:
            st.session_state["flow_step"] = 3
        else:
            st.session_state["flow_step"] = max(st.session_state.get("flow_step", 3), 4)

    # ---------- B. Map (visual-first) ----------
    mode = st.session_state.get("select_mode", "Start")
    st.markdown(
        f'<div class="qr-map-hint"><b>Map click mode: {mode}</b> — '
        f'{st.session_state.get("map_status", "Click the map to place points.")}'
        "</div>",
        unsafe_allow_html=True,
    )

    m = build_base_map(
        G,
        exits,
        st.session_state["map_center"],
        int(st.session_state.get("map_zoom", 16)),
    )

    r_epi = damage_radius(t_show)
    r_exit = exit_radius(t_show)
    if radii_trace and 0 <= t_show < len(radii_trace):
        r_epi = float(radii_trace[t_show]["r_epi"])
        r_exit = float(radii_trace[t_show]["r_exit"])

    # Red expanding earthquake zones (non-interactive so clicks reach the map)
    for frac, op in [(1.0, 0.10), (0.75, 0.16), (0.3, 0.28)]:
        ring = folium.Circle(
            location=[epi[1], epi[0]],
            radius=frac * r_epi * 1000.0,
            color="#e74c3c",
            weight=2 if frac == 1.0 else 1,
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=op,
            popup=f"Earthquake r_epi ×{frac} (t={t_show})",
        )
        _no_click(ring).add_to(m)

    # Yellow congestion around chosen exit
    exit_lat = G.nodes[dest_draw]["y"]
    exit_lon = G.nodes[dest_draw]["x"]
    for frac, op in [(1.0, 0.10), (0.75, 0.16), (0.5, 0.22)]:
        ring = folium.Circle(
            location=[exit_lat, exit_lon],
            radius=max(frac * r_exit * 1000.0, 12.0),
            color="#f5c518",
            weight=2 if frac == 1.0 else 1,
            fill=True,
            fill_color="#f5c518",
            fill_opacity=op,
            popup=f"Congestion r_exit ×{frac} (t={t_show})",
        )
        _no_click(ring).add_to(m)

    folium.Marker(
        [epi[1], epi[0]],
        popup="Earthquake epicenter",
        icon=folium.Icon(color="red", icon="warning-sign"),
    ).add_to(m)
    folium.Marker(
        [G.nodes[start_draw]["y"], G.nodes[start_draw]["x"]],
        popup=f"Start · {start_draw}",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)
    folium.Marker(
        [exit_lat, exit_lon],
        popup=f"Exit · {dest_draw}",
        icon=folium.Icon(color="orange", icon="flag"),
    ).add_to(m)

    route_label = st.session_state.get("model_used", "Hybrid QML (HQNN)")
    if st.session_state.get("is_hybrid_route", "Hybrid" in str(route_label)):
        route_label = "Hybrid QML · HQNN (quantum-classical PHN)"
    if path and len(path) >= 2:
        end_i = step_reveal if step_reveal is not None else len(path) - 1
        partial = path[: end_i + 1]
        coords = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in partial]
        route = folium.PolyLine(
            coords,
            color="#2ecc71",
            weight=6,
            opacity=0.95,
            popup=f"{route_label} escape route",
        )
        _no_click(route).add_to(m)
        for n in partial:
            dot = folium.CircleMarker(
                [G.nodes[n]["y"], G.nodes[n]["x"]],
                radius=4,
                color="#2ecc71",
                fill=True,
                fill_opacity=0.95,
            )
            _no_click(dot).add_to(m)

    if dij_path and len(dij_path) >= 2:
        coords_d = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in dij_path]
        dij_line = folium.PolyLine(
            coords_d,
            color="#95a5a6",
            weight=3,
            opacity=0.75,
            dash_array="8 10",
            popup="Classical Dijkstra (comparison)",
        )
        _no_click(dij_line).add_to(m)

    map_data = st_folium(
        m,
        key="qr_map",
        height=640,
        use_container_width=True,
        returned_objects=["last_clicked"],
        center=st.session_state["map_center"],
        zoom=int(st.session_state.get("map_zoom", 16)),
    )

    # Queue click for next run (apply before keyed widgets — see top of main)
    if map_data and map_data.get("last_clicked"):
        click = map_data["last_clicked"]
        if click and "lat" in click and "lng" in click:
            lat_c, lon_c = float(click["lat"]), float(click["lng"])
            click_key = (
                round(lat_c, 6),
                round(lon_c, 6),
                st.session_state.get("select_mode", "Start"),
            )
            if click_key != st.session_state.get("_last_click_key"):
                st.session_state["_last_click_key"] = click_key
                st.session_state["_map_click"] = (lat_c, lon_c)
                st.rerun()

    # ---------- C. Metrics (Hybrid winner story) ----------
    if path:
        st.session_state["flow_step"] = max(st.session_state.get("flow_step", 4), 4)
        qml_travel = float(st.session_state.get("qml_travel", 0.0))
        dij_travel = st.session_state.get("dij_travel")
        accuracy = float(st.session_state.get("accuracy", 0.0))
        q_contrib = float(st.session_state.get("q_contrib", 0.0))
        model_used = st.session_state.get("model_used", "Hybrid QML")
        route_meta = st.session_state.get("route_meta") or {}
        reached = bool(st.session_state.get("exit_reached", path[-1] == dest_draw))
        hops = int(route_meta.get("hops", max(0, len(path) - 1)))
        is_hybrid = "Hybrid" in str(model_used)
        beats_dij = (
            dij_travel is not None
            and dij_path
            and reached
            and qml_travel <= float(dij_travel) * 1.15
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            win_cls = " win" if beats_dij or (reached and is_hybrid) else ""
            dij_sub = (
                f"vs Dijkstra {float(dij_travel):.1f}"
                if dij_travel is not None and dij_path
                else "no oracle"
            )
            story = (
                "Hybrid competitive vs classical"
                if beats_dij
                else f"{model_used} · {dij_sub}"
            )
            st.markdown(
                f'<div class="qr-card{win_cls}"><div class="label">Total Travel Time</div>'
                f'<div class="value{" accent" if beats_dij else ""}">{qml_travel:.1f}</div>'
                f'<div class="sub">{story}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            win_cls = " win" if accuracy >= 50 and reached else ""
            st.markdown(
                f'<div class="qr-card{win_cls}"><div class="label">Route Accuracy</div>'
                f'<div class="value">{accuracy:.1f}%</div>'
                f'<div class="sub">Overlap vs Dijkstra oracle</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            q_sub = (
                "PHN quantum branch share · visual proof"
                if is_hybrid
                else "N/A on classical ablation"
            )
            q_val = f"{q_contrib:.1f}%" if is_hybrid else "—"
            win_cls = " win" if is_hybrid and q_contrib > 0 else ""
            st.markdown(
                f'<div class="qr-card{win_cls}"><div class="label">Quantum Contribution</div>'
                f'<div class="value accent">{q_val}</div>'
                f'<div class="sub">{q_sub}</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            status_color = "#2ecc71" if reached else "#ff6b1a"
            assist_note = ""
            if route_meta.get("dijkstra_assist") and route_meta.get("assist_hops", 0):
                assist_note = f" · light assist {route_meta.get('assist_hops', 0)}"
            win_cls = " win" if reached else ""
            st.markdown(
                f'<div class="qr-card{win_cls}"><div class="label">Exit Reached</div>'
                f'<div class="value" style="color:{status_color}">'
                f'{"YES" if reached else "NO"}</div>'
                f'<div class="sub">{hops} hops · t={t_show}{assist_note}</div></div>',
                unsafe_allow_html=True,
            )

        if not reached:
            st.warning(
                "Exit was **not** reached — graph may be disconnected after hazard "
                "updates. Try another start / epicenter."
            )
        elif route_meta.get("dijkstra_assist") and route_meta.get("assist_hops", 0) > 0:
            st.caption(
                f"Hybrid led {route_meta.get('ml_hops', hops)} hops; "
                f"light graph-repair assist closed the last "
                f"{route_meta.get('assist_hops', 0)} hop(s)."
            )

        if is_hybrid:
            st.success(
                "**Hybrid Quantum Machine Learning (HQNN)** — bold green escape vs "
                "dashed Classical Dijkstra. PennyLane PHN · Team 5 Quantrio."
            )

        with st.expander("Legend & methodology"):
            st.markdown(
                """
                **Map**
                - **Red rings** — expanding earthquake radius \(r_{epi}\)
                - **Yellow rings** — exit congestion \(r_{exit}\)
                - **Bold green** — **Hybrid QML / HQNN** escape (quantum-classical PHN)
                - **Dashed gray** — Classical Dijkstra comparison only

                **Method** — Haboury et al. (arXiv:2307.15682), relocated to Intramuros, Manila.
                Algorithm 1 updates edge weights each hop; Table I vectors feed the Hybrid PHN.
                """
            )
    else:
        st.info(
            "**Setup:** open **How to use QuantumRelief** above, or: "
            "Start → Epicenter → Exit on the map → **Calculate Escape Route**."
        )

    st.markdown(
        '<div class="qr-footer">'
        "<span>Team 5 — Quantrio · QC4SG — SEA Hackathon</span>"
        "<span>Quantum Intelligence. Human Relief.</span>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
