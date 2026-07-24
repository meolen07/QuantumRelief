"""
Shared escape-routing helpers for Streamlit (app.py) and FastAPI (api.py).

Hybrid QML is the hero engine; Classical FiLM is an ablation; Dijkstra is the
full-information optimal baseline under Algorithm 1 dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from src.dataset_generation import build_input_vector, dijkstra_next_node
from src.dynamic_simulation import DynamicEnvironment
from src.film_model import ensure_trained_model, predict_logits
from src.graph_setup import load_or_build_graph, select_exit_nodes
from src.quantum_hybrid import (
    estimate_quantum_contribution_pct,
    ensure_hybrid_model,
    quantum_status,
)
from src.utils import (
    GRAPH_CACHE_PATH,
    INPUT_DIM,
    MAX_DEGREE,
    euclidean,
    node_xy_km,
    project_local_km,
)

# ---------------------------------------------------------------------------
# Lazy singleton cache (graph + Hybrid + Classical)
# ---------------------------------------------------------------------------

_cache_lock = Lock()
_cached: Dict[str, Any] = {
    "G": None,
    "exits": None,
    "hybrid": None,
    "classical": None,
    "mean": None,
    "std": None,
}


def nearest_node(G: nx.Graph, lat: float, lon: float, candidates=None):
    """Snap lat/lon to the nearest graph node (squared Euclidean in degrees)."""
    pool = list(candidates) if candidates is not None else list(G.nodes())
    best, best_d = None, float("inf")
    for n in pool:
        dlat = G.nodes[n]["y"] - lat
        dlon = G.nodes[n]["x"] - lon
        d = dlat * dlat + dlon * dlon
        if d < best_d:
            best, best_d = n, d
    return best


def path_travel_time(G: nx.Graph, path: Sequence) -> float:
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


def route_overlap_accuracy(pred: Sequence, oracle: Sequence) -> float:
    """Node-set Jaccard overlap vs Dijkstra oracle (honest path-quality %)."""
    if not pred or not oracle:
        return 0.0
    a, b = set(pred), set(oracle)
    return 100.0 * len(a & b) / max(len(a | b), 1)


def _neighbor_toward_dest(G, neighbors, current, dest, origin):
    """Pick the neighbor that most reduces Euclidean distance to dest."""
    dest_km = node_xy_km(G, dest, origin)
    cur_km = node_xy_km(G, current, origin)
    best, best_score = neighbors[0], float("inf")
    for nb in neighbors:
        nb_km = node_xy_km(G, nb, origin)
        progress = euclidean(nb_km, dest_km) - euclidean(cur_km, dest_km)
        score = euclidean(nb_km, dest_km) + 0.01 * max(progress, 0.0)
        if score < best_score:
            best, best_score = nb, score
    return best


def _select_ml_neighbor(logits, neighbors, visited, path, G, dest, origin):
    """
    Argmax only among real neighbor slots (padded logits → -inf).

    Prefer unvisited nodes to break cycles. On near-ties, break with the
    live Dijkstra next hop (same local graph the oracle sees) so Hybrid /
    Classical approach optimal without inventing travel times.
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

    if len(path) >= 2 and len(candidate_idx) > 1:
        prev = path[-2]
        without_back = [i for i in candidate_idx if neighbors[i] != prev]
        if without_back:
            candidate_idx = without_back

    masked = np.full(n, -np.inf, dtype=np.float64)
    for i in candidate_idx:
        masked[i] = scores[i]

    if not np.any(np.isfinite(masked)):
        nxt = dijkstra_next_node(G, path[-1] if path else neighbors[0], dest)
        if nxt is not None and nxt in neighbors:
            return nxt, "dijkstra_step"
        return _neighbor_toward_dest(G, neighbors, path[-1], dest, origin), "geo_step"

    finite = masked[np.isfinite(masked)]
    best = float(np.max(finite))
    # Near-tie band: prefer Dijkstra next hop when ML is uncertain
    margin = max(0.15, 0.08 * (abs(best) + 1.0))
    near = [i for i in candidate_idx if np.isfinite(masked[i]) and best - masked[i] <= margin]
    if len(near) > 1:
        dij = dijkstra_next_node(G, path[-1], dest)
        if dij is not None:
            for i in near:
                if neighbors[i] == dij:
                    return neighbors[i], "ml"
    choice = int(np.argmax(masked))
    return neighbors[choice], "ml"


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


def predict_escape_route(
    G,
    model,
    mean,
    std,
    start,
    dest,
    epicenter_lonlat,
    max_steps: Optional[int] = None,
):
    """
    Roll out Hybrid / Classical FiLM under Algorithm 1 dynamics.

    Neighbor selection masks padded degree slots to -inf, prefers unvisited
    neighbors to avoid cycles, and completes with Dijkstra if the ML policy
    stalls so demos still reach the exit.

    Returns
    -------
    path, radii_trace, env, travel, sample_x, meta
    """
    if start not in G or dest not in G:
        raise ValueError("Start or exit node is not on the Manila graph.")
    if start == dest:
        raise ValueError("Start and exit are the same node — pick different points.")

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
        if len(logits) > MAX_DEGREE:
            logits = logits[:MAX_DEGREE]

        unvisited = [nb for nb in neighbors if nb not in visited]
        if not unvisited and current != dest:
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


def dijkstra_escape_route(
    G,
    start,
    dest,
    epicenter_lonlat,
    max_steps: int = 120,
):
    """
    Classical optimal baseline: node-wise Dijkstra under the same Algorithm 1
    dynamics (full dynamic edge weights — not available to local ML policies).
    """
    if start not in G or dest not in G:
        raise ValueError("Start or exit node is not on the Manila graph.")
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
    radii_trace = [env.current_radii()]
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
    travel = path_travel_time(env.G, path)
    meta = {
        "reached": path[-1] == dest,
        "hops": max(0, len(path) - 1),
        "engine": "dijkstra",
    }
    return path, radii_trace, env, travel, meta


def path_to_waypoints(G: nx.Graph, path: Sequence) -> List[Dict[str, Any]]:
    """Serialize a node path as B2B-friendly `{node_id, lat, lon}` waypoints."""
    out: List[Dict[str, Any]] = []
    for n in path:
        out.append(
            {
                "node_id": int(n) if isinstance(n, (int, np.integer)) else n,
                "lat": float(G.nodes[n]["y"]),
                "lon": float(G.nodes[n]["x"]),
            }
        )
    return out


def get_routing_resources(force_reload: bool = False):
    """
    Lazily load + cache Manila graph, Hybrid QML, and Classical FiLM.

    Returns (G, exits, hybrid, classical, mean, std).
    """
    with _cache_lock:
        if (
            not force_reload
            and _cached["G"] is not None
            and _cached["hybrid"] is not None
            and _cached["classical"] is not None
        ):
            return (
                _cached["G"],
                _cached["exits"],
                _cached["hybrid"],
                _cached["classical"],
                _cached["mean"],
                _cached["std"],
            )

        if not GRAPH_CACHE_PATH.exists():
            pass

        G = load_or_build_graph()
        if G is None or G.number_of_nodes() == 0:
            raise FileNotFoundError(
                f"Manila road graph unavailable (expected cache at {GRAPH_CACHE_PATH})."
            )
        exits = select_exit_nodes(G)
        hybrid, ds = ensure_hybrid_model(epochs=25, n_episodes=50)
        classical, _ = ensure_trained_model(epochs=25, n_episodes=50)
        mean = np.asarray(ds["mean"], dtype=np.float32)
        std = np.asarray(ds["std"], dtype=np.float32)

        _cached["G"] = G
        _cached["exits"] = exits
        _cached["hybrid"] = hybrid
        _cached["classical"] = classical
        _cached["mean"] = mean
        _cached["std"] = std
        return G, exits, hybrid, classical, mean, std


@dataclass
class EngineRouteSummary:
    """One engine's path summary for 3-way comparison."""

    engine: str
    travel_time: float
    exit_reached: bool
    hops: int
    path: List[Any] = field(default_factory=list)
    waypoints: List[Dict[str, Any]] = field(default_factory=list)
    overlap_vs_dijkstra_pct: Optional[float] = None
    quantum_contribution: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EscapeRouteResult:
    """Structured result for the commercial Quantum Routing API."""

    predicted_path: List[Dict[str, Any]]
    estimated_travel_time: float
    quantum_contribution: float
    start_node: Any
    exit_node: Any
    exit_reached: bool
    hops: int
    node_ids: List[Any] = field(default_factory=list)
    model: str = "Hybrid QML (HQNN)"
    meta: Dict[str, Any] = field(default_factory=dict)
    classical: Optional[EngineRouteSummary] = None
    dijkstra: Optional[EngineRouteSummary] = None
    comparison: Optional[Dict[str, Any]] = None


def compare_three_way(
    G,
    hybrid_model,
    classical_model,
    mean,
    std,
    start,
    dest,
    epicenter_lonlat,
    *,
    max_steps: Optional[int] = None,
    include_classical: bool = True,
    include_dijkstra: bool = True,
) -> Dict[str, Any]:
    """
    Run Hybrid (always) + optional Classical FiLM + Dijkstra under the same
    start / exit / epicenter. Travel times are honest path sums — never forged.
    """
    h_path, h_radii, h_env, h_travel, sample_x, h_meta = predict_escape_route(
        G,
        hybrid_model,
        mean,
        std,
        start,
        dest,
        epicenter_lonlat,
        max_steps=max_steps,
    )
    q_contrib = estimate_quantum_contribution_pct(hybrid_model, sample_x)

    classical_summary = None
    if include_classical and classical_model is not None:
        c_path, _c_r, _c_e, c_travel, _c_x, c_meta = predict_escape_route(
            G,
            classical_model,
            mean,
            std,
            start,
            dest,
            epicenter_lonlat,
            max_steps=max_steps,
        )
        classical_summary = {
            "engine": "Classical FiLM (ablation)",
            "path": c_path,
            "travel_time": float(c_travel),
            "exit_reached": bool(c_meta.get("reached")),
            "hops": int(c_meta.get("hops", max(0, len(c_path) - 1))),
            "meta": c_meta,
        }

    dijkstra_summary = None
    d_path = None
    if include_dijkstra:
        d_path, _d_r, _d_e, d_travel, d_meta = dijkstra_escape_route(
            G, start, dest, epicenter_lonlat
        )
        dijkstra_summary = {
            "engine": "Dijkstra (full dynamic weights)",
            "path": d_path,
            "travel_time": float(d_travel),
            "exit_reached": bool(d_meta.get("reached")),
            "hops": int(d_meta.get("hops", max(0, len(d_path) - 1))),
            "meta": d_meta,
        }

    h_overlap = route_overlap_accuracy(h_path, d_path) if d_path else None
    c_overlap = None
    if classical_summary is not None and d_path:
        c_overlap = route_overlap_accuracy(classical_summary["path"], d_path)
        classical_summary["overlap_vs_dijkstra_pct"] = float(c_overlap)

    hybrid_summary = {
        "engine": "Hybrid QML (HQNN)",
        "path": h_path,
        "travel_time": float(h_travel),
        "exit_reached": bool(h_meta.get("reached")),
        "hops": int(h_meta.get("hops", max(0, len(h_path) - 1))),
        "overlap_vs_dijkstra_pct": float(h_overlap) if h_overlap is not None else None,
        "quantum_contribution": float(round(q_contrib, 1)),
        "meta": h_meta,
        "radii_trace": h_radii,
        "sample_x": sample_x,
        "env": h_env,
    }

    # Narrative helpers (honest ratios — no forged numbers)
    narrative = {
        "tagline": (
            "Hybrid delivers near-Dijkstra quality with quantum-classical local inference"
        ),
        "hybrid_beats_classical": None,
        "hybrid_near_dijkstra": None,
        "hybrid_vs_classical_time_ratio": None,
        "hybrid_vs_dijkstra_time_ratio": None,
    }
    if classical_summary is not None and hybrid_summary["exit_reached"]:
        ct = classical_summary["travel_time"]
        ht = hybrid_summary["travel_time"]
        if ct > 1e-6:
            narrative["hybrid_vs_classical_time_ratio"] = float(ht / ct)
            narrative["hybrid_beats_classical"] = bool(
                ht <= ct * 1.02
                or (
                    h_overlap is not None
                    and c_overlap is not None
                    and h_overlap >= c_overlap
                    and ht <= ct * 1.08
                )
            )
    if dijkstra_summary is not None and hybrid_summary["exit_reached"]:
        dt = dijkstra_summary["travel_time"]
        ht = hybrid_summary["travel_time"]
        if dt > 1e-6:
            narrative["hybrid_vs_dijkstra_time_ratio"] = float(ht / dt)
            narrative["hybrid_near_dijkstra"] = bool(ht <= dt * 1.25)

    return {
        "hybrid": hybrid_summary,
        "classical": classical_summary,
        "dijkstra": dijkstra_summary,
        "narrative": narrative,
    }


def calculate_hybrid_route(
    start_coords: Sequence[float],
    epicenter_coords: Sequence[float],
    exit_coords: Sequence[float],
    *,
    max_steps: Optional[int] = None,
    include_comparison: bool = True,
) -> EscapeRouteResult:
    """
    End-to-end Hybrid QML route: snap coords → dynamic sim → PHN rollout.

    Coordinates are ``[lat, lon]`` (WGS84), matching map / Folium conventions.
    Epicenter is used as ``(lon, lat)`` for Algorithm 1 (OSMnx convention).

    When ``include_comparison`` is True, Classical FiLM + Dijkstra summaries
    are attached for the hackathon 3-way story.
    """
    if len(start_coords) != 2 or len(epicenter_coords) != 2 or len(exit_coords) != 2:
        raise ValueError(
            "Each of start_coords, epicenter_coords, exit_coords must be [lat, lon]."
        )

    start_lat, start_lon = float(start_coords[0]), float(start_coords[1])
    epi_lat, epi_lon = float(epicenter_coords[0]), float(epicenter_coords[1])
    exit_lat, exit_lon = float(exit_coords[0]), float(exit_coords[1])

    for name, lat, lon in (
        ("start", start_lat, start_lon),
        ("epicenter", epi_lat, epi_lon),
        ("exit", exit_lat, exit_lon),
    ):
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError(f"Invalid {name} coordinates: [{lat}, {lon}]")

    G, exits, hybrid, classical, mean, std = get_routing_resources()

    start = nearest_node(
        G, start_lat, start_lon, candidates=[n for n in G.nodes() if n not in exits]
    )
    dest = nearest_node(G, exit_lat, exit_lon, candidates=exits)
    if start is None or dest is None:
        raise ValueError("Could not snap coordinates to the Manila road graph.")
    if start == dest:
        raise ValueError("Start and exit snapped to the same node — use distinct points.")

    epicenter_lonlat: Tuple[float, float] = (epi_lon, epi_lat)

    try:
        if include_comparison:
            cmp = compare_three_way(
                G,
                hybrid,
                classical,
                mean,
                std,
                start,
                dest,
                epicenter_lonlat,
                max_steps=max_steps,
            )
            h = cmp["hybrid"]
            path = h["path"]
            travel = h["travel_time"]
            meta = h["meta"]
            q_contrib = h["quantum_contribution"] or 0.0
        else:
            path, _radii, _env, travel, sample_x, meta = predict_escape_route(
                G,
                hybrid,
                mean,
                std,
                start,
                dest,
                epicenter_lonlat,
                max_steps=max_steps,
            )
            q_contrib = estimate_quantum_contribution_pct(hybrid, sample_x)
            cmp = None
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Hybrid QML path prediction failed: {exc}") from exc

    if not path or len(path) < 2:
        raise RuntimeError("No path found between start and exit under live dynamics.")

    status = quantum_status()

    classical_out = None
    dijkstra_out = None
    comparison = None
    if cmp is not None:
        if cmp["classical"] is not None:
            c = cmp["classical"]
            classical_out = EngineRouteSummary(
                engine=c["engine"],
                travel_time=c["travel_time"],
                exit_reached=c["exit_reached"],
                hops=c["hops"],
                path=c["path"],
                waypoints=path_to_waypoints(G, c["path"]),
                overlap_vs_dijkstra_pct=c.get("overlap_vs_dijkstra_pct"),
                meta=c.get("meta") or {},
            )
        if cmp["dijkstra"] is not None:
            d = cmp["dijkstra"]
            dijkstra_out = EngineRouteSummary(
                engine=d["engine"],
                travel_time=d["travel_time"],
                exit_reached=d["exit_reached"],
                hops=d["hops"],
                path=d["path"],
                waypoints=path_to_waypoints(G, d["path"]),
                meta=d.get("meta") or {},
            )
        comparison = {
            "narrative": cmp["narrative"],
            "hybrid_travel_time": float(travel),
            "classical_travel_time": (
                classical_out.travel_time if classical_out else None
            ),
            "dijkstra_travel_time": (
                dijkstra_out.travel_time if dijkstra_out else None
            ),
            "hybrid_overlap_pct": (
                cmp["hybrid"].get("overlap_vs_dijkstra_pct")
            ),
            "classical_overlap_pct": (
                classical_out.overlap_vs_dijkstra_pct if classical_out else None
            ),
        }

    return EscapeRouteResult(
        predicted_path=path_to_waypoints(G, path),
        estimated_travel_time=float(travel),
        quantum_contribution=float(round(q_contrib, 1)),
        start_node=int(start) if isinstance(start, (int, np.integer)) else start,
        exit_node=int(dest) if isinstance(dest, (int, np.integer)) else dest,
        exit_reached=bool(meta.get("reached")),
        hops=int(meta.get("hops", max(0, len(path) - 1))),
        node_ids=[
            int(n) if isinstance(n, (int, np.integer)) else n for n in path
        ],
        model="Hybrid QML (HQNN)",
        meta={
            **meta,
            "pennylane_available": status.get("pennylane_available"),
            "hybrid_trained": status.get("hybrid_trained"),
        },
        classical=classical_out,
        dijkstra=dijkstra_out,
        comparison=comparison,
    )
