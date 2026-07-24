"""
Phase 1 — Data Engineering & Graph Setup (Manila / Intramuros).

Downloads a compact drivable road network via OSMnx, caches to disk,
extracts edge features (speed, length, travel time), computes edge
betweenness centrality, and defines exit nodes + earthquake epicenter.
"""

from __future__ import annotations

import json
import random
from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .utils import (
    GRAPH_CACHE_PATH,
    GRAPH_META_PATH,
    MANILA_BBOX,
    MAX_DEGREE,
    Coord,
    ensure_dirs,
    get_graph_origin,
    node_xy,
    node_xy_km,
)

# Default speed (km/h) by highway type when OSM maxspeed is missing
DEFAULT_SPEEDS = {
    "motorway": 80,
    "trunk": 60,
    "primary": 50,
    "secondary": 40,
    "tertiary": 30,
    "residential": 20,
    "living_street": 15,
    "unclassified": 25,
    "service": 15,
    "default": 30,
}


def _highway_speed(highway) -> float:
    if isinstance(highway, list):
        highway = highway[0] if highway else "default"
    if highway is None:
        return DEFAULT_SPEEDS["default"]
    return float(DEFAULT_SPEEDS.get(str(highway), DEFAULT_SPEEDS["default"]))


def _parse_maxspeed(raw) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, list):
        raw = raw[0]
    try:
        # e.g. "40", "40 mph", "40|50"
        token = str(raw).split("|")[0].strip().split()[0]
        return float(token)
    except (ValueError, IndexError):
        return None


def _build_synthetic_manila_graph(n_nodes: int = 320, seed: int = 7) -> nx.Graph:
    """
    Offline fallback if OSMnx/Overpass is unreachable.

    Creates a planar-ish grid with random shortcuts over the Intramuros bbox,
    targeting paper-scale size (~357 nodes, max degree ≤ 5).
    """
    rng = np.random.default_rng(seed)
    bbox = MANILA_BBOX
    side = int(np.ceil(np.sqrt(n_nodes)))
    lons = np.linspace(bbox["west"], bbox["east"], side)
    lats = np.linspace(bbox["south"], bbox["north"], side)
    G = nx.Graph()
    node_id = 0
    grid = {}
    for i in range(side):
        for j in range(side):
            if node_id >= n_nodes:
                break
            G.add_node(node_id, x=float(lons[j]), y=float(lats[i]))
            grid[(i, j)] = node_id
            node_id += 1
        if node_id >= n_nodes:
            break

    def add_road(a, b):
        if a is None or b is None or a == b or G.has_edge(a, b):
            return
        if G.degree(a) >= MAX_DEGREE or G.degree(b) >= MAX_DEGREE:
            return
        xa, ya = G.nodes[a]["x"], G.nodes[a]["y"]
        xb, yb = G.nodes[b]["x"], G.nodes[b]["y"]
        # rough metres
        length = (
            ((xa - xb) * 111_320 * np.cos(np.radians(ya))) ** 2
            + ((ya - yb) * 111_320) ** 2
        ) ** 0.5
        G.add_edge(
            a,
            b,
            length=float(length),
            highway="residential",
            maxspeed="30",
        )

    for (i, j), n in grid.items():
        for di, dj in ((0, 1), (1, 0)):
            add_road(n, grid.get((i + di, j + dj)))
    # A few diagonal shortcuts
    keys = list(grid.keys())
    for _ in range(n_nodes // 8):
        k1, k2 = rng.choice(len(keys), size=2, replace=False)
        add_road(grid[keys[int(k1)]], grid[keys[int(k2)]])

    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    return G


def download_manila_graph(force: bool = False):
    """
    Download / load cached undirected drivable graph for Intramuros, Manila.

    Target size similar to the paper (~357 nodes, max degree ≤ 5).
    Falls back to a synthetic district graph if Overpass is unavailable.
    """
    ensure_dirs()
    if GRAPH_CACHE_PATH.exists() and not force:
        print(f"[QuantumRelief] Loading cached graph from {GRAPH_CACHE_PATH}")
        G = nx.read_graphml(GRAPH_CACHE_PATH)
        # GraphML stores node ids as strings; normalize
        G = nx.relabel_nodes(G, lambda n: int(n) if str(n).isdigit() else n)
        _coerce_numeric_attrs(G)
        return G

    bbox = MANILA_BBOX
    G = None
    try:
        import osmnx as ox

        print("[QuantumRelief] Downloading Manila (Intramuros) road network via OSMnx…")
        try:
            ox.settings.timeout = 300
        except Exception:
            pass

        bbox_tuple = (bbox["west"], bbox["south"], bbox["east"], bbox["north"])
        try:
            G = ox.graph_from_bbox(
                bbox_tuple,
                network_type="drive",
                simplify=True,
            )
        except TypeError:
            G = ox.graph_from_bbox(
                bbox["north"],
                bbox["south"],
                bbox["east"],
                bbox["west"],
                network_type="drive",
                simplify=True,
            )

        if hasattr(ox, "convert") and hasattr(ox.convert, "to_undirected"):
            G = ox.convert.to_undirected(G)
        else:
            G = ox.utils_graph.get_undirected(G)
        G = nx.Graph(G)
    except Exception as exc:
        print(f"[QuantumRelief] OSMnx download failed ({exc}); using synthetic graph.")
        G = _build_synthetic_manila_graph()

    # If still too large, take largest connected component only
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    # Cap degree by removing excess low-importance edges if needed
    G = _cap_max_degree(G, max_degree=MAX_DEGREE)

    annotate_edge_features(G)
    print("[QuantumRelief] Computing edge betweenness centrality…")
    compute_edge_betweenness(G)

    # Strip non-GraphML-serializable attributes (geometry lists, etc.)
    G = _sanitize_for_graphml(G)

    # Persist
    nx.write_graphml(G, GRAPH_CACHE_PATH)
    meta = {
        "place": "Intramuros, Manila, Philippines",
        "bbox": bbox,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "max_degree": max(dict(G.degree()).values()) if G.number_of_nodes() else 0,
        "min_degree": min(dict(G.degree()).values()) if G.number_of_nodes() else 0,
    }
    GRAPH_META_PATH.write_text(json.dumps(meta, indent=2))
    print(
        f"[QuantumRelief] Cached graph: {meta['n_nodes']} nodes, "
        f"{meta['n_edges']} edges, max_degree={meta['max_degree']}"
    )
    return G


def _coerce_numeric_attrs(G) -> None:
    """GraphML may store numbers as strings — coerce key attributes."""
    float_node = ("x", "y")
    for _, data in G.nodes(data=True):
        for k in float_node:
            if k in data:
                data[k] = float(data[k])
    float_edge = (
        "length",
        "speed_kph",
        "travel_time",
        "travel_time_nominal",
        "betweenness",
        "weight",
    )
    for _, _, data in G.edges(data=True):
        for k in float_edge:
            if k in data:
                try:
                    data[k] = float(data[k])
                except (TypeError, ValueError):
                    pass


def _sanitize_for_graphml(G: nx.Graph) -> nx.Graph:
    """Keep only scalar attributes that GraphML can serialize."""
    keep_node = {"x", "y", "street_count"}
    keep_edge = {
        "length",
        "speed_kph",
        "travel_time",
        "travel_time_nominal",
        "betweenness",
        "weight",
        "name",
        "highway",
        "maxspeed",
        "osmid",
    }
    H = nx.Graph()
    for n, data in G.nodes(data=True):
        clean = {}
        for k, v in data.items():
            if k not in keep_node:
                continue
            if isinstance(v, (list, dict, tuple)):
                continue
            clean[k] = v
        H.add_node(n, **clean)
    for u, v, data in G.edges(data=True):
        clean = {}
        for k, v_ in data.items():
            if k not in keep_edge:
                continue
            if isinstance(v_, (list, dict, tuple)):
                # flatten first element for common OSM list attrs
                if k in ("highway", "name", "maxspeed", "osmid") and v_:
                    clean[k] = str(v_[0])
                continue
            clean[k] = v_
        H.add_edge(u, v, **clean)
    return H


def _cap_max_degree(G: nx.Graph, max_degree: int = MAX_DEGREE) -> nx.Graph:
    """
    Ensure max node degree ≤ max_degree (paper: max degree 5).

    Preferentially drop longer edges until degree constraint is met,
    without disconnecting the graph when possible.
    """
    G = G.copy()
    changed = True
    while changed:
        changed = False
        for node, deg in sorted(G.degree(), key=lambda x: -x[1]):
            if deg <= max_degree:
                continue
            # Candidate edges sorted by length (drop longest first)
            neighbors = list(G.neighbors(node))
            edged = []
            for nb in neighbors:
                length = G.edges[node, nb].get("length", 1.0)
                if isinstance(length, list):
                    length = length[0]
                edged.append((float(length), nb))
            edged.sort(reverse=True)
            for _, nb in edged:
                if G.degree(node) <= max_degree:
                    break
                # Avoid disconnecting if possible
                G.remove_edge(node, nb)
                if not nx.is_connected(G):
                    G.add_edge(node, nb, length=_)
                else:
                    changed = True
    # Drop isolated nodes after capping
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    return G


def annotate_edge_features(G: nx.Graph) -> None:
    """Extract speed limit, length, nominal travel time (paper Sec. II B)."""
    for u, v, data in G.edges(data=True):
        length_m = data.get("length", 50.0)
        if isinstance(length_m, list):
            length_m = float(length_m[0])
        else:
            length_m = float(length_m)

        speed = _parse_maxspeed(data.get("maxspeed"))
        if speed is None:
            speed = _highway_speed(data.get("highway"))

        # travel time in hours → convert to minutes for nicer scale
        travel_time_min = (length_m / 1000.0) / max(speed, 1.0) * 60.0
        # Paper: sample from Gaussian around nominal for traffic variability
        data["length"] = length_m
        data["speed_kph"] = speed
        data["travel_time_nominal"] = travel_time_min
        data["travel_time"] = max(
            1e-4, float(np.random.normal(travel_time_min, 0.05 * travel_time_min))
        )
        data["weight"] = data["travel_time"]
        data.setdefault("betweenness", 0.0)


def compute_edge_betweenness(G: nx.Graph) -> None:
    """
    Edge betweenness on the static original graph (paper Sec. III A).

    Computed without earthquake/traffic effects.
    """
    # Use travel_time_nominal as weight for shortest paths
    bc = nx.edge_betweenness_centrality(G, weight="travel_time_nominal", normalized=True)
    for (u, v), val in bc.items():
        G.edges[u, v]["betweenness"] = float(val)


def reset_weights_to_nominal(G: nx.Graph, gaussian_noise: bool = True) -> None:
    """Reset dynamic weights to nominal (optionally re-sampled) travel times."""
    for _, _, data in G.edges(data=True):
        nom = float(data.get("travel_time_nominal", data.get("travel_time", 1.0)))
        if gaussian_noise:
            data["travel_time"] = max(1e-4, float(np.random.normal(nom, 0.05 * nom)))
        else:
            data["travel_time"] = nom
        data["weight"] = data["travel_time"]


def select_exit_nodes(
    G: nx.Graph, n_exits: int = 3, seed: Optional[int] = 42
) -> List:
    """
    Define exit nodes at strategic perimeter locations (paper Sec. II B).

    Prefer nodes near the convex hull / bounding-box exterior.
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    if len(nodes) <= n_exits:
        return nodes

    xs = {n: G.nodes[n]["x"] for n in nodes}
    ys = {n: G.nodes[n]["y"] for n in nodes}
    xmin, xmax = min(xs.values()), max(xs.values())
    ymin, ymax = min(ys.values()), max(ys.values())

    # Score by distance to nearest bbox edge (higher = more exterior)
    scored = []
    for n in nodes:
        d = min(xs[n] - xmin, xmax - xs[n], ys[n] - ymin, ymax - ys[n])
        scored.append((d, n))
    scored.sort()  # smallest distance-to-edge first = most exterior
    candidates = [n for _, n in scored[: max(n_exits * 8, n_exits)]]

    # Spread exits: greedy farthest-point among perimeter candidates
    exits = [rng.choice(candidates)]
    while len(exits) < n_exits:
        best, best_d = None, -1.0
        for n in candidates:
            if n in exits:
                continue
            dmin = min(euclid_lonlat(G, n, e) for e in exits)
            if dmin > best_d:
                best_d, best = dmin, n
        if best is None:
            break
        exits.append(best)
    return exits


def euclid_lonlat(G, a, b) -> float:
    ax, ay = node_xy(G, a)
    bx, by = node_xy(G, b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def random_epicenter(
    G: nx.Graph, seed: Optional[int] = None
) -> Tuple[Coord, Coord]:
    """
    Sample a random earthquake epicenter inside the graph bounding box.

    Returns ((lon, lat), (x_km, y_km)) where km coords use the local projection.
    """
    rng = np.random.default_rng(seed)
    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]
    lon = float(rng.uniform(min(xs), max(xs)))
    lat = float(rng.uniform(min(ys), max(ys)))
    origin = get_graph_origin(G)
    from .utils import project_local_km

    return (lon, lat), project_local_km(lon, lat, origin[0], origin[1])


def random_start_node(
    G: nx.Graph, exits: Sequence, seed: Optional[int] = None
):
    """Random start node that is not an exit."""
    rng = random.Random(seed)
    candidates = [n for n in G.nodes() if n not in exits]
    return rng.choice(candidates)


def load_or_build_graph(force: bool = False):
    """Public entry: ensure features exist even when loading from cache."""
    G = download_manila_graph(force=force)
    # Ensure attributes present (older caches)
    sample_edge = next(iter(G.edges(data=True)))[2]
    if "travel_time_nominal" not in sample_edge:
        annotate_edge_features(G)
    if "betweenness" not in sample_edge or float(sample_edge.get("betweenness", 0)) == 0:
        # Recompute if missing/zero-filled
        if all(float(d.get("betweenness", 0)) == 0 for *_, d in G.edges(data=True)):
            compute_edge_betweenness(G)
    return G


if __name__ == "__main__":
    G = load_or_build_graph()
    exits = select_exit_nodes(G)
    epi_ll, epi_km = random_epicenter(G, seed=0)
    print(f"Nodes={G.number_of_nodes()} Edges={G.number_of_edges()}")
    print(f"Exits={exits}")
    print(f"Epicenter lon/lat={epi_ll} km={epi_km}")
    print(f"Max degree={max(dict(G.degree()).values())}")
