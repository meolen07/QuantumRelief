"""
Phase 3a — Dataset generation via node-wise Dijkstra on the dynamic graph.

Builds Table I input vectors (size 36) labelled by the next adjacent node
chosen by Dijkstra at each decision point.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

from .dynamic_simulation import DynamicEnvironment
from .graph_setup import (
    load_or_build_graph,
    random_epicenter,
    random_start_node,
    reset_weights_to_nominal,
    select_exit_nodes,
)
from .utils import (
    DATASET_PATH,
    INPUT_DIM,
    MAX_DEGREE,
    Coord,
    cosine_similarity_to_exit,
    ensure_dirs,
    euclidean,
    get_graph_origin,
    node_xy_km,
    project_local_km,
)


def ordered_neighbors(G: nx.Graph, node) -> List:
    """Deterministic neighbor ordering (sorted by node id)."""
    return sorted(G.neighbors(node), key=lambda n: str(n))


def build_input_vector(
    G: nx.Graph,
    current,
    start,
    dest,
    epicenter_km: Coord,
    origin: Coord,
) -> Tuple[np.ndarray, List]:
    """
    Construct the 36-dim Table I input vector:

      [x_epi, y_epi, x_start, y_start, x_dest, y_dest,
       x_e1, y_e1, w1, e1, d1, c1,  … (×5, zero-padded)]
    """
    neighbors = ordered_neighbors(G, current)
    cur_km = node_xy_km(G, current, origin)
    start_km = node_xy_km(G, start, origin)
    dest_km = node_xy_km(G, dest, origin)

    vec: List[float] = [
        epicenter_km[0],
        epicenter_km[1],
        start_km[0],
        start_km[1],
        dest_km[0],
        dest_km[1],
    ]

    for i in range(MAX_DEGREE):
        if i < len(neighbors):
            nb = neighbors[i]
            nb_km = node_xy_km(G, nb, origin)
            data = G.edges[current, nb]
            w = float(data.get("weight", data.get("travel_time", 1.0)))
            e = float(data.get("betweenness", 0.0))
            # Heuristic 1: getting closer? — Euclidean from neighbor to dest
            d = euclidean(nb_km, dest_km)
            # Heuristic 2: heading toward dest? — cosine similarity
            c = cosine_similarity_to_exit(cur_km, nb_km, dest_km)
            vec.extend([nb_km[0], nb_km[1], w, e, d, c])
        else:
            vec.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert len(vec) == INPUT_DIM, f"Expected {INPUT_DIM}, got {len(vec)}"
    return np.asarray(vec, dtype=np.float32), neighbors


def dijkstra_next_node(G: nx.Graph, current, dest):
    """
    Node-wise Dijkstra: shortest path on *current* weights; return next hop.
    """
    if current == dest:
        return None
    try:
        path = nx.shortest_path(G, current, dest, weight="weight")
    except nx.NetworkXNoPath:
        return None
    if len(path) < 2:
        return None
    return path[1]


def collect_episode(
    G: nx.Graph,
    start,
    dest,
    epicenter_lonlat: Coord,
    exit_nodes: Sequence,
    max_steps: int = 150,
) -> List[Tuple[np.ndarray, int]]:
    """
    Run Algorithm 1 with node-wise Dijkstra as the oracle chooser.
    Record (x_36, label_index) at each decision.
    """
    reset_weights_to_nominal(G, gaussian_noise=True)
    env = DynamicEnvironment(
        G=G.copy(),
        epicenter_lonlat=epicenter_lonlat,
        exit_nodes=exit_nodes,
    )
    # Align exit list used for traffic with the chosen destination
    # (paper samples one of three exits per instance)
    env.exit_nodes = [dest]
    lon_e = G.nodes[dest]["x"]
    lat_e = G.nodes[dest]["y"]
    env.exit_coords_km = {
        dest: project_local_km(lon_e, lat_e, env.origin[0], env.origin[1])
    }

    env.initialize()
    origin = env.origin
    samples: List[Tuple[np.ndarray, int]] = []
    current = start

    for _ in range(max_steps):
        if current == dest:
            break
        env.update_ongoing_effects()
        x, neighbors = build_input_vector(
            env.G, current, start, dest, env.epicenter_km, origin
        )
        nxt = dijkstra_next_node(env.G, current, dest)
        if nxt is None or nxt not in neighbors:
            break
        label = neighbors.index(nxt)
        samples.append((x, label))
        current = nxt
        env.t += 1
    return samples


def generate_dataset(
    n_episodes: int = 80,
    seed: int = 42,
    save: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate a modest supervised dataset for the demo prototype."""
    ensure_dirs()
    G = load_or_build_graph()
    exits = select_exit_nodes(G, n_exits=3, seed=seed)
    rng = np.random.default_rng(seed)

    X_list, y_list = [], []
    print(f"[QuantumRelief] Generating {n_episodes} routing episodes…")
    for i in tqdm(range(n_episodes)):
        epi_ll, _ = random_epicenter(G, seed=int(rng.integers(0, 1_000_000)))
        dest = exits[int(rng.integers(0, len(exits)))]
        start = random_start_node(G, exits=[dest], seed=int(rng.integers(0, 1_000_000)))
        # Skip trivial / unreachable
        try:
            nx.shortest_path(G, start, dest, weight="travel_time_nominal")
        except nx.NetworkXNoPath:
            continue
        samples = collect_episode(G, start, dest, epi_ll, exits)
        for x, y in samples:
            X_list.append(x)
            y_list.append(y)

    if not X_list:
        raise RuntimeError("Dataset generation produced no samples.")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    # Feature normalisation (z-score) — store mean/std for inference
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std

    out = {"X": Xn, "y": y, "mean": mean, "std": std, "X_raw": X}
    if save:
        np.savez_compressed(DATASET_PATH, **out)
        print(f"[QuantumRelief] Saved dataset ({len(y)} samples) → {DATASET_PATH}")
    return out


def load_dataset() -> Dict[str, np.ndarray]:
    if not DATASET_PATH.exists():
        return generate_dataset()
    data = np.load(DATASET_PATH)
    return {k: data[k] for k in data.files}


if __name__ == "__main__":
    ds = generate_dataset(n_episodes=40)
    print("X", ds["X"].shape, "y", ds["y"].shape)
