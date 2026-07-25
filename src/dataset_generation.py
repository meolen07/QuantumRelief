"""
Phase 3a — Dataset generation via node-wise Dijkstra on the dynamic graph.

Builds Table I input vectors (size 36) labelled by the next adjacent node
chosen by Dijkstra at each decision point.

Hard mode (--hard): stronger hazard_intensity so Dijkstra labels reflect severe
congestion, plus oversampling of Classical-failure start/epi/exit seeds from
data/hard_seeds.json or data/demo_scenarios.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    DATA_DIR,
    DATASET_PATH,
    INPUT_DIM,
    MAX_DEGREE,
    Coord,
    cosine_similarity_to_exit,
    ensure_dirs,
    euclidean,
    node_xy_km,
    project_local_km,
)


# Hard-mode defaults (training labels only; inference stays intensity=1.0)
HARD_HAZARD_INTENSITY = 2.0
HARD_OVERSAMPLE = 8
HARD_SEEDS_PATH = DATA_DIR / "hard_seeds.json"
DEMO_SCENARIOS_PATH = DATA_DIR / "demo_scenarios.json"


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
    hazard_intensity: float = 1.0,
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
        hazard_intensity=hazard_intensity,
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


def _coerce_node_id(G: nx.Graph, raw) -> Any:
    """Map JSON node ids onto graph node types (int vs str)."""
    if raw in G:
        return raw
    try:
        as_int = int(raw)
        if as_int in G:
            return as_int
    except (TypeError, ValueError):
        pass
    as_str = str(raw)
    if as_str in G:
        return as_str
    raise KeyError(f"Node {raw!r} not in graph")


def load_hard_seeds(
    path: Optional[Path] = None,
    G: Optional[nx.Graph] = None,
) -> List[Dict[str, Any]]:
    """
    Load Classical-failure (start, epi, exit) seeds.

    Prefers data/hard_seeds.json; falls back to data/demo_scenarios.json.
    """
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend([HARD_SEEDS_PATH, DEMO_SCENARIOS_PATH])

    payload = None
    used = None
    for p in candidates:
        if p.exists():
            payload = json.loads(p.read_text(encoding="utf-8"))
            used = p
            break
    if payload is None:
        print("[QuantumRelief] No hard seeds found (hard_seeds.json / demo_scenarios.json).")
        return []

    raw_list = payload.get("seeds") or payload.get("scenarios") or []
    seeds: List[Dict[str, Any]] = []
    for row in raw_list:
        try:
            start = row["start_node"]
            dest = row.get("dest_node", row.get("exit_node"))
            epi_lat = float(row["epi_lat"])
            epi_lon = float(row["epi_lon"])
            if G is not None:
                start = _coerce_node_id(G, start)
                dest = _coerce_node_id(G, dest)
            seeds.append(
                {
                    "start_node": start,
                    "dest_node": dest,
                    "epi_lat": epi_lat,
                    "epi_lon": epi_lon,
                    "id": row.get("id"),
                    "source": str(used),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"  skip seed ({exc}): {row.get('id', row)}")
            continue
    print(f"[QuantumRelief] Loaded {len(seeds)} hard seeds from {used}")
    return seeds


def collect_hard_seed_samples(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    intensities: Sequence[float] = (1.0, 2.0),
    repeats: int = 4,
    seeds_path: Optional[Path] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect Dijkstra-labelled trajectories for Classical-failure hard seeds.

    Normalises with the provided dataset mean/std so samples can be mixed into
    an existing ``routing_dataset.npz`` Hybrid subset. Includes intensity 1.0
    (matches inference) and 2.0 (matches hard training labels) to close the
    next-hop-acc vs full-route travel gap.
    """
    G = load_or_build_graph()
    exits = select_exit_nodes(G, n_exits=3, seed=42)
    seeds = load_hard_seeds(path=seeds_path, G=G)
    if not seeds:
        return (
            np.zeros((0, INPUT_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std_safe = np.maximum(std, 1e-6)
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for _ in range(max(1, int(repeats))):
        for intensity in intensities:
            for spec in seeds:
                try:
                    start = _coerce_node_id(G, spec["start_node"])
                    dest = _coerce_node_id(G, spec["dest_node"])
                except KeyError:
                    continue
                epi_ll = (float(spec["epi_lon"]), float(spec["epi_lat"]))
                samples = collect_episode(
                    G,
                    start,
                    dest,
                    epi_ll,
                    exits,
                    hazard_intensity=float(intensity),
                )
                for x, y in samples:
                    X_list.append(x)
                    y_list.append(int(y))

    if not X_list:
        return (
            np.zeros((0, INPUT_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    X_raw = np.stack(X_list, axis=0).astype(np.float32)
    Xn = (X_raw - mean) / std_safe
    y = np.asarray(y_list, dtype=np.int64)
    # Light shuffle so oversampled copies aren't contiguous in every batch
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    print(
        f"[QuantumRelief] Hard-seed hybrid oversample: {len(y)} samples "
        f"(seeds={len(seeds)}, repeats={repeats}, intensities={list(intensities)})"
    )
    return Xn[order], y[order]


def write_hard_seeds_from_scenarios(
    scenarios_path: Path = DEMO_SCENARIOS_PATH,
    out_path: Path = HARD_SEEDS_PATH,
) -> Path:
    """Extract compact seed list from demo_scenarios.json → hard_seeds.json."""
    ensure_dirs()
    if not scenarios_path.exists():
        raise FileNotFoundError(f"Missing scenarios: {scenarios_path}")
    payload = json.loads(scenarios_path.read_text(encoding="utf-8"))
    seeds = []
    for row in payload.get("scenarios") or []:
        seeds.append(
            {
                "id": row.get("id"),
                "start_node": row["start_node"],
                "dest_node": row["dest_node"],
                "epi_lat": row["epi_lat"],
                "epi_lon": row["epi_lon"],
                "metrics": row.get("metrics"),
            }
        )
    out = {
        "source": str(scenarios_path),
        "n_seeds": len(seeds),
        "note": (
            "Classical-failure start/epi/exit seeds for hard dataset oversampling. "
            "Derived from find_advantage_scenarios / demo_scenarios.json."
        ),
        "seeds": seeds,
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[QuantumRelief] Wrote {len(seeds)} seeds → {out_path}")
    return out_path


def _episode_plan(
    n_episodes: int,
    seeds: Sequence[Dict[str, Any]],
    oversample: int,
    rng: np.random.Generator,
) -> List[Optional[Dict[str, Any]]]:
    """
    Build a shuffled list of episode specs.

    Each entry is either a seed dict (oversampled) or None (= random episode).
    Random slots fill up to ``n_episodes``; seed slots are oversample × |seeds|
    mixed in (extra episodes beyond n_episodes when seeds are present).
    """
    plan: List[Optional[Dict[str, Any]]] = [None] * n_episodes
    if seeds and oversample > 0:
        for seed in seeds:
            for _ in range(oversample):
                plan.append(dict(seed))
    rng.shuffle(plan)
    return plan


def generate_dataset(
    n_episodes: int = 80,
    seed: int = 42,
    save: bool = True,
    hard: bool = False,
    hazard_intensity: float = 1.0,
    oversample: int = 1,
    seed_scenarios: Optional[Sequence[Dict[str, Any]]] = None,
    seeds_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate a supervised routing dataset.

    ``hard=True`` sets hazard_intensity=2.0 (unless overridden) and oversamples
    Classical-failure seeds 8× (unless ``oversample`` already > 1).
    """
    ensure_dirs()
    G = load_or_build_graph()
    exits = select_exit_nodes(G, n_exits=3, seed=seed)
    rng = np.random.default_rng(seed)

    if hard:
        if hazard_intensity <= 1.0:
            hazard_intensity = HARD_HAZARD_INTENSITY
        if oversample <= 1:
            oversample = HARD_OVERSAMPLE
        if seed_scenarios is None:
            seed_scenarios = load_hard_seeds(path=seeds_path, G=G)
            if not seed_scenarios and DEMO_SCENARIOS_PATH.exists():
                write_hard_seeds_from_scenarios()
                seed_scenarios = load_hard_seeds(path=HARD_SEEDS_PATH, G=G)
    elif seed_scenarios is None:
        seed_scenarios = []

    plan = _episode_plan(n_episodes, seed_scenarios, oversample if hard else 0, rng)
    n_seed_eps = sum(1 for p in plan if p is not None)
    n_rand_eps = sum(1 for p in plan if p is None)

    X_list, y_list = [], []
    print(
        f"[QuantumRelief] Generating {len(plan)} routing episodes "
        f"(random={n_rand_eps}, seed_oversample={n_seed_eps}, "
        f"hazard_intensity={hazard_intensity:.2f}, hard={hard})…"
    )
    for spec in tqdm(plan):
        if spec is not None:
            try:
                start = _coerce_node_id(G, spec["start_node"])
                dest = _coerce_node_id(G, spec["dest_node"])
            except KeyError:
                continue
            epi_ll = (float(spec["epi_lon"]), float(spec["epi_lat"]))
        else:
            epi_ll, _ = random_epicenter(G, seed=int(rng.integers(0, 1_000_000)))
            dest = exits[int(rng.integers(0, len(exits)))]
            start = random_start_node(
                G, exits=[dest], seed=int(rng.integers(0, 1_000_000))
            )

        # Skip trivial / unreachable
        try:
            nx.shortest_path(G, start, dest, weight="travel_time_nominal")
        except nx.NetworkXNoPath:
            continue
        samples = collect_episode(
            G,
            start,
            dest,
            epi_ll,
            exits,
            hazard_intensity=hazard_intensity,
        )
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

    out = {
        "X": Xn,
        "y": y,
        "mean": mean,
        "std": std,
        "X_raw": X,
        "meta_n_episodes": np.asarray([len(plan)], dtype=np.int64),
        "meta_hazard_intensity": np.asarray([hazard_intensity], dtype=np.float32),
        "meta_hard": np.asarray([1 if hard else 0], dtype=np.int8),
        "meta_n_seed_episodes": np.asarray([n_seed_eps], dtype=np.int64),
    }
    if save:
        np.savez_compressed(DATASET_PATH, **out)
        print(f"[QuantumRelief] Saved dataset ({len(y)} samples) → {DATASET_PATH}")
    return out


def load_dataset() -> Dict[str, np.ndarray]:
    if not DATASET_PATH.exists():
        return generate_dataset()
    data = np.load(DATASET_PATH)
    return {k: data[k] for k in data.files}


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate QuantumRelief Dijkstra-labelled routing dataset."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Random episode count (default 1000; recommended 800–1500).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hard",
        action="store_true",
        help=(
            "Hard mode: hazard_intensity=2.0 + oversample Classical-failure "
            "seeds from hard_seeds.json / demo_scenarios.json."
        ),
    )
    parser.add_argument(
        "--hazard-intensity",
        type=float,
        default=None,
        help="Override hazard intensity (default 1.0, or 2.0 with --hard).",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=None,
        help="Times to repeat each hard seed (default 8 with --hard).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Path to hard_seeds.json or demo_scenarios.json.",
    )
    parser.add_argument(
        "--write-seeds-only",
        action="store_true",
        help="Only sync hard_seeds.json from demo_scenarios.json, then exit.",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if args.write_seeds_only:
        write_hard_seeds_from_scenarios()
        return

    intensity = 1.0 if args.hazard_intensity is None else args.hazard_intensity
    oversample = 1 if args.oversample is None else args.oversample
    if args.hard:
        if args.hazard_intensity is None:
            intensity = HARD_HAZARD_INTENSITY
        if args.oversample is None:
            oversample = HARD_OVERSAMPLE

    ds = generate_dataset(
        n_episodes=args.episodes,
        seed=args.seed,
        save=not args.no_save,
        hard=args.hard,
        hazard_intensity=intensity,
        oversample=oversample,
        seeds_path=Path(args.seeds) if args.seeds else None,
    )
    print("X", ds["X"].shape, "y", ds["y"].shape)


if __name__ == "__main__":
    _cli()
