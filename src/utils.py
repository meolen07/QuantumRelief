"""Shared helpers for QuantumRelief (Manila adaptation of arXiv:2307.15682)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

# Project roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
GRAPH_CACHE_PATH = DATA_DIR / "manila_intramuros_graph.graphml"
GRAPH_META_PATH = DATA_DIR / "manila_intramuros_meta.json"
DATASET_PATH = DATA_DIR / "routing_dataset.npz"
MODEL_CHECKPOINT = MODELS_DIR / "film_classical.pt"
HYBRID_CHECKPOINT = MODELS_DIR / "film_hybrid.pt"

# Table I / paper constants
INPUT_DIM = 36  # 6 global + 5 * 6 edge features
MAX_DEGREE = 5
FILM_DIM = 2  # epicenter (x, y)
MAIN_DIM = 34  # remaining features
N_OUTPUTS = 5

# Manila / Intramuros–Ermita bounding box (compact district ~paper-scale graph).
# Tuned so OSMnx drive network lands near the paper's ~357 nodes after cleanup.
MANILA_BBOX = {
    "north": 14.5980,
    "south": 14.5780,
    "east": 120.9860,
    "west": 120.9685,
}

Coord = Tuple[float, float]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def node_xy(G, node) -> Coord:
    """Return (x, y) = (lon, lat) for an OSMnx/NetworkX node."""
    data = G.nodes[node]
    return float(data["x"]), float(data["y"])


def euclidean(p: Coord, q: Coord) -> float:
    """Euclidean distance between two coordinate pairs (paper Sec. III A)."""
    return math.hypot(q[0] - p[0], q[1] - p[1])


def cosine_similarity_to_exit(
    current: Coord, neighbor: Coord, exit_coord: Coord
) -> float:
    """
    Cosine similarity cos(θ) = A·B / (|A||B|) as in the paper.

    A = neighbor - current, B = exit - current.
    Paper Table I calls this 'Cosine distance' but gives the similarity formula.
    """
    ax, ay = neighbor[0] - current[0], neighbor[1] - current[1]
    bx, by = exit_coord[0] - current[0], exit_coord[1] - current[1]
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return (ax * bx + ay * by) / (na * nb)


def edge_midpoint(G, u, v) -> Coord:
    ux, uy = node_xy(G, u)
    vx, vy = node_xy(G, v)
    return (0.5 * (ux + vx), 0.5 * (uy + vy))


def project_local_km(lon: float, lat: float, lon0: float, lat0: float) -> Coord:
    """
    Approximate local km projection around (lon0, lat0).

    Paper radii (repi ≈ 0.5 at t=0) are consistent with ~km-scale local
    coordinates on a compact district map.
    """
    # metres per degree
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    x_km = (lon - lon0) * m_per_deg_lon / 1000.0
    y_km = (lat - lat0) * m_per_deg_lat / 1000.0
    return x_km, y_km


def get_graph_origin(G) -> Coord:
    """Centroid of node lon/lat used as local projection origin."""
    xs, ys = [], []
    for _, data in G.nodes(data=True):
        xs.append(data["x"])
        ys.append(data["y"])
    return float(np.mean(xs)), float(np.mean(ys))


def node_xy_km(G, node, origin: Coord) -> Coord:
    lon, lat = node_xy(G, node)
    return project_local_km(lon, lat, origin[0], origin[1])


def pad_to_max_degree(items: Sequence, fill=None, max_degree: int = MAX_DEGREE):
    """Zero-pad / truncate neighbor feature slots to max_degree (paper Table I)."""
    out = list(items)[:max_degree]
    while len(out) < max_degree:
        out.append(fill)
    return out


def softmax(logits: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(logits), dtype=np.float64)
    arr = arr - arr.max()
    e = np.exp(arr)
    return e / e.sum()
