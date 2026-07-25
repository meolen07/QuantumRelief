"""
Phase 2 — Dynamic Environment Simulation.

Implements paper Algorithm 1 (Subsequent weight update) and the
earthquake / traffic radius & weight-penalty formulas from Sec. II C.
Geography adapted from Furubira → Manila (Intramuros).

Product framing: these rings are an extreme dynamic-hazard regime. Everyday
traffic / closures / congestion enter through ``src.traffic_provider``
(``MockTrafficProvider`` in demo, ``LiveTrafficProvider`` stub for production).
Escape applies ``EdgeDisruptionSet`` soft penalties from that provider.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx

from .utils import Coord, edge_midpoint, get_graph_origin, project_local_km

# Soft congestion / soft-block / flood multipliers (Algorithm-1 spirit: prefer
# high weight over hard removal so routing still reaches an exit).
DISRUPTION_SOFT_MULT = 5.0
DISRUPTION_SOFT_BLOCK_MULT = 8.0
DISRUPTION_FLOOD_MULT = 12.0  # Ondoy-like soft flood stand-in (stronger than closed)


def _edge_key(u: Any, v: Any) -> Tuple[Any, Any]:
    return tuple(sorted((u, v)))


@dataclass
class EdgeDisruptionSet:
    """
    Localized road disruption (traffic jam / soft closure stand-in).

    Edges keep connectivity; travel weights are multiplied so Hybrid /
    Classical / Dijkstra all see the same soft cost — same spirit as
    Algorithm 1 soft penalties. Live TomTom / HERE feeds are roadmap.
    """

    edges: List[Tuple[Any, Any]] = field(default_factory=list)
    multiplier: float = DISRUPTION_SOFT_MULT
    seed: Optional[int] = None
    kind: str = "congestion"  # congestion | soft_block | flood

    def normalized_edges(self) -> List[Tuple[Any, Any]]:
        seen = set()
        out: List[Tuple[Any, Any]] = []
        for u, v in self.edges:
            key = _edge_key(u, v)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "edges": [[u, v] for u, v in self.normalized_edges()],
            "multiplier": float(self.multiplier),
            "seed": self.seed,
            "kind": self.kind,
        }

    @classmethod
    def from_serializable(cls, raw: Optional[Dict[str, Any]]) -> Optional["EdgeDisruptionSet"]:
        if not raw:
            return None
        edges_raw = raw.get("edges") or []
        edges = []
        for pair in edges_raw:
            if pair is None or len(pair) < 2:
                continue
            edges.append((pair[0], pair[1]))
        if not edges:
            return None
        return cls(
            edges=edges,
            multiplier=float(raw.get("multiplier", DISRUPTION_SOFT_MULT)),
            seed=raw.get("seed"),
            kind=str(raw.get("kind", "congestion")),
        )


def sample_random_disruptions(
    G: nx.Graph,
    *,
    n_seed_edges: int = 1,
    corridor_extra: int = 3,
    multiplier: float = DISRUPTION_SOFT_MULT,
    soft_block: bool = False,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> EdgeDisruptionSet:
    """
    Pick 1–few seed edges and grow a small corridor of adjacent edges.

    Soft high-weight penalties (not hard deletes) so paths remain feasible.
    """
    if rng is None:
        rng = random.Random(seed)

    undirected = [(u, v) for u, v in G.edges()]
    if not undirected:
        return EdgeDisruptionSet(edges=[], multiplier=multiplier, seed=seed)

    n_seed = max(1, min(int(n_seed_edges), len(undirected)))
    seeds = rng.sample(undirected, n_seed)
    selected: Dict[Tuple[Any, Any], Tuple[Any, Any]] = {}
    for u, v in seeds:
        selected[_edge_key(u, v)] = (u, v)

    # Grow a local corridor: repeatedly add a random unused neighbor edge
    # of an already selected edge (BFS-ish expansion).
    extra = max(0, int(corridor_extra))
    frontier = list(seeds)
    attempts = 0
    while extra > 0 and frontier and attempts < len(undirected) * 4:
        attempts += 1
        bu, bv = frontier[rng.randrange(len(frontier))]
        candidates = []
        for node in (bu, bv):
            for nb in G.neighbors(node):
                key = _edge_key(node, nb)
                if key not in selected:
                    candidates.append((node, nb))
        if not candidates:
            # Drop this frontier edge and try another
            frontier = [e for e in frontier if e != (bu, bv)]
            if not frontier:
                frontier = list(selected.values())
            continue
        nu, nv = candidates[rng.randrange(len(candidates))]
        selected[_edge_key(nu, nv)] = (nu, nv)
        frontier.append((nu, nv))
        extra -= 1

    mult = float(DISRUPTION_SOFT_BLOCK_MULT if soft_block else multiplier)
    kind = "soft_block" if soft_block else "congestion"
    return EdgeDisruptionSet(
        edges=list(selected.values()),
        multiplier=mult,
        seed=seed,
        kind=kind,
    )


def sample_flood_corridor(
    G: nx.Graph,
    *,
    near_node: Optional[Any] = None,
    corridor_extra: int = 11,
    multiplier: float = DISRUPTION_FLOOD_MULT,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> EdgeDisruptionSet:
    """
    Ondoy-like soft flood stand-in: a larger coherent corridor with strong
    soft penalties (×12 default).

    No elevation in the cached OSM graph — prefer lower-latitude edges as a
    low-lying / waterfront proxy and grow a connected corridor (not scattered
    jams). Distinct from mild Congestion (×5) and Closed corridor (×8).
    """
    if rng is None:
        rng = random.Random(seed)

    undirected = [(u, v) for u, v in G.edges()]
    if not undirected:
        return EdgeDisruptionSet(
            edges=[], multiplier=multiplier, seed=seed, kind="flood"
        )

    def _edge_lat(u: Any, v: Any) -> float:
        return 0.5 * (float(G.nodes[u]["y"]) + float(G.nodes[v]["y"]))

    # Seed near the trip start when possible; else bias to lower-lat edges.
    seed_edge: Optional[Tuple[Any, Any]] = None
    if near_node is not None and near_node in G:
        incident = [(near_node, nb) for nb in G.neighbors(near_node)]
        if incident:
            # Prefer the lowest-lat incident edge (flood stand-in).
            seed_edge = min(incident, key=lambda e: _edge_lat(e[0], e[1]))
    if seed_edge is None:
        # Rank lower-lat edges; sample among the bottom quartile.
        ranked = sorted(undirected, key=lambda e: _edge_lat(e[0], e[1]))
        pool = ranked[: max(1, len(ranked) // 4)]
        seed_edge = pool[rng.randrange(len(pool))]

    selected: Dict[Tuple[Any, Any], Tuple[Any, Any]] = {
        _edge_key(seed_edge[0], seed_edge[1]): seed_edge
    }
    frontier = [seed_edge]
    extra = max(0, int(corridor_extra))
    attempts = 0
    while extra > 0 and frontier and attempts < len(undirected) * 4:
        attempts += 1
        bu, bv = frontier[rng.randrange(len(frontier))]
        candidates: List[Tuple[Any, Any]] = []
        for node in (bu, bv):
            for nb in G.neighbors(node):
                key = _edge_key(node, nb)
                if key not in selected:
                    candidates.append((node, nb))
        if not candidates:
            frontier = [e for e in frontier if e != (bu, bv)]
            if not frontier:
                frontier = list(selected.values())
            continue
        # Prefer expanding toward lower latitude (coherent "flooded" band).
        candidates.sort(key=lambda e: _edge_lat(e[0], e[1]))
        # Soft bias: often take among the lower half of candidates.
        cut = max(1, len(candidates) // 2)
        nu, nv = candidates[rng.randrange(cut)]
        selected[_edge_key(nu, nv)] = (nu, nv)
        frontier.append((nu, nv))
        extra -= 1

    return EdgeDisruptionSet(
        edges=list(selected.values()),
        multiplier=float(multiplier),
        seed=seed,
        kind="flood",
    )


def apply_edge_disruptions(
    G: nx.Graph,
    disruptions: Optional[
        Union[EdgeDisruptionSet, Sequence[Tuple[Any, Any]], Dict[str, Any]]
    ] = None,
    *,
    multiplier: Optional[float] = None,
) -> List[Tuple[Any, Any]]:
    """
    Apply soft weight penalties in-place. Returns the list of affected edges.

    Prefer soft high multipliers over removing edges so Dijkstra / ML policies
    can still route around congestion.
    """
    if disruptions is None:
        return []

    if isinstance(disruptions, dict):
        dset = EdgeDisruptionSet.from_serializable(disruptions)
        if dset is None:
            return []
    elif isinstance(disruptions, EdgeDisruptionSet):
        dset = disruptions
    else:
        dset = EdgeDisruptionSet(
            edges=list(disruptions),
            multiplier=float(multiplier or DISRUPTION_SOFT_MULT),
        )

    mult = float(multiplier if multiplier is not None else dset.multiplier)
    mult = max(mult, 1.0)
    applied: List[Tuple[Any, Any]] = []
    for u, v in dset.normalized_edges():
        if not G.has_edge(u, v):
            # Try original orientation if undirected key differs from MultiGraph
            if G.has_edge(v, u):
                u, v = v, u
            else:
                continue
        data = G.edges[u, v]
        w0 = float(data.get("weight", data.get("travel_time", 1.0)))
        data["weight"] = w0 * mult
        data["travel_time"] = data["weight"]
        data["disrupted"] = True
        data["disruption_mult"] = mult
        applied.append((u, v))
    return applied


def disruption_edge_latlons(
    G: nx.Graph,
    disruptions: Optional[
        Union[EdgeDisruptionSet, Sequence[Tuple[Any, Any]], Dict[str, Any]]
    ],
) -> List[List[List[float]]]:
    """Folium PolyLine coords ``[[lat, lon], [lat, lon]]`` per disrupted edge."""
    if disruptions is None:
        return []
    if isinstance(disruptions, dict):
        dset = EdgeDisruptionSet.from_serializable(disruptions)
    elif isinstance(disruptions, EdgeDisruptionSet):
        dset = disruptions
    else:
        dset = EdgeDisruptionSet(edges=list(disruptions))
    if dset is None:
        return []
    coords: List[List[List[float]]] = []
    for u, v in dset.normalized_edges():
        if not G.has_edge(u, v) and not G.has_edge(v, u):
            continue
        if u not in G.nodes or v not in G.nodes:
            continue
        coords.append(
            [
                [float(G.nodes[u]["y"]), float(G.nodes[u]["x"])],
                [float(G.nodes[v]["y"]), float(G.nodes[v]["x"])],
            ]
        )
    return coords


def damage_radius(t: float, intensity: float = 1.0) -> float:
    """
    Earthquake damage radius (paper Sec. II C):

      r_epi = 0.5 + √(0.0002 × t)

    ``intensity`` (≥1) scales the radius for hard training labels without
    changing the paper formula shape. Default 1.0 preserves inference / viz.
    """
    return float(intensity) * (0.5 + math.sqrt(0.0002 * t))


def exit_radius(t: float, intensity: float = 1.0) -> float:
    """
    Traffic congestion radius (paper Sec. II C):

      r_exit = √(0.00075 × t)

    ``intensity`` scales the radius for hard-mode dataset generation.
    """
    return float(intensity) * math.sqrt(0.00075 * max(t, 0.0))


def _apply_initial_earthquake(w: float, d_epi: float, r_epi: float) -> float:
    """
    Initial static earthquake effect (t = 0), paper Sec. II C:

      w ← 5w   if d_epi ≤ 0.3 r_epi
      w ← 2w   if 0.3 r_epi < d_epi ≤ 0.75 r_epi
      w ← 1.3w if 0.75 r_epi < d_epi ≤ r_epi
      w        otherwise
    """
    if d_epi <= 0.3 * r_epi:
        return w * 5.0
    if d_epi <= 0.75 * r_epi:
        return w * 2.0
    if d_epi <= r_epi:
        return w * 1.3
    return w


def _apply_ongoing_earthquake(w: float, d_epi: float, r_epi: float, t: float) -> float:
    """
    Ongoing earthquake effect, paper Sec. II C:

      min{w × √(0.003 t)+1, 5}  if d_epi ≤ 0.3 r_epi
      min{w × √(0.002 t)+1, 4}  if 0.3 r_epi < d_epi ≤ 0.75 r_epi
      min{w × √(0.001 t)+1, 3}  if 0.75 r_epi < d_epi ≤ r_epi
    """
    if t <= 0:
        return w
    if d_epi <= 0.3 * r_epi:
        return min(w * math.sqrt(0.003 * t) + 1.0, 5.0)
    if d_epi <= 0.75 * r_epi:
        return min(w * math.sqrt(0.002 * t) + 1.0, 4.0)
    if d_epi <= r_epi:
        return min(w * math.sqrt(0.001 * t) + 1.0, 3.0)
    return w


def _apply_traffic(w: float, d_exit: float, r_exit: float, t: float) -> float:
    """
    Ongoing traffic congestion near exits, paper Sec. II C:

      min{w × √(0.03 t)+1, 5}  if d_exit ≤ 0.5 r_exit
      min{w × √(0.02 t)+1, 4}  if 0.5 r_exit < d_exit ≤ 0.75 r_exit
      min{w × √(0.01 t)+1, 3}  if 0.75 r_exit < d_exit ≤ r_exit
    """
    if t <= 0 or r_exit <= 0:
        return w
    if d_exit <= 0.5 * r_exit:
        return min(w * math.sqrt(0.03 * t) + 1.0, 5.0)
    if d_exit <= 0.75 * r_exit:
        return min(w * math.sqrt(0.02 * t) + 1.0, 4.0)
    if d_exit <= r_exit:
        return min(w * math.sqrt(0.01 * t) + 1.0, 3.0)
    return w


@dataclass
class DynamicEnvironment:
    """
    Dynamic road graph following Algorithm 1.

    Coordinates for distance calculations use local km projection so that
    paper radii (r_epi ≈ 0.5 km at t=0) are meaningful on a district map.

    ``hazard_intensity`` (default 1.0) widens radii and accelerates ongoing
    penalty growth for hard training labels. Streamlit / routing inference
    leave it at 1.0 so paper-scale dynamics are unchanged at demo time.
    """

    G: nx.Graph
    epicenter_lonlat: Coord
    exit_nodes: Sequence
    t: int = 0
    hazard_intensity: float = 1.0
    origin: Coord = field(default_factory=lambda: (0.0, 0.0))
    epicenter_km: Coord = field(default_factory=lambda: (0.0, 0.0))
    exit_coords_km: Dict = field(default_factory=dict)
    _baseline_weights: Dict[Tuple, float] = field(default_factory=dict)

    def __post_init__(self):
        self.hazard_intensity = max(float(self.hazard_intensity), 1e-6)
        self.origin = get_graph_origin(self.G)
        lon, lat = self.epicenter_lonlat
        self.epicenter_km = project_local_km(lon, lat, self.origin[0], self.origin[1])
        self.exit_coords_km = {}
        for ex in self.exit_nodes:
            lon_e, lat_e = self.G.nodes[ex]["x"], self.G.nodes[ex]["y"]
            self.exit_coords_km[ex] = project_local_km(
                lon_e, lat_e, self.origin[0], self.origin[1]
            )
        # Snapshot nominal weights as Algorithm 1 baseline
        self._baseline_weights = {}
        for u, v, data in self.G.edges(data=True):
            key = tuple(sorted((u, v)))
            self._baseline_weights[key] = float(
                data.get("travel_time", data.get("weight", 1.0))
            )
            data["weight"] = self._baseline_weights[key]

    def clone(self) -> "DynamicEnvironment":
        env = DynamicEnvironment(
            G=self.G.copy(),
            epicenter_lonlat=self.epicenter_lonlat,
            exit_nodes=list(self.exit_nodes),
            t=self.t,
            hazard_intensity=self.hazard_intensity,
        )
        return env

    @property
    def _intensity(self) -> float:
        return float(self.hazard_intensity)

    def _edge_center_km(self, u, v) -> Coord:
        mid = edge_midpoint(self.G, u, v)
        return project_local_km(mid[0], mid[1], self.origin[0], self.origin[1])

    def _d_epi(self, u, v) -> float:
        cx, cy = self._edge_center_km(u, v)
        return math.hypot(cx - self.epicenter_km[0], cy - self.epicenter_km[1])

    def _d_exit_min(self, u, v) -> float:
        cx, cy = self._edge_center_km(u, v)
        return min(
            math.hypot(cx - ex[0], cy - ex[1]) for ex in self.exit_coords_km.values()
        )

    def apply_initial_earthquake(self) -> None:
        """Algorithm 1 step 3 — initial earthquake effect at t=0."""
        r_epi = damage_radius(0, self._intensity)
        for u, v, data in self.G.edges(data=True):
            key = tuple(sorted((u, v)))
            w0 = self._baseline_weights[key]
            w = _apply_initial_earthquake(w0, self._d_epi(u, v), r_epi)
            data["weight"] = w
            data["travel_time"] = w
            # New baseline after initial shock (paper: used as baseline for next steps)
            self._baseline_weights[key] = w

    def update_ongoing_effects(self) -> None:
        """
        Algorithm 1 steps 5–6: ongoing earthquake then traffic, at current t.

        Paper: 'at each step, all w's are updated and used as the baseline
        values for the next step.'

        Hard mode: radii grow with ``hazard_intensity``; ongoing √(c·t) terms
        use t_eff = t × intensity so severe congestion appears in Dijkstra labels.
        """
        intensity = self._intensity
        r_epi = damage_radius(self.t, intensity)
        r_ex = exit_radius(self.t, intensity)
        t_eff = float(self.t) * intensity
        for u, v, data in self.G.edges(data=True):
            key = tuple(sorted((u, v)))
            w = self._baseline_weights[key]
            w = _apply_ongoing_earthquake(w, self._d_epi(u, v), r_epi, t_eff)
            w = _apply_traffic(w, self._d_exit_min(u, v), r_ex, t_eff)
            data["weight"] = w
            data["travel_time"] = w
            self._baseline_weights[key] = w

    def step(self, next_node=None) -> None:
        """
        One Algorithm 1 loop iteration after the initial earthquake:

          Update ongoing earthquake → Update traffic → Travel → t += 1
        """
        self.update_ongoing_effects()
        self.t += 1

    def initialize(self) -> None:
        """Algorithm 1 lines 1–3."""
        self.t = 0
        self.apply_initial_earthquake()

    def current_radii(self) -> Dict[str, float]:
        return {
            "r_epi": damage_radius(self.t, self._intensity),
            "r_exit": exit_radius(self.t, self._intensity),
            "t": float(self.t),
            "hazard_intensity": self._intensity,
        }


def run_simulation_loop(
    env: DynamicEnvironment,
    start,
    choose_next,
    max_steps: int = 200,
) -> List:
    """
    Full Algorithm 1 simulation.

    choose_next(env, current) -> next_node
    """
    env.initialize()
    path = [start]
    current = start
    exits = set(env.exit_nodes)
    for _ in range(max_steps):
        if current in exits:
            break
        # steps 5–6 before travel
        env.update_ongoing_effects()
        nxt = choose_next(env, current)
        if nxt is None or nxt == current:
            break
        path.append(nxt)
        current = nxt
        env.t += 1  # step 8
    return path
