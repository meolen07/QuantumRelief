"""
Phase 2 — Dynamic Environment Simulation.

Implements paper Algorithm 1 (Subsequent weight update) and the
earthquake / traffic radius & weight-penalty formulas from Sec. II C.
Geography adapted from Furubira → Manila (Intramuros).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

from .utils import Coord, edge_midpoint, get_graph_origin, project_local_km


def damage_radius(t: float) -> float:
    """Earthquake damage radius: r_epi = 0.5 + √(0.0002 × t)."""
    return 0.5 + math.sqrt(0.0002 * t)


def exit_radius(t: float) -> float:
    """Traffic congestion radius: r_exit = √(0.00075 × t)."""
    return math.sqrt(0.00075 * max(t, 0.0))


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
    """

    G: nx.Graph
    epicenter_lonlat: Coord
    exit_nodes: Sequence
    t: int = 0
    origin: Coord = field(default_factory=lambda: (0.0, 0.0))
    epicenter_km: Coord = field(default_factory=lambda: (0.0, 0.0))
    exit_coords_km: Dict = field(default_factory=dict)
    _baseline_weights: Dict[Tuple, float] = field(default_factory=dict)

    def __post_init__(self):
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
        )
        return env

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
        r_epi = damage_radius(0)
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
        """
        r_epi = damage_radius(self.t)
        r_ex = exit_radius(self.t)
        for u, v, data in self.G.edges(data=True):
            key = tuple(sorted((u, v)))
            w = self._baseline_weights[key]
            w = _apply_ongoing_earthquake(w, self._d_epi(u, v), r_epi, self.t)
            w = _apply_traffic(w, self._d_exit_min(u, v), r_ex, self.t)
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
            "r_epi": damage_radius(self.t),
            "r_exit": exit_radius(self.t),
            "t": float(self.t),
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
