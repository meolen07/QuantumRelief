"""
Mock city traffic / incident feed (product service).

Demo mode ships a **named conditions snapshot** — same shape production would
get from TomTom / HERE — so Live Escape can open on “current road conditions”
without paid API calls.

Scenarios rotate on a deterministic time bucket (feels live; reproducible).
The Quantathon judge flood pin is one catalog entry (``judge_flood``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx

from src.dynamic_simulation import (
    DISRUPTION_FLOOD_MULT,
    DISRUPTION_SOFT_BLOCK_MULT,
    DISRUPTION_SOFT_MULT,
    EdgeDisruptionSet,
    sample_flood_corridor,
    sample_random_disruptions,
)
from src.utils import DATA_DIR

DEMO_SCENARIOS_PATH = DATA_DIR / "demo_scenarios.json"

# Rotate “live” bucket every N minutes (deterministic seed from wall clock).
FEED_BUCKET_MINUTES = 5

# Human-facing catalog — ids are stable; seeds / params drive edge sampling.
SCENARIO_CATALOG: Tuple[Dict[str, Any], ...] = (
    {
        "id": "rush_hour_arterial",
        "name": "Rush-hour congestion",
        "blurb": "Peak traffic on Intramuros arterials.",
        "incidents": (
            {
                "kind": "congestion",
                "label": "Congestion on arterial",
                "severity": 0.55,
                "area_hint": "Intramuros core",
                "corridor_extra": 4,
                "n_seed_edges": 1,
                "seed_offset": 101,
            },
        ),
    },
    {
        "id": "flood_pasig",
        "name": "Flood watch · Pasig corridor",
        "blurb": "Soft flood stand-in on a low-lying corridor (Ondoy-like ×12).",
        "incidents": (
            {
                "kind": "flood",
                "label": "Flooded corridor near Pasig",
                "severity": 0.9,
                "area_hint": "Pasig-side low ground",
                "corridor_extra": 11,
                "seed_offset": 17025,
            },
        ),
    },
    {
        "id": "closure_historic",
        "name": "Temporary corridor closure",
        "blurb": "Soft closed corridor (still passable at high weight).",
        "incidents": (
            {
                "kind": "soft_block",
                "label": "Closed corridor · historic core",
                "severity": 0.75,
                "area_hint": "Intramuros walls",
                "corridor_extra": 5,
                "n_seed_edges": 1,
                "seed_offset": 220,
            },
        ),
    },
    {
        "id": "mixed_evening",
        "name": "Evening mix · jam + closure",
        "blurb": "Congestion plus a soft closure elsewhere on the grid.",
        "incidents": (
            {
                "kind": "congestion",
                "label": "Congestion on arterial",
                "severity": 0.5,
                "area_hint": "North gate approach",
                "corridor_extra": 3,
                "n_seed_edges": 1,
                "seed_offset": 330,
            },
            {
                "kind": "soft_block",
                "label": "Soft closure · side street",
                "severity": 0.65,
                "area_hint": "East wall lanes",
                "corridor_extra": 3,
                "n_seed_edges": 1,
                "seed_offset": 331,
            },
        ),
    },
    {
        "id": "quiet_morning",
        "name": "Quiet morning · light jam",
        "blurb": "Mild congestion only — baseline city day.",
        "incidents": (
            {
                "kind": "congestion",
                "label": "Light congestion",
                "severity": 0.3,
                "area_hint": "Central plaza rim",
                "corridor_extra": 2,
                "n_seed_edges": 1,
                "seed_offset": 40,
            },
        ),
    },
    {
        "id": "judge_flood",
        "name": "Judge demo · flooded corridor",
        "blurb": "Pinned Quantathon flood seed (Hybrid travel-win corridor).",
        "judge_pin": True,
        "incidents": (
            {
                "kind": "flood",
                "label": "Flooded corridor near Pasig",
                "severity": 0.95,
                "area_hint": "Judge-pinned flood",
                "corridor_extra": 11,
                "seed_offset": 17025,
            },
        ),
    },
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def time_bucket_id(
    when: Optional[datetime] = None,
    *,
    bucket_minutes: int = FEED_BUCKET_MINUTES,
) -> int:
    """Integer bucket for deterministic scenario rotation."""
    when = when or _utc_now()
    epoch = int(when.timestamp())
    return epoch // max(1, int(bucket_minutes) * 60)


def _load_judge_flood_params() -> Dict[str, int]:
    """Pinned flood seed from demo_scenarios.json when present."""
    defaults = {"seed": 17025, "corridor_extra": 11, "near_start": True}
    path = Path(DEMO_SCENARIOS_PATH)
    if not path.exists():
        return defaults
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return defaults
    jd = payload.get("judge_demo") or {}
    seed = jd.get("flood_seed")
    extra = jd.get("corridor_extra")
    if seed is not None:
        defaults["seed"] = int(seed)
    if extra is not None:
        defaults["corridor_extra"] = int(extra)
    # Scenario-specific overrides from catalog pin table in app / scenarios.
    sid = str(jd.get("scenario_id") or payload.get("default_scenario_id") or "")
    by_id = {
        "qa_1": {"seed": 17012, "corridor_extra": 11},
        "qa_2": {"seed": 17025, "corridor_extra": 11},
        "qa_3": {"seed": 17025, "corridor_extra": 8},
        "qa_4": {"seed": 17025, "corridor_extra": 8},
        "qa_5": {"seed": 17025, "corridor_extra": 6},
    }
    if sid in by_id:
        defaults.update(by_id[sid])
    defaults["scenario_id"] = sid or "qa_2"
    return defaults


@dataclass
class TrafficIncident:
    """One human-readable disruption on the mock feed."""

    id: str
    kind: str  # congestion | soft_block | flood
    label: str
    severity: float
    area_hint: str
    edges: List[Tuple[Any, Any]] = field(default_factory=list)
    multiplier: float = DISRUPTION_SOFT_MULT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "severity": float(self.severity),
            "area_hint": self.area_hint,
            "edge_count": len(self.edges),
            "edges": [[u, v] for u, v in self.edges],
            "multiplier": float(self.multiplier),
        }


@dataclass
class CityConditionsSnapshot:
    """Named city conditions — product-shaped mock of a live feed response."""

    city: str
    as_of: str
    scenario_id: str
    scenario_name: str
    blurb: str
    bucket_id: int
    incidents: List[TrafficIncident]
    disruptions: EdgeDisruptionSet
    feed: str = "simulated"  # honest: not a paid live API
    source: str = "MockTrafficFeed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "city": self.city,
            "as_of": self.as_of,
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "blurb": self.blurb,
            "bucket_id": self.bucket_id,
            "feed": self.feed,
            "source": self.source,
            "incident_count": len(self.incidents),
            "incidents": [i.to_dict() for i in self.incidents],
            "disruptions": self.disruptions.to_serializable(),
        }

    @property
    def edge_disruptions(self) -> Dict[str, Any]:
        """Serializable disruptions for routing_service / session state."""
        return self.disruptions.to_serializable()


def _merge_disruption_sets(
    parts: Sequence[EdgeDisruptionSet],
) -> EdgeDisruptionSet:
    """Union edges; keep the strongest multiplier; prefer flood kind if present."""
    if not parts:
        return EdgeDisruptionSet(edges=[], multiplier=DISRUPTION_SOFT_MULT, kind="none")
    seen: Dict[Tuple[Any, Any], Tuple[Any, Any]] = {}
    max_mult = 1.0
    kind = "congestion"
    seed = parts[0].seed
    for p in parts:
        max_mult = max(max_mult, float(p.multiplier))
        if p.kind == "flood":
            kind = "flood"
        elif p.kind == "soft_block" and kind != "flood":
            kind = "soft_block"
        for u, v in p.normalized_edges():
            key = tuple(sorted((u, v)))
            seen[key] = (u, v)
        if p.seed is not None:
            seed = p.seed
    return EdgeDisruptionSet(
        edges=list(seen.values()),
        multiplier=float(max_mult),
        seed=seed,
        kind=kind,
    )


def _sample_incident(
    G: nx.Graph,
    *,
    spec: Dict[str, Any],
    incident_id: str,
    base_seed: int,
    near_node: Any = None,
) -> Tuple[TrafficIncident, EdgeDisruptionSet]:
    kind = str(spec.get("kind") or "congestion").strip().lower()
    seed = int(base_seed) + int(spec.get("seed_offset") or 0)
    corridor = int(spec.get("corridor_extra") or 3)
    n_seed = int(spec.get("n_seed_edges") or 1)
    label = str(spec.get("label") or kind)
    severity = float(spec.get("severity") or 0.5)
    area = str(spec.get("area_hint") or "")

    if kind in ("flood", "flooded"):
        dset = sample_flood_corridor(
            G,
            near_node=near_node,
            corridor_extra=corridor,
            multiplier=DISRUPTION_FLOOD_MULT,
            seed=seed,
        )
        mult = DISRUPTION_FLOOD_MULT
        kind_out = "flood"
    elif kind in ("soft_block", "closed", "closure", "block"):
        dset = sample_random_disruptions(
            G,
            n_seed_edges=n_seed,
            corridor_extra=corridor,
            multiplier=DISRUPTION_SOFT_BLOCK_MULT,
            soft_block=True,
            seed=seed,
        )
        mult = DISRUPTION_SOFT_BLOCK_MULT
        kind_out = "soft_block"
    else:
        dset = sample_random_disruptions(
            G,
            n_seed_edges=n_seed,
            corridor_extra=corridor,
            multiplier=DISRUPTION_SOFT_MULT,
            soft_block=False,
            seed=seed,
        )
        mult = DISRUPTION_SOFT_MULT
        kind_out = "congestion"

    edges = dset.normalized_edges()
    incident = TrafficIncident(
        id=incident_id,
        kind=kind_out,
        label=label,
        severity=severity,
        area_hint=area,
        edges=edges,
        multiplier=float(mult),
    )
    return incident, dset


def build_snapshot_for_scenario(
    G: nx.Graph,
    scenario: Dict[str, Any],
    *,
    when: Optional[datetime] = None,
    near_node: Any = None,
    bucket_id: Optional[int] = None,
    city: str = "Manila · Intramuros",
) -> CityConditionsSnapshot:
    """Materialize a scenario into edges + human incident labels."""
    when = when or _utc_now()
    bucket = int(bucket_id if bucket_id is not None else time_bucket_id(when))
    sid = str(scenario.get("id") or "unknown")
    base_seed = (bucket * 9973 + sum(ord(c) for c in sid)) % (2**31 - 1)

    near = near_node
    incident_specs = list(scenario.get("incidents") or ())
    # Judge pin: use demo_scenarios flood seed + prefer curated start if available.
    if scenario.get("judge_pin"):
        pin = _load_judge_flood_params()
        if incident_specs:
            first = dict(incident_specs[0])
            first["seed_offset"] = 0  # absolute seed below
            first["corridor_extra"] = int(pin.get("corridor_extra", 11))
            incident_specs[0] = first
            base_seed = int(pin.get("seed", 17025))
        if near is None and pin.get("near_start"):
            # Leave near_node to caller; flood sampler still works without it.
            pass

    incidents: List[TrafficIncident] = []
    parts: List[EdgeDisruptionSet] = []
    for i, spec in enumerate(incident_specs):
        inc, dset = _sample_incident(
            G,
            spec=spec,
            incident_id=f"{sid}:{i}",
            base_seed=base_seed,
            near_node=near if str(spec.get("kind", "")).lower().startswith("flood") else None,
        )
        # Absolute seed override for judge pin first incident.
        if scenario.get("judge_pin") and i == 0:
            pin = _load_judge_flood_params()
            dset = sample_flood_corridor(
                G,
                near_node=near,
                corridor_extra=int(pin.get("corridor_extra", 11)),
                multiplier=DISRUPTION_FLOOD_MULT,
                seed=int(pin.get("seed", 17025)),
            )
            inc = TrafficIncident(
                id=f"{sid}:0",
                kind="flood",
                label=str(spec.get("label") or "Flooded corridor near Pasig"),
                severity=float(spec.get("severity") or 0.95),
                area_hint=str(spec.get("area_hint") or "Judge-pinned flood"),
                edges=dset.normalized_edges(),
                multiplier=DISRUPTION_FLOOD_MULT,
            )
        incidents.append(inc)
        parts.append(dset)

    merged = _merge_disruption_sets(parts)
    return CityConditionsSnapshot(
        city=city,
        as_of=when.isoformat(timespec="seconds"),
        scenario_id=sid,
        scenario_name=str(scenario.get("name") or sid),
        blurb=str(scenario.get("blurb") or ""),
        bucket_id=bucket,
        incidents=incidents,
        disruptions=merged,
    )


def pick_scenario(
    *,
    scenario_id: Optional[str] = None,
    bucket_id: Optional[int] = None,
    rotate: bool = True,
) -> Dict[str, Any]:
    """Select catalog entry by id or by time-bucket rotation."""
    if scenario_id:
        for s in SCENARIO_CATALOG:
            if s["id"] == scenario_id:
                return dict(s)
    if not rotate:
        return dict(SCENARIO_CATALOG[0])
    bid = int(bucket_id if bucket_id is not None else time_bucket_id())
    # Exclude judge_flood from automatic rotation so product open ≠ Quantathon pin.
    rotatable = [s for s in SCENARIO_CATALOG if not s.get("judge_pin")]
    if not rotatable:
        rotatable = list(SCENARIO_CATALOG)
    return dict(rotatable[bid % len(rotatable)])


class MockTrafficFeed:
    """
    Product service: current city conditions from a simulated feed.

    ``current(G)`` — snapshot for the active time bucket (or forced scenario).
    ``refresh(G)`` — advance to the next catalog scenario (manual refresh).
    """

    city: str = "Manila · Intramuros"

    def __init__(self) -> None:
        self._forced_scenario_id: Optional[str] = None
        self._manual_offset: int = 0
        self._last: Optional[CityConditionsSnapshot] = None

    def force_scenario(self, scenario_id: Optional[str]) -> None:
        self._forced_scenario_id = scenario_id
        self._last = None

    def clear_force(self) -> None:
        self._forced_scenario_id = None

    def catalog(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": s["id"],
                "name": s["name"],
                "blurb": s.get("blurb", ""),
                "judge_pin": bool(s.get("judge_pin")),
            }
            for s in SCENARIO_CATALOG
        ]

    def current(
        self,
        G: nx.Graph,
        *,
        near_node: Any = None,
        when: Optional[datetime] = None,
        scenario_id: Optional[str] = None,
    ) -> CityConditionsSnapshot:
        when = when or _utc_now()
        bucket = time_bucket_id(when) + int(self._manual_offset)
        sid = scenario_id or self._forced_scenario_id
        scenario = pick_scenario(scenario_id=sid, bucket_id=bucket, rotate=sid is None)
        snap = build_snapshot_for_scenario(
            G,
            scenario,
            when=when,
            near_node=near_node,
            bucket_id=bucket,
            city=self.city,
        )
        self._last = snap
        return snap

    def refresh(
        self,
        G: nx.Graph,
        *,
        near_node: Any = None,
        when: Optional[datetime] = None,
    ) -> CityConditionsSnapshot:
        """Rotate to the next catalog scenario (product “refresh feed”)."""
        self._manual_offset += 1
        # Drop forced id so refresh actually moves the story.
        self._forced_scenario_id = None
        return self.current(G, near_node=near_node, when=when)

    def last(self) -> Optional[CityConditionsSnapshot]:
        return self._last


_feed: Optional[MockTrafficFeed] = None


def get_mock_traffic_feed(*, force_reload: bool = False) -> MockTrafficFeed:
    global _feed
    if _feed is None or force_reload:
        _feed = MockTrafficFeed()
    return _feed


def reset_mock_traffic_feed() -> None:
    global _feed
    _feed = None
