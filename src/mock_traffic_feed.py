"""
Mock city traffic / incident feed (product service).

Demo mode ships a **named conditions snapshot** — same shape production would
get from TomTom / HERE — so Live Escape can open on “current road conditions”
without paid API calls.

Scenarios rotate by **Manila time-of-day** (morning / rush / evening / night)
plus a deterministic wall-clock bucket (feels live; reproducible).
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

# Manila civic clock for time-of-day scenario pools.
MANILA_TZ = "Asia/Manila"

# Human-facing catalog — ids are stable; seeds / params drive edge sampling.
# Labels lean on Intramuros geography the graph can support (Pasig, Fort,
# walls corridor, Padre Burgos) — no POI geocoding.
SCENARIO_CATALOG: Tuple[Dict[str, Any], ...] = (
    {
        "id": "quiet_morning",
        "name": "Quiet morning · light jam",
        "blurb": "Mild congestion on plaza-rim lanes — baseline civic morning.",
        "periods": ("morning", "night"),
        "incidents": (
            {
                "kind": "congestion",
                "label": "Light congestion · plaza rim",
                "severity": 0.3,
                "area_hint": "Central plaza / municipal core",
                "corridor_extra": 2,
                "n_seed_edges": 1,
                "seed_offset": 40,
            },
        ),
    },
    {
        "id": "rush_hour_arterial",
        "name": "Rush-hour · Fort / Burgos arterials",
        "blurb": "Peak traffic on Intramuros approaches (Fort Santiago · Burgos).",
        "periods": ("rush", "evening"),
        "incidents": (
            {
                "kind": "congestion",
                "label": "Congestion · Fort / Padre Burgos approach",
                "severity": 0.55,
                "area_hint": "North Gate · Fort Santiago corridor",
                "corridor_extra": 4,
                "n_seed_edges": 1,
                "seed_offset": 101,
            },
        ),
    },
    {
        "id": "flood_pasig",
        "name": "Flood watch · Pasig riverside",
        "blurb": "Soft flood stand-in on Pasig-side low ground (Ondoy-like ×12).",
        "periods": ("morning", "rush", "evening", "night"),
        "incidents": (
            {
                "kind": "flood",
                "label": "Flooded corridor · Pasig riverside",
                "severity": 0.9,
                "area_hint": "Pasig River · east / northeast low ground",
                "corridor_extra": 11,
                "seed_offset": 17025,
            },
        ),
    },
    {
        "id": "closure_walls",
        "name": "Soft closure · walls corridor",
        "blurb": "Temporary soft-closed stretch along the historic walls.",
        "periods": ("midday", "night"),
        "incidents": (
            {
                "kind": "soft_block",
                "label": "Closed corridor · Intramuros walls",
                "severity": 0.75,
                "area_hint": "Muralla / walls corridor",
                "corridor_extra": 5,
                "n_seed_edges": 1,
                "seed_offset": 220,
            },
        ),
    },
    {
        "id": "closure_historic",
        "name": "Temporary corridor closure",
        "blurb": "Soft closed corridor in the historic core (still passable).",
        "periods": ("midday", "night"),
        "incidents": (
            {
                "kind": "soft_block",
                "label": "Closed corridor · historic core",
                "severity": 0.75,
                "area_hint": "Intramuros walls · historic core",
                "corridor_extra": 5,
                "n_seed_edges": 1,
                "seed_offset": 220,
            },
        ),
    },
    {
        "id": "mixed_evening",
        "name": "Evening mix · jam + walls closure",
        "blurb": "North Gate congestion plus a soft closure on east wall lanes.",
        "periods": ("evening",),
        "incidents": (
            {
                "kind": "congestion",
                "label": "Congestion · North Gate approach",
                "severity": 0.5,
                "area_hint": "North Gate · Fort Santiago approach",
                "corridor_extra": 3,
                "n_seed_edges": 1,
                "seed_offset": 330,
            },
            {
                "kind": "soft_block",
                "label": "Soft closure · east wall lanes",
                "severity": 0.65,
                "area_hint": "East walls / Victoria corridor",
                "corridor_extra": 3,
                "n_seed_edges": 1,
                "seed_offset": 331,
            },
        ),
    },
    {
        "id": "night_quiet",
        "name": "Night · sparse traffic",
        "blurb": "Light overnight jam; corridors mostly clear.",
        "periods": ("night",),
        "incidents": (
            {
                "kind": "congestion",
                "label": "Sparse night traffic",
                "severity": 0.2,
                "area_hint": "Intramuros core · overnight",
                "corridor_extra": 2,
                "n_seed_edges": 1,
                "seed_offset": 55,
            },
        ),
    },
    {
        "id": "judge_flood",
        "name": "Judge demo · flooded corridor",
        "blurb": "Pinned Quantathon flood seed (Hybrid travel-win corridor).",
        "judge_pin": True,
        "periods": (),
        "incidents": (
            {
                "kind": "flood",
                "label": "Flooded corridor · Pasig riverside",
                "severity": 0.95,
                "area_hint": "Judge-pinned flood · Pasig side",
                "corridor_extra": 11,
                "seed_offset": 17025,
            },
        ),
    },
)

# Preferred catalog ids per Manila time-of-day (judge_flood excluded).
TIME_OF_DAY_POOLS: Dict[str, Tuple[str, ...]] = {
    "morning": ("quiet_morning", "flood_pasig", "closure_historic"),
    "rush": ("rush_hour_arterial", "flood_pasig", "closure_walls"),
    "midday": ("rush_hour_arterial", "closure_walls", "closure_historic", "flood_pasig"),
    "evening": ("mixed_evening", "rush_hour_arterial", "flood_pasig"),
    "night": ("night_quiet", "closure_historic", "flood_pasig", "quiet_morning"),
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_manila(when: Optional[datetime] = None) -> datetime:
    """Wall clock in Asia/Manila (falls back to UTC+8 if zoneinfo missing)."""
    when = when or _utc_now()
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        return when.astimezone(ZoneInfo(MANILA_TZ))
    except Exception:
        from datetime import timedelta

        return when.astimezone(timezone(timedelta(hours=8)))


def time_of_day_period(when: Optional[datetime] = None) -> str:
    """
    Manila civic period for scenario pools.

    morning 05–09 · rush 09–11 & 16–19 · midday 11–16 · evening 19–22 · night else
    """
    local = _to_manila(when)
    h = int(local.hour)
    if 5 <= h < 9:
        return "morning"
    if 9 <= h < 11 or 16 <= h < 19:
        return "rush"
    if 11 <= h < 16:
        return "midday"
    if 19 <= h < 22:
        return "evening"
    return "night"


def time_of_day_label(period: str) -> str:
    return {
        "morning": "Morning",
        "rush": "Rush hour",
        "midday": "Midday",
        "evening": "Evening",
        "night": "Night",
    }.get(period, period.title())


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
    time_of_day: str = ""
    time_of_day_label: str = ""
    local_clock: str = ""

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
            "time_of_day": self.time_of_day,
            "time_of_day_label": self.time_of_day_label,
            "local_clock": self.local_clock,
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
    local = _to_manila(when)
    period = time_of_day_period(when)
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
                label=str(spec.get("label") or "Flooded corridor · Pasig riverside"),
                severity=float(spec.get("severity") or 0.95),
                area_hint=str(spec.get("area_hint") or "Judge-pinned flood · Pasig side"),
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
        time_of_day=period,
        time_of_day_label=time_of_day_label(period),
        local_clock=local.strftime("%H:%M %Z"),
    )


def _catalog_by_id() -> Dict[str, Dict[str, Any]]:
    return {str(s["id"]): dict(s) for s in SCENARIO_CATALOG}


def pick_scenario(
    *,
    scenario_id: Optional[str] = None,
    bucket_id: Optional[int] = None,
    rotate: bool = True,
    when: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Select catalog entry by id, else by Manila time-of-day + bucket."""
    if scenario_id:
        by_id = _catalog_by_id()
        if scenario_id in by_id:
            return by_id[scenario_id]
    if not rotate:
        return dict(SCENARIO_CATALOG[0])

    when = when or _utc_now()
    bid = int(bucket_id if bucket_id is not None else time_bucket_id(when))
    period = time_of_day_period(when)
    pool_ids = TIME_OF_DAY_POOLS.get(period) or ()
    by_id = _catalog_by_id()
    pool = [by_id[i] for i in pool_ids if i in by_id and not by_id[i].get("judge_pin")]
    if not pool:
        pool = [dict(s) for s in SCENARIO_CATALOG if not s.get("judge_pin")]
    if not pool:
        pool = [dict(s) for s in SCENARIO_CATALOG]
    return dict(pool[bid % len(pool)])


class MockTrafficFeed:
    """
    Product service: current city conditions from a simulated feed.

    ``current(G)`` — snapshot for the active time-of-day + bucket (or forced).
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
                "periods": list(s.get("periods") or ()),
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
        scenario = pick_scenario(
            scenario_id=sid, bucket_id=bucket, rotate=sid is None, when=when
        )
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
