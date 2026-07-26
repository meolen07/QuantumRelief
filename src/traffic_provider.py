"""
Traffic / disruption feed providers.

Same Earthquake Escape pipeline for demo and production; only the feed differs:

  Demo  → MockTrafficProvider + MockTrafficFeed  (quake-forward simulated conditions)
  Live  → LiveTrafficProvider  (TomTom / HERE stub — needs TRAFFIC_API_KEY)

Configure with ``QR_TRAFFIC_MODE=demo|live`` (default ``demo``). Optional
``TRAFFIC_API_KEY`` unlocks the live stub path (still no paid API calls).

Production swap: keep routing_service / Earthquake Escape unchanged; only replace
the provider returned by ``get_traffic_provider()``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import networkx as nx

from src.dynamic_simulation import (
    DISRUPTION_FLOOD_MULT,
    DISRUPTION_SOFT_BLOCK_MULT,
    DISRUPTION_SOFT_MULT,
    EdgeDisruptionSet,
    apply_edge_disruptions,
    sample_flood_corridor,
    sample_random_disruptions,
)

DisruptionInput = Optional[
    Union[EdgeDisruptionSet, Sequence[Tuple[Any, Any]], dict]
]


class TrafficNotConfiguredError(RuntimeError):
    """Live traffic mode requested but API key / provider is not ready."""


@dataclass(frozen=True)
class TrafficModeInfo:
    """UI / ops label for the active traffic feed."""

    mode: str  # "demo" | "live"
    badge: str  # product-facing badge string
    provider_name: str
    live_ready: bool
    detail: str


def resolve_traffic_mode(explicit: Optional[str] = None) -> str:
    """
    Resolve ``demo`` vs ``live``.

    Precedence: explicit arg → ``QR_TRAFFIC_MODE`` env → default ``demo``.
    """
    raw = (explicit if explicit is not None else os.environ.get("QR_TRAFFIC_MODE", "demo"))
    mode = str(raw or "demo").strip().lower()
    if mode in ("live", "prod", "production"):
        return "live"
    return "demo"


def traffic_api_key() -> Optional[str]:
    key = os.environ.get("TRAFFIC_API_KEY") or os.environ.get("QR_TRAFFIC_API_KEY")
    if key is None:
        return None
    key = str(key).strip()
    return key or None


class TrafficProvider(ABC):
    """
    Source of edge weight multipliers (congestion / closure / flood).

    Escape and ``routing_service`` obtain and apply disruptions only through
    this interface — demo mocks vs live API stubs share the same apply path.
    """

    name: str = "traffic"
    mode: str = "demo"

    @abstractmethod
    def get_edge_disruptions(
        self,
        G: nx.Graph,
        *,
        kind: str = "congestion",
        near_node: Any = None,
        n_seed_edges: int = 1,
        corridor_extra: int = 3,
        seed: Optional[int] = None,
        multiplier: Optional[float] = None,
    ) -> EdgeDisruptionSet:
        """
        Return edges + soft multipliers for the current map overlay.

        ``kind``: ``congestion`` | ``soft_block`` | ``flood``
        """

    def apply_to_graph(
        self,
        G: nx.Graph,
        disruptions: DisruptionInput = None,
        *,
        multiplier: Optional[float] = None,
    ) -> List[Tuple[Any, Any]]:
        """Apply soft weight penalties in-place; sole apply entry for routing."""
        return apply_edge_disruptions(G, disruptions, multiplier=multiplier)

    def current_conditions(self, G: nx.Graph, *, near_node: Any = None):
        """
        Named city conditions snapshot when the provider supports a feed.

        Default: None (live stub until wired). Mock overrides.
        """
        _ = (G, near_node)
        return None

    def refresh_conditions(self, G: nx.Graph, *, near_node: Any = None):
        """Rotate / refresh the feed snapshot when supported."""
        return self.current_conditions(G, near_node=near_node)

    def mode_info(self) -> TrafficModeInfo:
        return TrafficModeInfo(
            mode=self.mode,
            badge=self.badge_label(),
            provider_name=self.name,
            live_ready=False,
            detail="",
        )

    def badge_label(self) -> str:
        if self.mode == "live":
            return "Live conditions · traffic API"
        return "Live conditions · simulated feed"


class MockTrafficProvider(TrafficProvider):
    """
    Production-shaped mock feed:

    - ``current_conditions`` / ``refresh_conditions`` → ``MockTrafficFeed``
      city snapshot (incidents + merged ``EdgeDisruptionSet``)
    - ``get_edge_disruptions`` still samples congestion / closure / flood for
      manual overlays (and judge pins via ``seed`` / ``corridor_extra``)
    """

    name = "mock"
    mode = "demo"

    def current_conditions(self, G: nx.Graph, *, near_node: Any = None):
        from src.mock_traffic_feed import get_mock_traffic_feed

        return get_mock_traffic_feed().current(G, near_node=near_node)

    def refresh_conditions(self, G: nx.Graph, *, near_node: Any = None):
        from src.mock_traffic_feed import get_mock_traffic_feed

        return get_mock_traffic_feed().refresh(G, near_node=near_node)

    def get_edge_disruptions(
        self,
        G: nx.Graph,
        *,
        kind: str = "congestion",
        near_node: Any = None,
        n_seed_edges: int = 1,
        corridor_extra: int = 3,
        seed: Optional[int] = None,
        multiplier: Optional[float] = None,
    ) -> EdgeDisruptionSet:
        kind_norm = str(kind or "congestion").strip().lower()
        if kind_norm in ("flood", "flooded"):
            return sample_flood_corridor(
                G,
                near_node=near_node,
                corridor_extra=int(corridor_extra),
                multiplier=float(multiplier or DISRUPTION_FLOOD_MULT),
                seed=seed,
            )
        soft_block = kind_norm in ("soft_block", "closed", "closure", "block")
        if soft_block:
            mult = float(multiplier or DISRUPTION_SOFT_BLOCK_MULT)
        else:
            mult = float(multiplier or DISRUPTION_SOFT_MULT)
        return sample_random_disruptions(
            G,
            n_seed_edges=int(n_seed_edges),
            corridor_extra=int(corridor_extra),
            multiplier=mult,
            soft_block=soft_block,
            seed=seed,
        )

    def mode_info(self) -> TrafficModeInfo:
        return TrafficModeInfo(
            mode="demo",
            badge=self.badge_label(),
            provider_name=self.name,
            live_ready=False,
            detail=(
                "Simulated Manila conditions catalog "
                "(congestion / closure / flood / earthquake hazard). "
                "Swap to LiveTrafficProvider for production — same Escape pipeline."
            ),
        )


class LiveTrafficProvider(TrafficProvider):
    """
    Stub for TomTom / HERE live traffic.

    Requires ``TRAFFIC_API_KEY`` (or ``QR_TRAFFIC_API_KEY``). Does **not** call
    paid APIs yet — when configured, returns an empty disruption set and leaves
    a TODO for the real integration.
    """

    name = "live"
    mode = "live"

    def __init__(self, api_key: Optional[str] = None, *, allow_empty: bool = True):
        self._api_key = (api_key if api_key is not None else traffic_api_key())
        self._allow_empty = bool(allow_empty)

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def get_edge_disruptions(
        self,
        G: nx.Graph,
        *,
        kind: str = "congestion",
        near_node: Any = None,
        n_seed_edges: int = 1,
        corridor_extra: int = 3,
        seed: Optional[int] = None,
        multiplier: Optional[float] = None,
    ) -> EdgeDisruptionSet:
        if not self.is_configured:
            raise TrafficNotConfiguredError(
                "Live traffic mode requires TRAFFIC_API_KEY (or QR_TRAFFIC_API_KEY). "
                "Set QR_TRAFFIC_MODE=demo for simulated city conditions, "
                "or configure a traffic API key for the live stub."
            )
        # TODO: TomTom / HERE incident + flow → edge multipliers.
        # Keep the graph usable: empty overlay until the paid client is wired.
        _ = (G, kind, near_node, n_seed_edges, corridor_extra, seed, multiplier)
        if not self._allow_empty:
            raise TrafficNotConfiguredError(
                "Live traffic API client is not implemented yet "
                "(TomTom / HERE TODO). Use QR_TRAFFIC_MODE=demo."
            )
        return EdgeDisruptionSet(
            edges=[],
            multiplier=float(multiplier or DISRUPTION_SOFT_MULT),
            seed=seed,
            kind=str(kind or "congestion"),
        )

    def mode_info(self) -> TrafficModeInfo:
        if self.is_configured:
            detail = (
                "TRAFFIC_API_KEY set — live client stub (no paid calls yet; "
                "empty overlay until TomTom / HERE is wired)."
            )
        else:
            detail = (
                "Live mode without TRAFFIC_API_KEY — disruption requests fail "
                "with a clear configure message."
            )
        return TrafficModeInfo(
            mode="live",
            badge=self.badge_label(),
            provider_name=self.name,
            live_ready=self.is_configured,
            detail=detail,
        )


_provider: Optional[TrafficProvider] = None
_provider_mode: Optional[str] = None


def get_traffic_provider(
    mode: Optional[str] = None,
    *,
    force_reload: bool = False,
) -> TrafficProvider:
    """
    Singleton provider for the active ``QR_TRAFFIC_MODE``.

    Default is ``MockTrafficProvider`` (Cloud / judge-safe, no API keys).
    Production: set ``QR_TRAFFIC_MODE=live`` (+ key) to swap the provider only.
    """
    global _provider, _provider_mode
    resolved = resolve_traffic_mode(mode)
    if (
        not force_reload
        and _provider is not None
        and _provider_mode == resolved
    ):
        return _provider
    if resolved == "live":
        _provider = LiveTrafficProvider()
    else:
        _provider = MockTrafficProvider()
    _provider_mode = resolved
    return _provider


def reset_traffic_provider() -> None:
    """Clear singleton (tests / mode switches)."""
    global _provider, _provider_mode
    _provider = None
    _provider_mode = None


def apply_provider_disruptions(
    G: nx.Graph,
    disruptions: DisruptionInput = None,
    *,
    provider: Optional[TrafficProvider] = None,
    multiplier: Optional[float] = None,
) -> List[Tuple[Any, Any]]:
    """Apply disruptions via the active (or given) traffic provider."""
    prov = provider or get_traffic_provider()
    return prov.apply_to_graph(G, disruptions, multiplier=multiplier)


def traffic_mode_badge(mode: Optional[str] = None) -> str:
    """Short UI badge: simulated feed vs live traffic API."""
    return get_traffic_provider(mode).badge_label()


def active_feed_disruptions(
    G: nx.Graph,
    *,
    near_node: Any = None,
    refresh: bool = False,
) -> Tuple[Optional[dict], Optional[Any]]:
    """
    Load (or refresh) the active provider's city conditions for routing.

    Returns ``(edge_disruptions_serializable, snapshot_or_None)``.
    Snapshot may also carry ``epicenter_lonlat`` / ``has_disaster`` when the
    mock feed includes an earthquake / hazard incident.
    In live mode without a wired client, returns ``(None, None)``.
    """
    prov = get_traffic_provider()
    snap = (
        prov.refresh_conditions(G, near_node=near_node)
        if refresh
        else prov.current_conditions(G, near_node=near_node)
    )
    if snap is None:
        return None, None
    disruptions = getattr(snap, "edge_disruptions", None)
    if disruptions is None and hasattr(snap, "disruptions"):
        disruptions = snap.disruptions.to_serializable()
    return disruptions, snap


def epicenter_from_snapshot(snap: Any) -> Optional[Tuple[float, float]]:
    """
    Extract ``(lon, lat)`` epicenter from a city-conditions snapshot.

    Returns None when the feed has no active disaster / hazard incident.
    """
    if snap is None:
        return None
    epi = getattr(snap, "epicenter_lonlat", None)
    if epi is not None and len(epi) == 2:
        return float(epi[0]), float(epi[1])
    if isinstance(snap, dict):
        if snap.get("epi_lon") is not None and snap.get("epi_lat") is not None:
            return float(snap["epi_lon"]), float(snap["epi_lat"])
        for inc in snap.get("incidents") or []:
            if str(inc.get("kind") or "").lower() in (
                "earthquake",
                "quake",
                "hazard",
                "hazard_zone",
                "disaster",
                "seismic",
            ):
                if inc.get("epi_lon") is not None and inc.get("epi_lat") is not None:
                    return float(inc["epi_lon"]), float(inc["epi_lat"])
    return None
