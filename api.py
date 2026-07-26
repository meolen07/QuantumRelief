"""
QuantumRelief API — commercial Quantum Routing REST surface (B2B).

Exposes Hybrid QML routing for Manila (Intramuros) via FastAPI.
Optional Classical FiLM + Dijkstra comparison fields support the product
3-way story. Reuses ``src/routing_service.py``.

Traffic architecture
--------------------
Demo (default): ``MockTrafficProvider`` + ``MockTrafficFeed`` city conditions
are applied automatically when ``use_mock_feed`` is true (or omitted in demo
mode). Production: set ``QR_TRAFFIC_MODE=live`` and swap to
``LiveTrafficProvider`` (TomTom / HERE stub) — same ``/calculate_route``
contract; only the feed provider changes. No paid API calls in this repo.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.routing_service import (  # noqa: E402
    calculate_hybrid_route,
    get_routing_resources,
)
from src.traffic_provider import (  # noqa: E402
    get_traffic_provider,
    resolve_traffic_mode,
)
from src.utils import GRAPH_CACHE_PATH  # noqa: E402


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RoutingRequest(BaseModel):
    """Request body for Hybrid QML escape routing.

    All coordinates are WGS84 ``[latitude, longitude]`` pairs
    (same convention as Folium / the Streamlit Earthquake Escape UX).

    Disruptions: omit ``edge_disruptions`` in demo mode to use the active
    mock city feed; pass an explicit serializable disruption dict to override;
    set ``use_mock_feed=false`` to route without feed overlays.
    """

    start_coords: List[float] = Field(
        ...,
        description="Trip start as [lat, lon] (WGS84).",
        min_length=2,
        max_length=2,
        examples=[[14.5895, 120.9750]],
    )
    epicenter_coords: List[float] = Field(
        ...,
        description=(
            "Optional extreme hazard epicenter as [lat, lon] (WGS84). "
            "Always required by the current contract; place far from the "
            "corridor for everyday traffic-only runs."
        ),
        min_length=2,
        max_length=2,
        examples=[[14.5850, 120.9780]],
    )
    exit_coords: List[float] = Field(
        ...,
        description="Target exit / safe zone as [lat, lon] (WGS84).",
        min_length=2,
        max_length=2,
        examples=[[14.5920, 120.9720]],
    )
    include_comparison: bool = Field(
        True,
        description=(
            "If true, attach Classical FiLM + Dijkstra comparison fields "
            "(travel time, exit reached, path overlap)."
        ),
    )
    use_mock_feed: Optional[bool] = Field(
        None,
        description=(
            "When true (or omitted while QR_TRAFFIC_MODE=demo), apply the "
            "active MockTrafficFeed city conditions. Production live mode "
            "ignores this until LiveTrafficProvider is wired. Set false to "
            "disable automatic feed overlays."
        ),
    )
    edge_disruptions: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Optional serializable EdgeDisruptionSet override: "
            "{edges:[[u,v],...], multiplier:float, kind:str, seed:int}. "
            "When set, skips the automatic mock feed."
        ),
    )

    @field_validator("start_coords", "epicenter_coords", "exit_coords")
    @classmethod
    def _validate_lat_lon(cls, v: List[float]) -> List[float]:
        if len(v) != 2:
            raise ValueError("Coordinate must be a list of exactly 2 floats: [lat, lon].")
        lat, lon = float(v[0]), float(v[1])
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitude out of range: {lat}")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitude out of range: {lon}")
        return [lat, lon]


class PathWaypoint(BaseModel):
    """One node on the predicted escape path."""

    node_id: Any = Field(..., description="Graph node ID (OSM/NetworkX).")
    lat: float = Field(..., description="Node latitude (WGS84).")
    lon: float = Field(..., description="Node longitude (WGS84).")


class EngineCompare(BaseModel):
    """Optional Classical / Dijkstra summary for 3-way compare."""

    engine: str
    estimated_travel_time: float
    exit_reached: bool
    hops: int
    overlap_vs_dijkstra_pct: Optional[float] = None
    predicted_path: Optional[List[PathWaypoint]] = None


class RoutingResponse(BaseModel):
    """Hybrid QML routing result for B2B consumers."""

    predicted_path: List[PathWaypoint] = Field(
        ...,
        description="Ordered escape waypoints: node_id + lat/lon (Hybrid hero).",
    )
    estimated_travel_time: float = Field(
        ...,
        description="Sum of live edge travel weights along the Hybrid path.",
    )
    quantum_contribution: float = Field(
        ...,
        description="PHN quantum-branch share percentage (e.g. 45.3).",
        examples=[45.3],
    )
    exit_reached: Optional[bool] = Field(
        None, description="Whether the Hybrid path terminates at the snapped exit."
    )
    hops: Optional[int] = Field(None, description="Number of edges traversed (Hybrid).")
    start_node: Optional[Any] = Field(None, description="Snapped start node ID.")
    exit_node: Optional[Any] = Field(None, description="Snapped exit node ID.")
    node_ids: Optional[List[Any]] = Field(
        None, description="Parallel list of node IDs along predicted_path."
    )
    model: Optional[str] = Field(
        "Hybrid QML (HQNN)", description="Engine used for the hero prediction."
    )
    classical: Optional[EngineCompare] = Field(
        None, description="Classical FiLM ablation comparison (optional)."
    )
    dijkstra: Optional[EngineCompare] = Field(
        None, description="Dijkstra oracle baseline with full dynamic weights."
    )
    comparison: Optional[Dict[str, Any]] = Field(
        None,
        description="Narrative + travel-time ratios for Hybrid vs Classical / Dijkstra.",
    )
    feed: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Active traffic feed metadata (mock city conditions or live stub). "
            "Demo = simulated catalog; Prod = live provider once wired."
        ),
    )


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Best-effort preload; first request also lazy-loads if this fails."""
    try:
        get_routing_resources()
    except Exception:
        pass
    yield


app = FastAPI(
    title="QuantumRelief API",
    description=(
        "B2B Quantum Routing API for hybrid quantum–classical routing under "
        "dynamic edge costs (Manila Intramuros). "
        "**Architecture:** Demo = same route contract + `MockTrafficProvider` / "
        "`MockTrafficFeed`; Production = same contract + `LiveTrafficProvider` "
        "(TomTom / HERE stub via `QR_TRAFFIC_MODE=live`). "
        "Hybrid QML is the hero; optional Classical FiLM + Dijkstra fields "
        "support a 3-way comparison. Tagline: The map is always dynamic — "
        "Hybrid QML routes under changing edge costs."
    ),
    version="1.2.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Liveness / readiness probes."},
        {
            "name": "routing",
            "description": (
                "Hybrid QML dynamic-edge routing (B2B) + optional 3-way compare. "
                "Applies mock city feed disruptions in demo mode by default."
            ),
        },
        {
            "name": "traffic",
            "description": "Traffic feed mode / conditions (demo mock vs live stub).",
        },
    ],
)


@app.get("/", tags=["health"], summary="API health check")
def root():
    """Confirm the QuantumRelief API process is up."""
    info = get_traffic_provider().mode_info()
    return {
        "status": "QuantumRelief API is running",
        "traffic_mode": info.mode,
        "traffic_badge": info.badge,
        "architecture": "Demo=MockTrafficFeed · Prod=LiveTrafficProvider",
    }


@app.get(
    "/api/v1/traffic/status",
    tags=["traffic"],
    summary="Active traffic feed mode",
)
def traffic_status():
    """Report demo vs live provider (production swaps provider only)."""
    info = get_traffic_provider().mode_info()
    return {
        "mode": info.mode,
        "badge": info.badge,
        "provider": info.provider_name,
        "live_ready": info.live_ready,
        "detail": info.detail,
        "resolved": resolve_traffic_mode(),
        "note": (
            "Production: set QR_TRAFFIC_MODE=live + TRAFFIC_API_KEY; "
            "route contract unchanged."
        ),
    }


def _engine_payload(summary) -> Optional[EngineCompare]:
    if summary is None:
        return None
    wps = None
    if summary.waypoints:
        wps = [PathWaypoint(**wp) for wp in summary.waypoints]
    return EngineCompare(
        engine=summary.engine,
        estimated_travel_time=summary.travel_time,
        exit_reached=summary.exit_reached,
        hops=summary.hops,
        overlap_vs_dijkstra_pct=summary.overlap_vs_dijkstra_pct,
        predicted_path=wps,
    )


@app.post(
    "/api/v1/calculate_route",
    response_model=RoutingResponse,
    tags=["routing"],
    summary="Calculate Hybrid QML route (+ optional 3-way compare)",
)
def calculate_route(body: RoutingRequest) -> RoutingResponse:
    """
    Snap start / epicenter / exit to the Manila graph, apply active feed
    disruptions (mock city conditions in demo), run Algorithm 1 dynamics,
    predict with Hybrid QML, and optionally attach Classical + Dijkstra.

    Production swap: only ``TrafficProvider`` changes — this endpoint stays.
    """
    if not GRAPH_CACHE_PATH.exists():
        pass

    try:
        result = calculate_hybrid_route(
            body.start_coords,
            body.epicenter_coords,
            body.exit_coords,
            include_comparison=body.include_comparison,
            edge_disruptions=body.edge_disruptions,
            use_mock_feed=body.use_mock_feed,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        msg = str(exc)
        if "No path" in msg:
            raise HTTPException(status_code=404, detail=msg) from exc
        raise HTTPException(status_code=500, detail=msg) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Model or routing failure: {exc}",
        ) from exc

    if not result.exit_reached and result.hops == 0:
        raise HTTPException(
            status_code=404,
            detail="No path found between start and exit under live dynamics.",
        )

    return RoutingResponse(
        predicted_path=[PathWaypoint(**wp) for wp in result.predicted_path],
        estimated_travel_time=result.estimated_travel_time,
        quantum_contribution=result.quantum_contribution,
        exit_reached=result.exit_reached,
        hops=result.hops,
        start_node=result.start_node,
        exit_node=result.exit_node,
        node_ids=result.node_ids,
        model=result.model,
        classical=_engine_payload(result.classical),
        dijkstra=_engine_payload(result.dijkstra),
        comparison=result.comparison,
        feed=result.feed,
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
