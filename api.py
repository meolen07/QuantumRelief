"""
QuantumRelief API — commercial Quantum Routing REST surface (B2B roadmap).

Exposes Hybrid QML emergency escape routing for Manila (Intramuros) via FastAPI.
Reuses ``src/routing_service.py`` and existing Phase 1–3 engines — does not
rewrite graph / ML logic.
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
from src.utils import GRAPH_CACHE_PATH  # noqa: E402


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RoutingRequest(BaseModel):
    """Request body for Hybrid QML escape routing.

    All coordinates are WGS84 ``[latitude, longitude]`` pairs
    (same convention as Folium / the Streamlit Crisis UX).
    """

    start_coords: List[float] = Field(
        ...,
        description="Evacuee start as [lat, lon] (WGS84).",
        min_length=2,
        max_length=2,
        examples=[[14.5895, 120.9750]],
    )
    epicenter_coords: List[float] = Field(
        ...,
        description="Earthquake epicenter as [lat, lon] (WGS84).",
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


class RoutingResponse(BaseModel):
    """Hybrid QML routing result for B2B consumers."""

    predicted_path: List[PathWaypoint] = Field(
        ...,
        description="Ordered escape waypoints: node_id + lat/lon.",
    )
    estimated_travel_time: float = Field(
        ...,
        description="Sum of live edge travel weights along the path (nominal minutes).",
    )
    quantum_contribution: float = Field(
        ...,
        description="PHN quantum-branch share percentage (e.g. 45.3).",
        examples=[45.3],
    )
    exit_reached: Optional[bool] = Field(
        None, description="Whether the path terminates at the snapped exit."
    )
    hops: Optional[int] = Field(None, description="Number of edges traversed.")
    start_node: Optional[Any] = Field(None, description="Snapped start node ID.")
    exit_node: Optional[Any] = Field(None, description="Snapped exit node ID.")
    node_ids: Optional[List[Any]] = Field(
        None, description="Parallel list of node IDs along predicted_path."
    )
    model: Optional[str] = Field(
        "Hybrid QML (HQNN)", description="Engine used for this prediction."
    )


# ---------------------------------------------------------------------------
# App lifespan — optional warm load (still safe if models train later)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Best-effort preload; first request also lazy-loads if this fails."""
    try:
        get_routing_resources()
    except Exception:
        # Graph / checkpoint may still be building (e.g. mid-retrain) — lazy on demand
        pass
    yield


app = FastAPI(
    title="QuantumRelief API",
    description=(
        "Commercial Quantum Routing API for hybrid quantum–classical emergency "
        "escape routing in Manila (Intramuros). Snaps WGS84 coordinates to the "
        "road graph, applies Algorithm 1 hazard dynamics, and rolls out the "
        "Hybrid QML (PennyLane PHN) next-hop policy."
    ),
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "health",
            "description": "Liveness / readiness probes.",
        },
        {
            "name": "routing",
            "description": "Hybrid QML emergency escape routing (B2B).",
        },
    ],
)


@app.get("/", tags=["health"], summary="API health check")
def root():
    """Confirm the QuantumRelief API process is up."""
    return {"status": "QuantumRelief API is running"}


@app.post(
    "/api/v1/calculate_route",
    response_model=RoutingResponse,
    tags=["routing"],
    summary="Calculate Hybrid QML escape route",
)
def calculate_route(body: RoutingRequest) -> RoutingResponse:
    """
    Snap start / epicenter / exit to the Manila graph, run Algorithm 1
    dynamics with the epicenter, and predict an escape path with Hybrid QML.
    """
    if not GRAPH_CACHE_PATH.exists():
        # load_or_build_graph can synthesize offline — try resources first
        pass

    try:
        result = calculate_hybrid_route(
            body.start_coords,
            body.epicenter_coords,
            body.exit_coords,
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
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
