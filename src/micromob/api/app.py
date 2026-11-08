"""``POST /v1/discharge``: saturation flow and dynamic PCU for a mix (runs one simulation in a worker thread)."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .. import __version__
from ..engine import TYPES, Road
from ..experiments import discharge


class DischargeRequest(BaseModel):
    mix: dict[str, float] = Field(description="shares by type: car, motorcycle, scooter, cargo_bike")
    approach_width_m: float = Field(7.0, ge=3.0, le=15.0)
    vehicles: int = Field(120, ge=20, le=600)
    seed: int = 0


def create_app() -> FastAPI:
    app = FastAPI(title="Mixed-traffic discharge", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/discharge")
    async def run(r: DischargeRequest) -> dict[str, float]:
        bad = set(r.mix) - set(TYPES)
        if bad or not r.mix or any(v < 0 for v in r.mix.values()) or sum(r.mix.values()) <= 0:
            raise HTTPException(422, f"mix must be non-negative shares of {sorted(TYPES)}")
        return await run_in_threadpool(discharge, r.mix, r.vehicles, r.seed, None, Road(width=r.approach_width_m))

    return app
