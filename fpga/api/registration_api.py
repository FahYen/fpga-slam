from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

MAX_REG_CORRESPONDENCES = 16384

app = FastAPI(title="SG-SLAM FPGA Registration API", version="0.1.0")


class RegistrationAccumulateRequest(BaseModel):
    src_xyz: List[float] = Field(description="Flattened source XYZ array [x0,y0,z0,...]")
    tgt_xyz: List[float] = Field(description="Flattened target XYZ array [x0,y0,z0,...]")
    labels: List[int] = Field(description="Semantic labels aligned with correspondence index")
    correspondence_count: Optional[int] = Field(
        default=None,
        description="Requested correspondence count. If omitted, inferred from input lengths.",
    )
    kernel: float = Field(default=1.0 / 3.0, gt=0.0)
    backend: Literal["cpu-proto", "xrt-stub"] = Field(default="cpu-proto")

    @model_validator(mode="after")
    def validate_shapes(self) -> "RegistrationAccumulateRequest":
        if len(self.src_xyz) % 3 != 0:
            raise ValueError("src_xyz length must be divisible by 3")
        if len(self.tgt_xyz) % 3 != 0:
            raise ValueError("tgt_xyz length must be divisible by 3")

        src_points = len(self.src_xyz) // 3
        tgt_points = len(self.tgt_xyz) // 3
        if src_points != tgt_points:
            raise ValueError("src_xyz and tgt_xyz must contain the same number of points")
        if len(self.labels) != src_points:
            raise ValueError("labels length must match point count")

        return self


class RegistrationAccumulateResponse(BaseModel):
    backend: str
    requested_count: int
    used_count: int
    dropped_count: int
    jtj_out: List[float]
    jtr_out: List[float]


class RegistrationKernelLimits(BaseModel):
    max_correspondences: int


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/v1/registration/limits", response_model=RegistrationKernelLimits)
def get_limits() -> RegistrationKernelLimits:
    return RegistrationKernelLimits(max_correspondences=MAX_REG_CORRESPONDENCES)


def _semantic_weight(label: int) -> float:
    if label in (16, 18, 19):
        return 1.2
    if label in (20, 21, 22, 23, 24, 25):
        return 0.0
    return 1.0


def _accumulate_cpu_proto(payload: RegistrationAccumulateRequest) -> RegistrationAccumulateResponse:
    src_points = len(payload.src_xyz) // 3
    requested = payload.correspondence_count if payload.correspondence_count is not None else src_points
    if requested < 0:
        requested = 0

    bounded_by_input = min(requested, src_points)
    used_count = min(bounded_by_input, MAX_REG_CORRESPONDENCES)
    dropped_count = requested - used_count

    jtj = [0.0] * 36
    jtr = [0.0] * 6

    kernel = float(payload.kernel)
    kernel_sq = kernel * kernel

    for i in range(used_count):
        base = i * 3
        sx = float(payload.src_xyz[base + 0])
        sy = float(payload.src_xyz[base + 1])
        sz = float(payload.src_xyz[base + 2])

        tx = float(payload.tgt_xyz[base + 0])
        ty = float(payload.tgt_xyz[base + 1])
        tz = float(payload.tgt_xyz[base + 2])

        rx = sx - tx
        ry = sy - ty
        rz = sz - tz

        residual2 = rx * rx + ry * ry + rz * rz
        denom = kernel + residual2
        w = kernel_sq / (denom * denom)
        semantic_w = _semantic_weight(int(payload.labels[i]))

        # J = [I | -hat(s)]
        J = [
            [1.0, 0.0, 0.0, 0.0, sz, -sy],
            [0.0, 1.0, 0.0, -sz, 0.0, sx],
            [0.0, 0.0, 1.0, sy, -sx, 0.0],
        ]
        r = [rx, ry, rz]

        for c in range(6):
            jtr_acc = J[0][c] * r[0] + J[1][c] * r[1] + J[2][c] * r[2]
            jtr[c] += w * semantic_w * jtr_acc

        for c0 in range(6):
            for c1 in range(6):
                jtj_acc = J[0][c0] * J[0][c1] + J[1][c0] * J[1][c1] + J[2][c0] * J[2][c1]
                jtj[c0 * 6 + c1] += w * jtj_acc

    return RegistrationAccumulateResponse(
        backend="cpu-proto",
        requested_count=requested,
        used_count=used_count,
        dropped_count=dropped_count,
        jtj_out=jtj,
        jtr_out=jtr,
    )


@app.post("/v1/registration/accumulate", response_model=RegistrationAccumulateResponse)
def registration_accumulate(payload: RegistrationAccumulateRequest) -> RegistrationAccumulateResponse:
    if payload.backend == "cpu-proto":
        return _accumulate_cpu_proto(payload)

    if payload.backend == "xrt-stub":
        raise HTTPException(
            status_code=501,
            detail="xrt-stub backend is not implemented yet. Use backend=cpu-proto for now.",
        )

    raise HTTPException(status_code=400, detail="Unsupported backend")
