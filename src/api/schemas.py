"""Pydantic request / response schemas for the FEM Surrogate API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CAT_COLS: list[str] = ["geometry_type", "material_category", "dimension_category"]


# Request

class PredictionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "geometry_type":      "with_hole",
            "length_m":           0.2,
            "height_m":           0.05,
            "young_modulus_pa":   210000000000,
            "poisson_ratio":      0.3,
            "traction_pa":        50000000,
            "material_category":  "steel",
            "dimension_category": "small",
            "hole_radius_ratio":  0.2,
        }
    })
    """Input parameters for a single FEM surrogate prediction.

    You may provide:
    - Only the raw physical parameters (feature engineering is applied automatically), or
    - Pre-computed features matching the model's feature_cols vector.

    Minimum required fields for automatic computation:
        geometry_type, length_m, height_m, young_modulus_pa,
        poisson_ratio, traction_pa, material_category, dimension_category
    """

    # Geometry
    geometry_type: Literal["with_hole", "without_hole", "with_hole_moving"] = Field(
        ..., description="Plate geometry variant"
    )
    length_m: float = Field(..., gt=0, description="Plate length in metres (> 0)")
    height_m: float = Field(..., gt=0, description="Plate height in metres (> 0)")

    # Material
    young_modulus_pa: float = Field(..., gt=0, description="Young's modulus in Pa")
    poisson_ratio: float = Field(..., gt=0, lt=0.5, description="Poisson ratio in (0, 0.5)")
    traction_pa: float = Field(..., ge=0, description="Applied traction stress in Pa")

    # Categorical labels (used for model encoding)
    material_category: str = Field(..., min_length=1, description="e.g. 'steel', 'aluminum', 'titanium'")
    dimension_category: str = Field(..., min_length=1, description="e.g. 'small', 'medium', 'large'")

    # Optional hole parameters
    hole_radius_ratio: Optional[float] = Field(
        None, gt=0, lt=0.5,
        description="Hole radius as a fraction of min(L, H) — required for with_hole geometries"
    )
    hole_cx_ratio: Optional[float] = Field(
        None, gt=0, lt=1,
        description="Hole centre x position as a fraction of plate length — required for with_hole_moving"
    )
    hole_cy_ratio: Optional[float] = Field(
        None, gt=0, lt=1,
        description="Hole centre y position as a fraction of plate height — required for with_hole_moving"
    )

    # Mesh parameters (used by feature engineering for compatibility; not model features)
    mesh_nx: int = Field(120, ge=8, description="Mesh divisions along x")
    mesh_ny: int = Field(24, ge=4, description="Mesh divisions along y")

    @field_validator("material_category", "dimension_category")
    @classmethod
    def no_empty_string(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()

    @model_validator(mode="after")
    def check_geometry_constraints(self) -> "PredictionRequest":
        gt = self.geometry_type
        if gt in ("with_hole", "with_hole_moving") and self.hole_radius_ratio is None:
            raise ValueError(f"hole_radius_ratio is required for geometry_type='{gt}'")
        if gt == "with_hole_moving":
            if self.hole_cx_ratio is None or self.hole_cy_ratio is None:
                raise ValueError("hole_cx_ratio and hole_cy_ratio are required for 'with_hole_moving'")
        if gt == "without_hole":
            if self.hole_radius_ratio is not None:
                raise ValueError("hole_radius_ratio must not be set for 'without_hole'")
        return self


# Response

class PredictionResult(BaseModel):
    max_displacement_m: float = Field(..., description="Predicted maximum displacement in metres")
    max_von_mises_pa: float = Field(..., description="Predicted maximum von Mises stress in Pa")


class PredictionResponse(BaseModel):
    model_version: str = Field(..., description="Model version used for inference")
    predictions: PredictionResult
    input_summary: dict = Field(..., description="Echo of the main input parameters")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    model_loaded: bool
    model_version: Optional[str] = None


class VersionResponse(BaseModel):
    api_version: str
    model_version: Optional[str]
    model_name: str
    feature_count: int
