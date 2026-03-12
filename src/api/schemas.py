"""Schémas Pydantic de requête / réponse pour l'API Surrogate FEM."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Requête ───────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Paramètres d'entrée pour une prédiction surrogate FEM unique.

    Vous pouvez fournir :
    - Uniquement les paramètres physiques bruts (le feature engineering est appliqué automatiquement), ou
    - Des features pré-calculées correspondant au vecteur feature_cols du modèle.

    Champs minimaux requis pour le calcul automatique :
        geometry_type, length_m, height_m, young_modulus_pa,
        poisson_ratio, traction_pa, material_category, dimension_category
    """

    # Géométrie
    geometry_type: Literal["with_hole", "without_hole", "with_hole_moving"] = Field(
        ..., description="Variante de géométrie de la plaque"
    )
    length_m: float = Field(..., gt=0, description="Longueur de la plaque en mètres (> 0)")
    height_m: float = Field(..., gt=0, description="Hauteur de la plaque en mètres (> 0)")

    # Matériau
    young_modulus_pa: float = Field(..., gt=0, description="Module de Young en Pa")
    poisson_ratio: float = Field(..., gt=0, lt=0.5, description="Coefficient de Poisson dans (0, 0.5)")
    traction_pa: float = Field(..., ge=0, description="Contrainte de traction appliquée en Pa")

    # Labels catégoriels (utilisés pour l'encodage du modèle)
    material_category: str = Field(..., min_length=1, description="ex. 'steel', 'aluminum', 'titanium'")
    dimension_category: str = Field(..., min_length=1, description="ex. 'small', 'medium', 'large'")

    # Paramètres optionnels du trou
    hole_radius_ratio: Optional[float] = Field(
        None, gt=0, lt=0.5,
        description="Rayon du trou en fraction de min(L, H) — requis pour les géométries with_hole"
    )
    hole_cx_ratio: Optional[float] = Field(
        None, gt=0, lt=1,
        description="Position x du centre du trou en fraction de la longueur de plaque — requis pour with_hole_moving"
    )
    hole_cy_ratio: Optional[float] = Field(
        None, gt=0, lt=1,
        description="Position y du centre du trou en fraction de la hauteur de plaque — requis pour with_hole_moving"
    )

    # Paramètres maillage (utilisés par le feature engineering pour compatibilité ; pas des features modèle)
    mesh_nx: int = Field(120, ge=8, description="Divisions du maillage selon x")
    mesh_ny: int = Field(24, ge=4, description="Divisions du maillage selon y")

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


# ── Réponse ───────────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    max_displacement_m: float = Field(..., description="Déplacement maximum prédit en mètres")
    max_von_mises_pa: float = Field(..., description="Contrainte von Mises maximale prédite en Pa")


class PredictionResponse(BaseModel):
    model_version: str = Field(..., description="Version du modèle utilisée pour l'inférence")
    predictions: PredictionResult
    input_summary: dict = Field(..., description="Écho des paramètres d'entrée principaux")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    model_loaded: bool
    model_version: Optional[str] = None


class VersionResponse(BaseModel):
    api_version: str
    model_version: Optional[str]
    model_name: str
    feature_count: int
