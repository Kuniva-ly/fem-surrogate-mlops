import argparse
import os
from pathlib import Path

import pandas as pd

GEOMETRY_TYPES = {"with_hole", "without_hole", "with_hole_moving"}

SCHEMA_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'geometry_type_enum') THEN
        CREATE TYPE geometry_type_enum AS ENUM ('with_hole', 'without_hole', 'with_hole_moving');
    END IF;
END$$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'geometry_type_enum') THEN
        IF NOT EXISTS (
            SELECT 1
            FROM pg_enum e
            JOIN pg_type t ON t.oid = e.enumtypid
            WHERE t.typname = 'geometry_type_enum' AND e.enumlabel = 'with_hole_moving'
        ) THEN
            ALTER TYPE geometry_type_enum ADD VALUE 'with_hole_moving';
        END IF;
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS simulation_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    simulation_id UUID NOT NULL UNIQUE,
    ts TIMESTAMPTZ NOT NULL,
    geometry_type geometry_type_enum NOT NULL,
    material_category TEXT NOT NULL,
    dimension_category TEXT NOT NULL,
    length_m DOUBLE PRECISION NOT NULL CHECK (length_m > 0),
    height_m DOUBLE PRECISION NOT NULL CHECK (height_m > 0),
    young_modulus_pa DOUBLE PRECISION NOT NULL CHECK (young_modulus_pa > 0),
    poisson_ratio DOUBLE PRECISION NOT NULL CHECK (poisson_ratio > 0 AND poisson_ratio < 0.5),
    traction_pa DOUBLE PRECISION NOT NULL CHECK (traction_pa >= 0),
    mesh_nx INTEGER NOT NULL CHECK (mesh_nx >= 8),
    mesh_ny INTEGER NOT NULL CHECK (mesh_ny >= 4),
    hole_radius_ratio DOUBLE PRECISION CHECK (hole_radius_ratio > 0 AND hole_radius_ratio < 0.5),
    hole_cx_ratio DOUBLE PRECISION CHECK (hole_cx_ratio > 0 AND hole_cx_ratio < 1),
    hole_cy_ratio DOUBLE PRECISION CHECK (hole_cy_ratio > 0 AND hole_cy_ratio < 1),
    max_displacement_m DOUBLE PRECISION NOT NULL CHECK (max_displacement_m >= 0),
    max_von_mises_pa DOUBLE PRECISION NOT NULL CHECK (max_von_mises_pa >= 0),
    solver_name TEXT NOT NULL,
    solver_version TEXT NOT NULL,
    data_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        (
            geometry_type = 'with_hole'
            AND hole_radius_ratio IS NOT NULL
            AND hole_cx_ratio IS NULL
            AND hole_cy_ratio IS NULL
        )
        OR (
            geometry_type = 'with_hole_moving'
            AND hole_radius_ratio IS NOT NULL
            AND hole_cx_ratio IS NOT NULL
            AND hole_cy_ratio IS NOT NULL
        )
        OR (
            geometry_type = 'without_hole'
            AND hole_radius_ratio IS NULL
            AND hole_cx_ratio IS NULL
            AND hole_cy_ratio IS NULL
        )
    )
);
"""

UPSERT_SQL = """
INSERT INTO simulation_records (
    simulation_id,
    ts,
    geometry_type,
    material_category,
    dimension_category,
    length_m,
    height_m,
    young_modulus_pa,
    poisson_ratio,
    traction_pa,
    mesh_nx,
    mesh_ny,
    hole_radius_ratio,
    hole_cx_ratio,
    hole_cy_ratio,
    max_displacement_m,
    max_von_mises_pa,
    solver_name,
    solver_version,
    data_version
)
VALUES %s
ON CONFLICT (simulation_id) DO UPDATE SET
    ts = EXCLUDED.ts,
    geometry_type = EXCLUDED.geometry_type,
    material_category = EXCLUDED.material_category,
    dimension_category = EXCLUDED.dimension_category,
    length_m = EXCLUDED.length_m,
    height_m = EXCLUDED.height_m,
    young_modulus_pa = EXCLUDED.young_modulus_pa,
    poisson_ratio = EXCLUDED.poisson_ratio,
    traction_pa = EXCLUDED.traction_pa,
    mesh_nx = EXCLUDED.mesh_nx,
    mesh_ny = EXCLUDED.mesh_ny,
    hole_radius_ratio = EXCLUDED.hole_radius_ratio,
    hole_cx_ratio = EXCLUDED.hole_cx_ratio,
    hole_cy_ratio = EXCLUDED.hole_cy_ratio,
    max_displacement_m = EXCLUDED.max_displacement_m,
    max_von_mises_pa = EXCLUDED.max_von_mises_pa,
    solver_name = EXCLUDED.solver_name,
    solver_version = EXCLUDED.solver_version,
    data_version = EXCLUDED.data_version
"""


def _read_input(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {input_dir}")
    return pd.concat((pd.read_parquet(p) for p in files), ignore_index=True)


def _normalize_geometry_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["hole_radius_ratio", "hole_cx_ratio", "hole_cy_ratio"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "geometry_type" not in out.columns:
        out["geometry_type"] = pd.NA
    out["geometry_type"] = out["geometry_type"].astype("string").str.strip()
    return out


def _infer_geometry_type(data: pd.DataFrame) -> pd.Series:
    moving_mask = data["hole_cx_ratio"].notna() & data["hole_cy_ratio"].notna()
    hole_mask = data["hole_radius_ratio"].notna() & (data["hole_radius_ratio"] > 0)
    inferred = pd.Series("without_hole", index=data.index, dtype="string")
    inferred.loc[hole_mask] = "with_hole"
    inferred.loc[moving_mask] = "with_hole_moving"
    return inferred


def _resolve_geometry_type(df: pd.DataFrame, geometry_type: str) -> pd.DataFrame:
    out = _normalize_geometry_columns(df)

    valid_mask = out["geometry_type"].isin(list(GEOMETRY_TYPES))
    inferred = _infer_geometry_type(out)
    out.loc[~valid_mask, "geometry_type"] = inferred.loc[~valid_mask]

    if geometry_type != "auto":
        out["geometry_type"] = geometry_type

    if geometry_type == "with_hole":
        out["hole_radius_ratio"] = out["hole_radius_ratio"].fillna(0.15)
        out["hole_cx_ratio"] = pd.NA
        out["hole_cy_ratio"] = pd.NA
    elif geometry_type == "with_hole_moving":
        out["hole_radius_ratio"] = out["hole_radius_ratio"].fillna(0.15)
        out["hole_cx_ratio"] = out["hole_cx_ratio"].fillna(0.5)
        out["hole_cy_ratio"] = out["hole_cy_ratio"].fillna(0.5)
    elif geometry_type == "without_hole":
        out["hole_radius_ratio"] = pd.NA
        out["hole_cx_ratio"] = pd.NA
        out["hole_cy_ratio"] = pd.NA
    else:
        with_hole_mask = out["geometry_type"] == "with_hole"
        with_hole_moving_mask = out["geometry_type"] == "with_hole_moving"
        without_hole_mask = out["geometry_type"] == "without_hole"
        out.loc[with_hole_mask, "hole_radius_ratio"] = out.loc[with_hole_mask, "hole_radius_ratio"].fillna(0.15)
        out.loc[with_hole_mask, ["hole_cx_ratio", "hole_cy_ratio"]] = pd.NA
        out.loc[with_hole_moving_mask, "hole_radius_ratio"] = out.loc[with_hole_moving_mask, "hole_radius_ratio"].fillna(0.15)
        out.loc[with_hole_moving_mask, "hole_cx_ratio"] = out.loc[with_hole_moving_mask, "hole_cx_ratio"].fillna(0.5)
        out.loc[with_hole_moving_mask, "hole_cy_ratio"] = out.loc[with_hole_moving_mask, "hole_cy_ratio"].fillna(0.5)
        out.loc[without_hole_mask, ["hole_radius_ratio", "hole_cx_ratio", "hole_cy_ratio"]] = pd.NA

    invalid = ~out["geometry_type"].isin(list(GEOMETRY_TYPES))
    if bool(invalid.any()):
        bad_values = sorted(out.loc[invalid, "geometry_type"].dropna().astype(str).unique().tolist())
        raise ValueError(f"Invalid geometry_type values: {bad_values}")

    return out


def _prepare(df: pd.DataFrame, geometry_type: str) -> list[tuple]:
    required = [
        "simulation_id",
        "timestamp",
        "material_category",
        "dimension_category",
        "length_m",
        "height_m",
        "young_modulus_pa",
        "poisson_ratio",
        "traction_pa",
        "mesh_nx",
        "mesh_ny",
        "max_displacement_m",
        "max_von_mises_pa",
        "solver_name",
        "solver_version",
        "data_version",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = _resolve_geometry_type(df, geometry_type)
    data["ts"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")

    cols = [
        "simulation_id",
        "ts",
        "geometry_type",
        "material_category",
        "dimension_category",
        "length_m",
        "height_m",
        "young_modulus_pa",
        "poisson_ratio",
        "traction_pa",
        "mesh_nx",
        "mesh_ny",
        "hole_radius_ratio",
        "hole_cx_ratio",
        "hole_cy_ratio",
        "max_displacement_m",
        "max_von_mises_pa",
        "solver_name",
        "solver_version",
        "data_version",
    ]
    records = data[cols].where(pd.notna(data[cols]), None).itertuples(index=False, name=None)
    return list(records)


def load_to_postgres(
    input_dir: Path,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    geometry_type: str,
    truncate: bool,
) -> int:
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'psycopg2'. Install with: pip install psycopg2-binary"
        ) from exc

    df = _read_input(input_dir)
    rows = _prepare(df, geometry_type)
    if not rows:
        return 0

    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=database,
        user=user,
        password=password,
    )
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
                cur.execute(SCHEMA_SQL)
                if truncate:
                    cur.execute("TRUNCATE TABLE simulation_records;")
                execute_values(cur, UPSERT_SQL, rows, page_size=1000)
    finally:
        conn.close()

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load local parquet simulation data into PostgreSQL.")
    parser.add_argument("--input", required=True, type=Path, help="Folder containing parquet files")
    parser.add_argument(
        "--geometry-type",
        choices=["auto", "with_hole", "without_hole", "with_hole_moving"],
        default="auto",
    )
    parser.add_argument("--truncate", action="store_true", help="Truncate simulation_records before loading")
    parser.add_argument("--db-host", default=os.getenv("POSTGRES_HOST", "localhost"))
    parser.add_argument("--db-port", type=int, default=int(os.getenv("POSTGRES_PORT", "5432")))
    parser.add_argument("--db-name", default=os.getenv("POSTGRES_DB", "surrogate"))
    parser.add_argument("--db-user", default=os.getenv("POSTGRES_USER", "surrogate"))
    parser.add_argument("--db-password", default=os.getenv("POSTGRES_PASSWORD", "surrogate"))
    args = parser.parse_args()

    inserted = load_to_postgres(
        input_dir=args.input,
        host=args.db_host,
        port=args.db_port,
        database=args.db_name,
        user=args.db_user,
        password=args.db_password,
        geometry_type=args.geometry_type,
        truncate=args.truncate,
    )
    print(
        f"Loaded {inserted} rows into PostgreSQL table simulation_records "
        f"({args.db_host}:{args.db_port}/{args.db_name})."
    )


if __name__ == "__main__":
    main()
