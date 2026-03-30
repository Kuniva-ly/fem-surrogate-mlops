import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

GEOMETRY_TYPES = ("with_hole", "without_hole", "with_hole_moving")

REQUIRED_COLUMNS = {
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
}


def _coerce_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["hole_radius_ratio", "hole_cx_ratio", "hole_cy_ratio"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "geometry_type" not in out.columns:
        out["geometry_type"] = pd.NA
    out["geometry_type"] = out["geometry_type"].astype("string").str.strip()
    return out


def _assert_ranges(df: pd.DataFrame) -> None:
    checks = {
        "length_m": (df["length_m"] > 0).all(),
        "height_m": (df["height_m"] > 0).all(),
        "young_modulus_pa": (df["young_modulus_pa"] > 0).all(),
        "poisson_ratio": ((df["poisson_ratio"] > 0) & (df["poisson_ratio"] < 0.5)).all(),
        "traction_pa": (df["traction_pa"] >= 0).all(),
        "mesh_nx": (df["mesh_nx"] >= 8).all(),
        "mesh_ny": (df["mesh_ny"] >= 4).all(),
        "max_displacement_m": (df["max_displacement_m"] >= 0).all(),
        "max_von_mises_pa": (df["max_von_mises_pa"] >= 0).all(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Range validation failed for columns: {', '.join(failed)}")

    if (df["material_category"].astype(str).str.strip() == "").any():
        raise ValueError("material_category must be a non-empty string")
    if (df["dimension_category"].astype(str).str.strip() == "").any():
        raise ValueError("dimension_category must be a non-empty string")

    if ((df["hole_radius_ratio"].notna()) & ~((df["hole_radius_ratio"] > 0) & (df["hole_radius_ratio"] < 0.5))).any():
        raise ValueError("hole_radius_ratio must be in (0, 0.5) when present")
    if ((df["hole_cx_ratio"].notna()) & ~((df["hole_cx_ratio"] > 0) & (df["hole_cx_ratio"] < 1))).any():
        raise ValueError("hole_cx_ratio must be in (0, 1) when present")
    if ((df["hole_cy_ratio"].notna()) & ~((df["hole_cy_ratio"] > 0) & (df["hole_cy_ratio"] < 1))).any():
        raise ValueError("hole_cy_ratio must be in (0, 1) when present")

    gt = df["geometry_type"]
    gt_present = gt.notna() & (gt != "")
    invalid_gt = gt_present & (~gt.isin(GEOMETRY_TYPES))
    if bool(invalid_gt.any()):
        bad_values = sorted(gt.loc[invalid_gt].astype(str).unique().tolist())
        raise ValueError(f"Invalid geometry_type values: {bad_values}")

    with_hole = gt == "with_hole"
    without_hole = gt == "without_hole"
    with_hole_moving = gt == "with_hole_moving"

    if bool((with_hole & df["hole_radius_ratio"].isna()).any()):
        raise ValueError("with_hole rows must define hole_radius_ratio")
    if bool((with_hole & (df["hole_cx_ratio"].notna() | df["hole_cy_ratio"].notna())).any()):
        raise ValueError("with_hole rows must not define hole_cx_ratio/hole_cy_ratio")

    if bool((without_hole & (df["hole_radius_ratio"].notna() | df["hole_cx_ratio"].notna() | df["hole_cy_ratio"].notna())).any()):
        raise ValueError("without_hole rows must not define hole_* columns")

    if bool((with_hole_moving & df["hole_radius_ratio"].isna()).any()):
        raise ValueError("with_hole_moving rows must define hole_radius_ratio")
    if bool((with_hole_moving & (df["hole_cx_ratio"].isna() | df["hole_cy_ratio"].isna())).any()):
        raise ValueError("with_hole_moving rows must define hole_cx_ratio and hole_cy_ratio")


def validate_folder(input_dir: Path) -> None:
    parquet_files = sorted(input_dir.rglob("*.parquet"))
    csv_files = sorted(input_dir.rglob("*.csv"))

    # Prefer parquet if present to avoid schema mixing with legacy CSV files.
    if parquet_files:
        files = parquet_files
    else:
        files = csv_files

    if not files:
        raise FileNotFoundError(f"No parquet/csv files found in {input_dir}")

    frames = []
    for f in files:
        if f.suffix == ".parquet":
            frames.append(pd.read_parquet(f))
        else:
            frames.append(pd.read_csv(f))
    df = _coerce_optional_columns(pd.concat(frames, ignore_index=True))

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")

    if df[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Null values detected in required columns")

    if df["simulation_id"].duplicated().any():
        raise ValueError("Duplicate simulation_id detected")

    _assert_ranges(df)

    geometry_values = (
        df["geometry_type"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    stats = {
        "rows": len(df),
        "files": len(files),
        "geometry_types": sorted(geometry_values),
        "max_vm": float(df["max_von_mises_pa"].max()),
        "max_disp": float(df["max_displacement_m"].max()),
        "validated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    print("Validation successful:")
    for k, v in stats.items():
        print(f"- {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated simulation parquet data.")
    parser.add_argument("--input", required=True, type=Path, help="Input directory containing parquet files")
    args = parser.parse_args()
    validate_folder(args.input)


if __name__ == "__main__":
    main()
