import argparse
from pathlib import Path

import pandas as pd

TARGET_COLS = ["max_displacement_m", "max_von_mises_pa"]
BASE_DROP_COLS = ["simulation_id", "timestamp", "solver_name", "solver_version", "data_version"]
REQUIRED_COLS = {
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
}
OPTIONAL_HOLE_COLS = ["hole_radius_ratio", "hole_cx_ratio", "hole_cy_ratio"]
GEOMETRY_TYPES = ("with_hole", "without_hole", "with_hole_moving")


def _read_raw(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {input_dir}")
    return pd.concat((pd.read_parquet(p) for p in files), ignore_index=True)


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature building: {sorted(missing)}")

    out = df.copy()
    for col in OPTIONAL_HOLE_COLS:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "geometry_type" not in out.columns:
        out["geometry_type"] = pd.NA
    out["geometry_type"] = out["geometry_type"].astype("string").str.strip()
    invalid_geometry = out["geometry_type"].isna() | (~out["geometry_type"].isin(GEOMETRY_TYPES))
    inferred_geometry = pd.Series("without_hole", index=out.index, dtype="string")
    moving_mask = out["hole_cx_ratio"].notna() & out["hole_cy_ratio"].notna()
    hole_mask = out["hole_radius_ratio"].notna() & (out["hole_radius_ratio"] > 0)
    inferred_geometry.loc[hole_mask] = "with_hole"
    inferred_geometry.loc[moving_mask] = "with_hole_moving"
    out.loc[invalid_geometry, "geometry_type"] = inferred_geometry.loc[invalid_geometry]

    out["area_m2"] = out["length_m"] * out["height_m"]
    out["aspect_ratio"] = out["length_m"] / out["height_m"]
    out["traction_over_E"] = out["traction_pa"] / out["young_modulus_pa"]
    out["traction_over_area"] = out["traction_pa"] / out["area_m2"]
    out["mesh_total_cells"] = out["mesh_nx"] * out["mesh_ny"]
    out["mesh_density"] = out["mesh_total_cells"] / out["area_m2"]
    out["has_hole"] = hole_mask.astype(int)
    out["has_moving_hole"] = moving_mask.astype(int)

    # Keep geometry-specific optional columns numeric and model-safe.
    out["hole_radius_ratio"] = out["hole_radius_ratio"].fillna(0.0)
    out["hole_cx_ratio"] = out["hole_cx_ratio"].fillna(0.0)
    out["hole_cy_ratio"] = out["hole_cy_ratio"].fillna(0.0)
    return out


def _split(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
    if len(df) < 3:
        raise ValueError("Need at least 3 rows to build train/val/test splits")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError(
            "Invalid split sizes with current dataset size. "
            "Increase data volume or adjust ratios to keep non-empty splits."
        )

    train_df = shuffled.iloc[:n_train].copy()
    val_df = shuffled.iloc[n_train : n_train + n_val].copy()
    test_df = shuffled.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in TARGET_COLS + BASE_DROP_COLS]


def build_features(
    input_dir: Path,
    out_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    features_out_dir: Path | None = None,
) -> None:
    raw = _read_raw(input_dir)
    dataset = _engineer(raw)
    train_df, val_df, test_df = _split(dataset, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    feature_cols = _feature_columns(train_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    features_dir = features_out_dir or out_dir
    features_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    test_path = out_dir / "test.parquet"
    features_dataset_path = features_dir / "features.parquet"
    features_path = features_dir / "feature_columns.txt"

    dataset.to_parquet(features_dataset_path, index=False)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    features_path.write_text("\n".join(feature_cols), encoding="utf-8")

    print(f"Built features dataset from: {input_dir}")
    print(f"All rows: {len(dataset)} -> {features_dataset_path}")
    print(f"Train rows: {len(train_df)} -> {train_path}")
    print(f"Val rows: {len(val_df)} -> {val_path}")
    print(f"Test rows: {len(test_df)} -> {test_path}")
    print(f"Feature columns ({len(feature_cols)}): {features_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed feature splits from raw simulation parquet data.")
    parser.add_argument("--input", required=True, type=Path, help="Raw input folder containing parquet files")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Output folder for train/val/test parquet")
    parser.add_argument(
        "--features-out-dir",
        type=Path,
        default=None,
        help="Optional output folder dedicated to feature-store artifacts (features.parquet, feature_columns.txt)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_features(
        input_dir=args.input,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        features_out_dir=args.features_out_dir,
    )


if __name__ == "__main__":
    main()


