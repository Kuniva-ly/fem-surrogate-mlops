"""Pipeline d'ingénierie des features pour le surrogate ML FEM.

Modifications par rapport à l'original :
- Suppression des lignes orphelines : source with_hole mais hole_radius_ratio non enregistré
  (10 000 runs FEniCS de sim_v1/date=2026-02-10 où la colonne hole_radius_ratio
  était absente ; la remplir avec 0 corrompait toutes les features de trou).
- Ajout de features physiques :
    epsilon, delta_theory, Kt_theory, sigma_net, ligaments, eccentricity, espace log
- Suppression des features de maillage constantes (mesh_nx = 120, mesh_ny = 24 pour toutes
  les lignes, variance nulle, elles n'introduisaient que des artefacts de formule proxy).
- Correction de la valeur de remplissage hole_cx/cy : utiliser 0.5 (centre de la plaque) et non 0.0.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constantes ────────────────────────────────────────────────────────────────
TARGET_COLS    = ["max_displacement_m", "max_von_mises_pa"]
BASE_DROP_COLS = ["simulation_id", "timestamp", "solver_name", "solver_version", "data_version"]
REQUIRED_COLS  = {
    "simulation_id", "timestamp",
    "material_category", "dimension_category",
    "length_m", "height_m",
    "young_modulus_pa", "poisson_ratio", "traction_pa",
    "mesh_nx", "mesh_ny",
    "max_displacement_m", "max_von_mises_pa",
}
OPTIONAL_HOLE_COLS = ["hole_radius_ratio", "hole_cx_ratio", "hole_cy_ratio"]
GEOMETRY_TYPES     = ("with_hole", "without_hole", "with_hole_moving")

# Colonnes de maillage qui sont constantes dans le jeu de données actuel (120×24 pour chaque ligne).
# Les conserver n'ajoute que du bruit ; elles sont exclues de l'ensemble de features du modèle.
MESH_COLS = ["mesh_nx", "mesh_ny"]

_EPS = 1e-12


# ── E/S ───────────────────────────────────────────────────────────────────────
def _read_raw(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {input_dir}")
    return pd.concat((pd.read_parquet(p) for p in files), ignore_index=True)


# ── Qualité des données ───────────────────────────────────────────────────────
def _drop_ambiguous_geometry(df: pd.DataFrame, keep_ambiguous_as_without_hole: bool = False) -> pd.DataFrame:
    """Supprime les lignes dont la géométrie ne peut pas être déterminée.

    Une ligne est ambiguë quand :
      - geometry_type est NaN (non étiqueté explicitement), ET
      - hole_radius_ratio est NaN (pas de signal de taille de trou), ET
      - hole_cx_ratio / hole_cy_ratio sont NaN (pas de signal de position).

    Ce sont typiquement des runs FEniCS des premiers lots sim_v1
    sauvegardés avant que hole_radius_ratio soit inclus dans le schéma de sortie.
    Les traiter comme des plaques simples (fill=0) est physiquement incorrect car
    le solveur INCLUAIT bien un trou — le paramètre n'a juste jamais été persisté.
    """
    has_geometry_type = df["geometry_type"].notna() if "geometry_type" in df.columns else pd.Series(False, index=df.index)
    has_r  = (pd.to_numeric(df.get("hole_radius_ratio", pd.Series(np.nan, index=df.index)), errors="coerce").notna())
    has_cx = (pd.to_numeric(df.get("hole_cx_ratio",     pd.Series(np.nan, index=df.index)), errors="coerce").notna())

    has_signal = has_geometry_type | has_r | has_cx
    ambiguous_mask = ~has_signal
    n_ambiguous = int(ambiguous_mask.sum())
    if n_ambiguous == 0:
        return df.copy()

    if keep_ambiguous_as_without_hole:
        print(
            f"[build_features] Kept {n_ambiguous} ambiguous rows and treated them as without_hole "
            f"(conservative mode)."
        )
        out = df.copy()
        if "geometry_type" not in out.columns:
            out["geometry_type"] = pd.NA
        out.loc[ambiguous_mask, "geometry_type"] = "without_hole"
        return out

    print(
        f"[build_features] Dropped {n_ambiguous} rows with no geometry signal "
        f"(orphaned with_hole FEniCS runs missing hole_radius_ratio)."
    )
    return df.loc[has_signal].copy()


# ── Ingénierie des features ───────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, require_targets: bool = True) -> pd.DataFrame:
    """Calcule les features ML physiques à partir des colonnes de simulation brutes.

    Entrées attendues
    -----------------
    Obligatoires : colonnes listées dans REQUIRED_COLS (sans les cibles si require_targets=False).
    Optionnelles : hole_radius_ratio, hole_cx_ratio, hole_cy_ratio, geometry_type.

    Sortie
    ------
    Le dataframe d'entrée enrichi de colonnes de features dérivées.
    Les colonnes de maillage (mesh_nx, mesh_ny) ne sont PAS supprimées ici — elles transitent
    vers les fichiers parquet — mais _feature_columns() les exclut du modèle.
    """
    required_cols = set(REQUIRED_COLS)
    if not require_targets:
        required_cols = required_cols - set(TARGET_COLS)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature building: {sorted(missing)}")

    out = df.copy()

    # ── Colonnes optionnelles de trou ────────────────────────────────────────
    for col in OPTIONAL_HOLE_COLS:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # ── Inférence du type de géométrie ───────────────────────────────────────
    if "geometry_type" not in out.columns:
        out["geometry_type"] = pd.NA
    out["geometry_type"] = out["geometry_type"].astype("string").str.strip()

    moving_mask = out["hole_cx_ratio"].notna() & out["hole_cy_ratio"].notna()
    hole_mask   = out["hole_radius_ratio"].notna() & (out["hole_radius_ratio"] > 0)

    invalid_geometry = out["geometry_type"].isna() | (~out["geometry_type"].isin(GEOMETRY_TYPES))
    inferred = pd.Series("without_hole", index=out.index, dtype="string")
    inferred.loc[hole_mask]   = "with_hole"
    inferred.loc[moving_mask] = "with_hole_moving"
    out.loc[invalid_geometry, "geometry_type"] = inferred.loc[invalid_geometry]

    # Indicateurs binaires
    out["has_hole"]        = hole_mask.astype(int)
    out["has_moving_hole"] = moving_mask.astype(int)

    # ── Remplissage des valeurs par défaut des trous APRÈS inférence de géométrie ──
    # hole_cx/cy par défaut = 0.5 (centre de la plaque), PAS 0 (coin).
    out["hole_radius_ratio"] = out["hole_radius_ratio"].fillna(0.0)
    out["hole_cx_ratio"]     = out["hole_cx_ratio"].fillna(0.5)
    out["hole_cy_ratio"]     = out["hole_cy_ratio"].fillna(0.5)

    # ── Géométrie brute ──────────────────────────────────────────────────────
    L   = out["length_m"]
    H   = out["height_m"]
    E   = out["young_modulus_pa"]
    nu  = out["poisson_ratio"]
    sig = out["traction_pa"]
    r   = out["hole_radius_ratio"]
    cx  = out["hole_cx_ratio"]
    cy  = out["hole_cy_ratio"]

    out["area_m2"]      = L * H
    out["aspect_ratio"] = L / (H + _EPS)

    # ── Dimensions absolues du trou ──────────────────────────────────────────
    # rayon en mètres = ratio × min(L, H)
    radius_abs = r * pd.concat([L, H], axis=1).min(axis=1)
    out["radius_abs"] = radius_abs

    # Rapport diamètre du trou / hauteur de plaque (entrée Kt, notation Peterson d/W)
    d_over_W = (2.0 * radius_abs / (H + _EPS)).clip(0.0, 0.95)
    out["d_over_W"] = d_over_W

    # ── Concentration de contrainte théorique (formule de Peterson largeur finie) ──
    # Valide pour un trou centré ; pour les trous excentriques, la correction de ligament
    # ci-dessous capture l'amplification supplémentaire.
    # Kt = 3 quand d/W → 0 (plaque infinie) ; Kt → 1 sans trou.
    Kt_hole = (3.0 - 3.13 * d_over_W + 3.66 * d_over_W**2 - 1.53 * d_over_W**3).clip(1.0, 10.0)
    # Pour les plaques simples, pas de trou → Kt = 1.
    out["Kt_theory"] = np.where(out["has_hole"] == 1, Kt_hole, 1.0)

    # ── Contrainte de section nette ──────────────────────────────────────────
    # Section nette = fraction de la section transversale non obstruée par le trou.
    net_section_ratio = (1.0 - d_over_W).clip(1e-3, 1.0)
    out["net_section_ratio"] = net_section_ratio
    out["sigma_net"]         = sig / (net_section_ratio + _EPS)      # Pa

    # ── Déformation élastique et déplacement théorique ───────────────────────
    # epsilon = σ / E  (déformation d'ingénierie, sans dimension)
    # delta_theory = ε × L  (allongement axial théorique, mètres)
    # C'est le meilleur prédicteur unique de max_displacement_m (r=0.96 en log).
    out["epsilon"]       = sig / (E + _EPS)
    out["delta_theory"]  = out["epsilon"] * L                          # m
    out["biaxial_factor"] = 1.0 - nu**2                                # plane-stress

    # ── Distances de ligament (normalisées, pour le régime à trou mobile) ────
    # Les ligaments sont les largeurs de matière restantes autour du trou, exprimées
    # comme fractions de la dimension de la plaque (du centre du trou au bord moins le rayon).
    lig_left   = (cx       - r).clip(lower=0.0)
    lig_right  = (1.0 - cx - r).clip(lower=0.0)
    lig_bottom = (cy       - r).clip(lower=0.0)
    lig_top    = (1.0 - cy - r).clip(lower=0.0)

    out["lig_left"]   = lig_left
    out["lig_right"]  = lig_right
    out["lig_bottom"] = lig_bottom
    out["lig_top"]    = lig_top
    out["lig_min"]    = pd.concat([lig_left, lig_right, lig_bottom, lig_top], axis=1).min(axis=1)

    # Ratio de proximité du bord : grande valeur → trou très proche du bord → pic de contrainte
    out["edge_ratio"] = r / (out["lig_min"] + _EPS)

    # ── Excentricité (trou mobile uniquement) ────────────────────────────────
    # Distance du centre du trou au centre de la plaque, normalisée à [0, 0.5]
    out["eccentricity_x"] = (cx - 0.5).abs()
    out["eccentricity_y"] = (cy - 0.5).abs()
    out["eccentricity"]   = np.sqrt(out["eccentricity_x"]**2 + out["eccentricity_y"]**2)

    # ── Proxy de contrainte combinée ─────────────────────────────────────────
    # Kt × σ_net est une bonne approximation analytique du pic de von Mises.
    out["stress_amp_proxy"] = out["Kt_theory"] * out["sigma_net"]

    # ── Features en espace log ───────────────────────────────────────────────
    # Les modèles d'arbres divisent sur des seuils uniformes ; les transformations
    # en espace log compriment les plages multi-décades de E et σ en une distribution plus uniforme.
    out["logE"]           = np.log10(E.clip(lower=1e6))
    out["logS"]           = np.log10(sig.clip(lower=1e3))
    out["log_epsilon"]    = np.log10(out["epsilon"].clip(lower=_EPS))
    out["log_delta_th"]   = np.log10(out["delta_theory"].clip(lower=_EPS))
    out["log_sigma_net"]  = np.log10(out["sigma_net"].clip(lower=_EPS))
    out["log_Kt"]         = np.log10(out["Kt_theory"].clip(lower=1.0))
    out["log_lig_min"]    = np.log10(out["lig_min"].clip(lower=_EPS))
    out["log_edge_ratio"] = np.log10(out["edge_ratio"].clip(lower=_EPS))

    # ── Feature héritée conservée pour la compatibilité ascendante (= epsilon, renommage conservé) ──
    out["traction_over_E"] = out["epsilon"]   # σ/E = strain

    return out


def _engineer(df: pd.DataFrame, keep_ambiguous_as_without_hole: bool = False) -> pd.DataFrame:
    df = _drop_ambiguous_geometry(df, keep_ambiguous_as_without_hole=keep_ambiguous_as_without_hole)
    return engineer_features(df)


# ── Utilitaires de split ──────────────────────────────────────────────────────
def _stable_bucket(simulation_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{simulation_id}|{seed}".encode("utf-8")).hexdigest()
    value  = int(digest[:16], 16)
    return value / float(16**16 - 1)


def _split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    strategy: str = "hash",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
    if len(df) < 3:
        raise ValueError("Need at least 3 rows to build train/val/test splits")
    if strategy not in {"hash", "random"}:
        raise ValueError("strategy must be one of {'hash', 'random'}")
    if "simulation_id" not in df.columns:
        raise ValueError("Column 'simulation_id' is required for split assignment")

    work = df.copy().reset_index(drop=True)
    if strategy == "hash":
        split_key  = work["simulation_id"].astype(str).map(lambda x: _stable_bucket(x, seed))
        train_mask = split_key < train_ratio
        val_mask   = (split_key >= train_ratio) & (split_key < (train_ratio + val_ratio))
        test_mask  = split_key >= (train_ratio + val_ratio)
        train_df   = work.loc[train_mask].copy()
        val_df     = work.loc[val_mask].copy()
        test_df    = work.loc[test_mask].copy()
    else:
        shuffled = work.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n        = len(shuffled)
        n_train  = int(n * train_ratio)
        n_val    = int(n * val_ratio)
        train_df = shuffled.iloc[:n_train].copy()
        val_df   = shuffled.iloc[n_train : n_train + n_val].copy()
        test_df  = shuffled.iloc[n_train + n_val :].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Invalid split sizes with current dataset size. "
            "Increase data volume or adjust ratios to keep non-empty splits."
        )
    return train_df, val_df, test_df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Retourne les colonnes utilisables comme features ML (exclut les cibles, métadonnées, constantes de maillage)."""
    exclude = set(TARGET_COLS) | set(BASE_DROP_COLS) | set(MESH_COLS)
    return [c for c in df.columns if c not in exclude]


# ── Point d'entrée public ────────────────────────────────────────────────────
def build_features(
    input_dir: Path,
    out_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    split_strategy: str = "hash",
    features_out_dir: Path | None = None,
    keep_ambiguous_as_without_hole: bool = False,
) -> None:
    raw     = _read_raw(input_dir)
    dataset = _engineer(raw, keep_ambiguous_as_without_hole=keep_ambiguous_as_without_hole)
    train_df, val_df, test_df = _split(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        strategy=split_strategy,
    )
    feature_cols = _feature_columns(train_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    features_dir = features_out_dir or out_dir
    features_dir.mkdir(parents=True, exist_ok=True)

    train_path          = out_dir / "train.parquet"
    val_path            = out_dir / "val.parquet"
    test_path           = out_dir / "test.parquet"
    features_ds_path    = features_dir / "features.parquet"
    feature_cols_path   = features_dir / "feature_columns.txt"

    dataset.to_parquet(features_ds_path, index=False)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    feature_cols_path.write_text("\n".join(feature_cols), encoding="utf-8")

    print(f"Built features dataset from : {input_dir}")
    print(f"Total usable rows           : {len(dataset):,} -> {features_ds_path}")
    print(f"Train rows                  : {len(train_df):,} -> {train_path}")
    print(f"Val rows                    : {len(val_df):,} -> {val_path}")
    print(f"Test rows                   : {len(test_df):,} -> {test_path}")
    print(f"Feature columns ({len(feature_cols)})       : {feature_cols_path}")
    print(f"Split strategy              : {split_strategy} (seed={seed})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construit les splits de features physiques depuis les données parquet de simulation brutes."
    )
    parser.add_argument("--input", required=True, type=Path, help="Dossier d'entrée brut")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--features-out-dir", type=Path, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        choices=["hash", "random"],
        default="hash",
    )
    parser.add_argument(
        "--keep-ambiguous-as-without-hole",
        action="store_true",
        help=(
            "Conserver les lignes sans signal de géométrie et forcer geometry_type='without_hole' "
            "au lieu de les supprimer."
        ),
    )
    args = parser.parse_args()
    build_features(
        input_dir=args.input,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_strategy=args.split_strategy,
        features_out_dir=args.features_out_dir,
        keep_ambiguous_as_without_hole=args.keep_ambiguous_as_without_hole,
    )


if __name__ == "__main__":
    main()
