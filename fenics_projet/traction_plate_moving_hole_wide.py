import argparse
from pathlib import Path
from fenics_projet.traction_plate_moving_hole import _parse_range, generate_batch
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wide-range dataset for plate with moving hole.")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/raw/sim_v2_moving_hole_wide"))
    parser.add_argument("--data-version", default="sim_v2_moving_hole_wide")
    parser.add_argument("--backend", choices=["auto", "fenics", "proxy"], default="fenics")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--material-category", default="custom")
    parser.add_argument("--dimension-category", default="custom")
    parser.add_argument("--length-range", default="0.2,3.2")
    parser.add_argument("--height-range", default="0.04,1.2")
    parser.add_argument("--young-range", default="5e9,230e9")
    parser.add_argument("--poisson-range", default="0.15,0.45")
    parser.add_argument("--traction-range", default="0.1e6,3.0e6")
    parser.add_argument("--mesh-nx", type=int, default=120)
    parser.add_argument("--mesh-ny", type=int, default=24)
    parser.add_argument("--hole-radius-range", default="0.02,0.30")
    parser.add_argument("--hole-cx-range", default="0.05,0.95")
    parser.add_argument("--hole-cy-range", default="0.05,0.95")
    args = parser.parse_args()
    custom_ranges = {
        "material_category": args.material_category,
        "dimension_category": args.dimension_category,
        "length_m": _parse_range(args.length_range, float),
        "height_m": _parse_range(args.height_range, float),
        "young_modulus_pa": _parse_range(args.young_range, float),
        "poisson_ratio": _parse_range(args.poisson_range, float),
        "traction_pa": _parse_range(args.traction_range, float),
        "mesh_nx": args.mesh_nx,
        "mesh_ny": args.mesh_ny,
        "hole_radius_ratio": _parse_range(args.hole_radius_range, float),
        "hole_cx_ratio": _parse_range(args.hole_cx_range, float),
        "hole_cy_ratio": _parse_range(args.hole_cy_range, float),
    }
    generate_batch(
        n=args.n,
        seed=args.seed,
        out_root=args.out,
        data_version=args.data_version,
        backend=args.backend,
        output_format="parquet",
        chunk_size=args.chunk_size,
        sampling_mode="continuous",
        custom_ranges=custom_ranges,
    )
if __name__ == "__main__":
    main()
