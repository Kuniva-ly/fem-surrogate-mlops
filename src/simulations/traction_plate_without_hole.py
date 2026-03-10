import argparse
import datetime as dt
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulations.traction_plate_with_hole import (
    DIMENSION_CATEGORIES,
    MATERIAL_CATEGORIES,
    SOLVER_NAME,
    _fenics_jit_options,
    _log_chunk_stats,
    _parse_range,
    _solver_version,
    _try_import_fenics,
    write_dataset_chunk,
)


def solve_plate_fenics_without_hole(
    length_m: float,
    height_m: float,
    young_modulus_pa: float,
    poisson_ratio: float,
    traction_pa: float,
    mesh_nx: int,
    mesh_ny: int,
) -> tuple[float, float]:
    fenics = _try_import_fenics()
    if fenics is None:
        raise RuntimeError("FEniCS backend unavailable (missing MPI/dolfinx dependencies).")

    MPI, PETSc, ufl, fem, mesh, LinearProblem = fenics

    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([length_m, height_m])],
        [mesh_nx, mesh_ny],
        cell_type=mesh.CellType.triangle,
    )

    gdim = domain.geometry.dim
    fdim = domain.topology.dim - 1
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], length_m)

    left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
    right_facets = mesh.locate_entities_boundary(domain, fdim, right_boundary)

    right_facets = np.array(right_facets, dtype=np.int32)
    right_facets.sort()

    left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    u0 = np.zeros(gdim, dtype=PETSc.ScalarType)
    bc = fem.dirichletbc(u0, left_dofs, V)

    mu = young_modulus_pa / (2.0 * (1.0 + poisson_ratio))
    lmbda = young_modulus_pa * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sig(u):
        return lmbda * ufl.nabla_div(u) * ufl.Identity(gdim) + 2.0 * mu * eps(u)

    facet_tag = mesh.meshtags(
        domain,
        fdim,
        right_facets,
        np.full(len(right_facets), 1, dtype=np.int32),
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

    t = fem.Constant(domain, PETSc.ScalarType((traction_pa, 0.0)))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(sig(u), eps(v)) * ufl.dx
    Lform = ufl.dot(t, v) * ds(1)

    problem = LinearProblem(
        a,
        Lform,
        bcs=[bc],
        petsc_options_prefix="elasticity_no_hole_",
        jit_options=_fenics_jit_options(),
    )
    uh = problem.solve()

    u_vals = uh.x.array.reshape((-1, gdim))
    disp_mag = np.sqrt(np.sum(u_vals**2, axis=1))
    max_disp = float(disp_mag.max())

    s = sig(uh)
    s_dev = s - (1.0 / 3.0) * ufl.tr(s) * ufl.Identity(gdim)
    von_mises = ufl.sqrt(3.0 / 2.0 * ufl.inner(s_dev, s_dev))

    Q = fem.functionspace(domain, ("DG", 0))
    von_expr = fem.Expression(von_mises, Q.element.interpolation_points)
    von_field = fem.Function(Q)
    von_field.interpolate(von_expr)
    max_vm = float(von_field.x.array.max())

    return max_disp, max_vm


def solve_plate_proxy_without_hole(
    length_m: float,
    height_m: float,
    young_modulus_pa: float,
    poisson_ratio: float,
    traction_pa: float,
    mesh_nx: int,
    mesh_ny: int,
) -> tuple[float, float]:
    area = max(height_m * 1.0, 1e-12)
    displacement = traction_pa * (length_m**2) / (young_modulus_pa * area)
    poisson_factor = 1.0 / max(1.0 - poisson_ratio**2, 1e-6)
    mesh_factor = 1.0 + 20.0 / max(mesh_nx + mesh_ny, 1)
    vm = traction_pa * poisson_factor * mesh_factor
    return float(displacement), float(vm)


def sample_parameters(rng: np.random.Generator, sampling_mode: str, custom_ranges: dict) -> dict:
    mesh_nx = int(custom_ranges["mesh_nx"])
    mesh_ny = int(custom_ranges["mesh_ny"])
    if sampling_mode == "continuous":
        return {
            "material_category": custom_ranges.get("material_category", "custom"),
            "dimension_category": custom_ranges.get("dimension_category", "custom"),
            "length_m": float(rng.uniform(*custom_ranges["length_m"])),
            "height_m": float(rng.uniform(*custom_ranges["height_m"])),
            "young_modulus_pa": float(rng.uniform(*custom_ranges["young_modulus_pa"])),
            "poisson_ratio": float(rng.uniform(*custom_ranges["poisson_ratio"])),
            "traction_pa": float(rng.uniform(*custom_ranges["traction_pa"])),
            "mesh_nx": mesh_nx,
            "mesh_ny": mesh_ny,
        }

    material_category = rng.choice(["steel", "aluminum", "titanium"], p=[0.5, 0.35, 0.15]).item()
    dimension_category = rng.choice(["small", "medium", "large"], p=[0.3, 0.5, 0.2]).item()

    mat = MATERIAL_CATEGORIES[material_category]
    dim = DIMENSION_CATEGORIES[dimension_category]

    return {
        "material_category": material_category,
        "dimension_category": dimension_category,
        "length_m": float(rng.uniform(*dim["length_m"])),
        "height_m": float(rng.uniform(*dim["height_m"])),
        "young_modulus_pa": float(rng.uniform(*mat["young_modulus_pa"])),
        "poisson_ratio": float(rng.uniform(*mat["poisson_ratio"])),
        "traction_pa": float(rng.uniform(*mat["traction_pa"])),
        "mesh_nx": mesh_nx,
        "mesh_ny": mesh_ny,
    }


def generate_batch(
    n: int,
    seed: int,
    out_root: Path,
    data_version: str = "sim_v1_without_hole",
    backend: str = "auto",
    output_format: str = "parquet",
    chunk_size: int = 5000,
    sampling_mode: str = "categorical",
    custom_ranges: dict | None = None,
) -> Path:
    rng = np.random.default_rng(seed)
    date_partition = dt.datetime.now(dt.timezone.utc).strftime("date=%Y-%m-%d")
    out_dir = out_root / date_partition
    out_dir.mkdir(parents=True, exist_ok=True)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if sampling_mode not in {"categorical", "continuous"}:
        raise ValueError("sampling_mode must be one of: categorical, continuous")
    if custom_ranges is None:
        custom_ranges = {}

    if backend == "auto":
        if _try_import_fenics() is None:
            raise RuntimeError(
                "FEniCS backend unavailable. Install/run dolfinx environment and use --backend fenics."
            )
        backend = "fenics"

    if backend not in {"fenics", "proxy"}:
        raise ValueError("backend must be one of: auto, fenics, proxy")
    solver_version = _solver_version(backend)

    generated = 0
    part_idx = 0
    while generated < n:
        current_chunk = min(chunk_size, n - generated)
        rows = []
        for _ in range(current_chunk):
            params = sample_parameters(rng, sampling_mode, custom_ranges)
            solver_params = {
                "length_m": params["length_m"],
                "height_m": params["height_m"],
                "young_modulus_pa": params["young_modulus_pa"],
                "poisson_ratio": params["poisson_ratio"],
                "traction_pa": params["traction_pa"],
                "mesh_nx": params["mesh_nx"],
                "mesh_ny": params["mesh_ny"],
            }

            if backend == "fenics":
                max_disp, max_vm = solve_plate_fenics_without_hole(**solver_params)
            else:
                max_disp, max_vm = solve_plate_proxy_without_hole(**solver_params)

            rows.append(
                {
                    "simulation_id": str(uuid.uuid4()),
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "max_displacement_m": max_disp,
                    "max_von_mises_pa": max_vm,
                    "solver_name": f"{SOLVER_NAME}_{backend}",
                    "solver_version": solver_version,
                    "data_version": data_version,
                    "geometry_type": "without_hole",
                    **params,
                }
            )

        df = pd.DataFrame(rows)
        _log_chunk_stats(df, part_idx)
        out_file = write_dataset_chunk(df, out_dir, part_idx, output_format=output_format)
        generated += len(df)
        part_idx += 1
        print(f"Wrote chunk {part_idx} to {out_file} ({generated}/{n})")

    return out_dir


def run_single_default_case(
    backend: str = "auto",
    mesh_nx: int = 120,
    mesh_ny: int = 24,
) -> None:
    params = {
        "length_m": 1.0,
        "height_m": 0.2,
        "young_modulus_pa": 210e9,
        "poisson_ratio": 0.3,
        "traction_pa": 1e6,
        "mesh_nx": mesh_nx,
        "mesh_ny": mesh_ny,
    }

    if backend == "auto":
        if _try_import_fenics() is None:
            raise RuntimeError(
                "FEniCS backend unavailable. Install/run dolfinx environment and use --backend fenics."
            )
        backend = "fenics"

    if backend == "fenics":
        max_disp, max_vm = solve_plate_fenics_without_hole(**params)
    else:
        max_disp, max_vm = solve_plate_proxy_without_hole(**params)

    print("Max displacement:", max_disp)
    print("Max von Mises stress:", max_vm)
    print("Backend:", backend)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plate traction FEM simulation without hole (single or batch).")
    parser.add_argument("--n", type=int, default=1, help="Number of simulations for batch mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, help="Output root path (enables batch parquet export)")
    parser.add_argument("--data-version", default="sim_v1_without_hole")
    parser.add_argument("--backend", choices=["auto", "fenics", "proxy"], default="fenics")
    parser.add_argument(
        "--output-format",
        choices=["parquet"],
        default=os.getenv("SIM_OUTPUT_FORMAT", "parquet"),
        help="Dataset output format (default from SIM_OUTPUT_FORMAT, fallback: parquet)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("SIM_CHUNK_SIZE", "5000")),
        help="Number of rows per parquet part file (default from SIM_CHUNK_SIZE, fallback: 5000)",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["categorical", "continuous"],
        default="continuous",
        help="continuous: free ranges for broad coverage, categorical: preset classes",
    )
    parser.add_argument("--material-category", default="custom")
    parser.add_argument("--dimension-category", default="custom")
    parser.add_argument("--length-range", default="0.5,2.2")
    parser.add_argument("--height-range", default="0.08,0.6")
    parser.add_argument("--young-range", default="65e9,230e9")
    parser.add_argument("--poisson-range", default="0.25,0.35")
    parser.add_argument("--traction-range", default="0.2e6,2.2e6")
    parser.add_argument("--mesh-nx", type=int, default=120, help="Fixed mesh divisions along x")
    parser.add_argument("--mesh-ny", type=int, default=24, help="Fixed mesh divisions along y")
    args = parser.parse_args()

    if args.mesh_nx < 8 or args.mesh_ny < 4:
        raise ValueError("--mesh-nx must be >= 8 and --mesh-ny must be >= 4")

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
    }

    if args.out is None:
        run_single_default_case(args.backend, args.mesh_nx, args.mesh_ny)
        return

    generate_batch(
        args.n,
        args.seed,
        args.out,
        args.data_version,
        args.backend,
        args.output_format,
        args.chunk_size,
        args.sampling_mode,
        custom_ranges,
    )


if __name__ == "__main__":
    main()
