import argparse
import datetime as dt
import os
import platform
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOLVER_NAME = "dolfinx"

MATERIAL_CATEGORIES = {
    "steel": {
        "young_modulus_pa": (190e9, 230e9),
        "poisson_ratio": (0.27, 0.31),
        "traction_pa": (0.5e6, 2.2e6),
    },
    "aluminum": {
        "young_modulus_pa": (65e9, 78e9),
        "poisson_ratio": (0.31, 0.35),
        "traction_pa": (0.2e6, 1.2e6),
    },
    "titanium": {
        "young_modulus_pa": (105e9, 125e9),
        "poisson_ratio": (0.30, 0.35),
        "traction_pa": (0.4e6, 1.8e6),
    },
}

DIMENSION_CATEGORIES = {
    "small": {"length_m": (0.5, 0.9), "height_m": (0.08, 0.18), "mesh_nx": (60, 101), "mesh_ny": (12, 25)},
    "medium": {
        "length_m": (0.9, 1.4),
        "height_m": (0.15, 0.30),
        "mesh_nx": (90, 151),
        "mesh_ny": (18, 33),
    },
    "large": {
        "length_m": (1.4, 2.2),
        "height_m": (0.25, 0.60),
        "mesh_nx": (120, 201),
        "mesh_ny": (24, 49),
    },
}


def _fenics_jit_options() -> dict[str, Any]:
    # Isolate JIT cache per process to avoid stale lock/contention in shared caches.
    cache_root = Path(os.getenv("FENICS_JIT_CACHE_DIR", f"/tmp/fenics_jit_{os.getpid()}"))
    cache_root.mkdir(parents=True, exist_ok=True)
    timeout_s = int(os.getenv("FENICS_JIT_TIMEOUT", "300"))
    return {"cache_dir": str(cache_root), "timeout": timeout_s}


def _parse_range(value: str, cast=float) -> tuple:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid range '{value}'. Expected format: min,max")
    lo = cast(parts[0])
    hi = cast(parts[1])
    if lo >= hi:
        raise ValueError(f"Invalid range '{value}'. min must be < max")
    return lo, hi


def write_dataset_chunk(
    df: pd.DataFrame, out_dir: Path, part_idx: int, output_format: str = "parquet"
) -> Path:
    if output_format != "parquet":
        raise ValueError("Only parquet output is supported. Set output_format='parquet'.")

    parquet_path = out_dir / f"part-{part_idx:05d}.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except ImportError as exc:
        raise RuntimeError(
            "Parquet export requires a parquet engine. Install 'pyarrow' (recommended) "
            "or 'fastparquet'."
        ) from exc


def _try_import_fenics():
    try:
        from mpi4py import MPI
        from petsc4py import PETSc
        import ufl
        from dolfinx import fem, mesh
        from dolfinx.fem.petsc import LinearProblem
    except Exception:
        return None
    return MPI, PETSc, ufl, fem, mesh, LinearProblem


def _solver_version(backend: str) -> str:
    if backend != "fenics":
        return f"python={platform.python_version()}"
    fenics = _try_import_fenics()
    if fenics is None:
        return f"python={platform.python_version()}"
    _, PETSc, _, _, _, _ = fenics
    try:
        import dolfinx

        dolfinx_ver = getattr(dolfinx, "__version__", "unknown")
    except Exception:
        dolfinx_ver = "unknown"
    try:
        petsc_ver = ".".join(str(v) for v in PETSc.Sys.getVersion())
    except Exception:
        petsc_ver = "unknown"
    return f"dolfinx={dolfinx_ver};petsc={petsc_ver};python={platform.python_version()}"


def _log_chunk_stats(df: pd.DataFrame, part_idx: int) -> None:
    numeric_cols = [
        "length_m",
        "height_m",
        "young_modulus_pa",
        "poisson_ratio",
        "traction_pa",
        "mesh_nx",
        "mesh_ny",
        "hole_radius_ratio",
        "max_displacement_m",
        "max_von_mises_pa",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    if not available:
        return
    stats = df[available].agg(["min", "max", "mean", "std"]).to_dict()
    payload: dict[str, Any] = {}
    for col, values in stats.items():
        payload[col] = {k: float(v) if pd.notna(v) else None for k, v in values.items()}
    print(f"Chunk {part_idx + 1} stats: {payload}")


def solve_plate_fenics(
    length_m: float,
    height_m: float,
    young_modulus_pa: float,
    poisson_ratio: float,
    traction_pa: float,
    mesh_nx: int,
    mesh_ny: int,
    hole_radius_ratio: float = 0.15,
) -> tuple[float, float]:
    fenics = _try_import_fenics()
    if fenics is None:
        raise RuntimeError("FEniCS backend unavailable (missing MPI/dolfinx dependencies).")

    MPI, PETSc, ufl, fem, mesh, LinearProblem = fenics
    hole_radius_m = hole_radius_ratio * min(length_m, height_m)
    if hole_radius_m <= 0.0:
        raise ValueError("hole_radius_ratio must be > 0 to create a plate with a hole")
    if hole_radius_m >= 0.5 * min(length_m, height_m):
        raise ValueError("hole_radius_ratio too large: hole must stay inside the plate")

    try:
        import gmsh
        import dolfinx.io as dfx_io
    except Exception as exc:
        raise RuntimeError("gmsh + dolfinx.io are required to mesh a plate with hole") from exc

    model_to_mesh = None
    if hasattr(dfx_io, "gmshio"):
        model_to_mesh = dfx_io.gmshio.model_to_mesh
    elif hasattr(dfx_io, "gmsh"):
        model_to_mesh = dfx_io.gmsh.model_to_mesh
    if model_to_mesh is None:
        raise RuntimeError("Could not find model_to_mesh in dolfinx.io (expected gmshio or gmsh module)")

    char_len = min(length_m / max(mesh_nx, 1), height_m / max(mesh_ny, 1))
    gmsh.initialize()
    try:
        if MPI.COMM_WORLD.rank == 0:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("plate_with_hole")
            rect = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, length_m, height_m)
            cx = 0.5 * length_m
            cy = 0.5 * height_m
            hole = gmsh.model.occ.addDisk(cx, cy, 0.0, hole_radius_m, hole_radius_m)
            gmsh.model.occ.cut([(2, rect)], [(2, hole)], removeObject=True, removeTool=True)
            gmsh.model.occ.synchronize()
            # dolfinx.io.gmsh.model_to_mesh requires physical groups.
            surfaces = gmsh.model.getEntities(2)
            if not surfaces:
                raise RuntimeError("Gmsh failed to build 2D surface for plate-with-hole domain")
            gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], 1)
            gmsh.model.setPhysicalName(2, 1, "domain")
            boundary_curves = gmsh.model.getBoundary(surfaces, oriented=False, recursive=False)
            if boundary_curves:
                gmsh.model.addPhysicalGroup(1, [c[1] for c in boundary_curves], 2)
                gmsh.model.setPhysicalName(1, 2, "boundary")
            points = gmsh.model.getEntities(0)
            if points:
                gmsh.model.mesh.setSize(points, char_len)
            gmsh.model.mesh.generate(2)
        mesh_data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        # dolfinx versions differ: tuple(mesh, tags...) or MeshData with .mesh
        if hasattr(mesh_data, "mesh"):
            domain = mesh_data.mesh
        elif isinstance(mesh_data, (tuple, list)) and len(mesh_data) >= 1:
            domain = mesh_data[0]
        else:
            raise RuntimeError("Unsupported return type from dolfinx.io model_to_mesh")
    finally:
        gmsh.finalize()

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
        petsc_options_prefix="elasticity_",
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


def solve_plate_proxy(
    length_m: float,
    height_m: float,
    young_modulus_pa: float,
    poisson_ratio: float,
    traction_pa: float,
    mesh_nx: int,
    mesh_ny: int,
    hole_radius_ratio: float = 0.15,
) -> tuple[float, float]:
    radius = hole_radius_ratio * min(length_m, height_m)
    net_height = max(height_m - 2.0 * radius, 1e-12)
    area = max(net_height * 1.0, 1e-12)
    displacement = traction_pa * (length_m**2) / (young_modulus_pa * area)
    poisson_factor = 1.0 / max(1.0 - poisson_ratio**2, 1e-6)
    stress_concentration = 1.0 + 2.0 * min((2.0 * radius) / max(height_m, 1e-12), 0.95)
    mesh_factor = 1.0 + 20.0 / max(mesh_nx + mesh_ny, 1)
    vm = traction_pa * poisson_factor * mesh_factor * stress_concentration
    return float(displacement), float(vm)


def sample_parameters(rng: np.random.Generator, sampling_mode: str, custom_ranges: dict) -> dict:
    hole_lo, hole_hi = custom_ranges["hole_radius_ratio"]
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
            "hole_radius_ratio": float(rng.uniform(hole_lo, hole_hi)),
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
        "hole_radius_ratio": float(rng.uniform(hole_lo, hole_hi)),
    }


def generate_batch(
    n: int,
    seed: int,
    out_root: Path,
    data_version: str = "sim_v1",
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
                "hole_radius_ratio": params["hole_radius_ratio"],
            }
            if backend == "fenics":
                max_disp, max_vm = solve_plate_fenics(**solver_params)
            else:
                max_disp, max_vm = solve_plate_proxy(**solver_params)

            rows.append(
                {
                    "simulation_id": str(uuid.uuid4()),
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "max_displacement_m": max_disp,
                    "max_von_mises_pa": max_vm,
                    "solver_name": f"{SOLVER_NAME}_{backend}",
                    "solver_version": solver_version,
                    "data_version": data_version,
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
    hole_radius_ratio: float = 0.15,
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
        "hole_radius_ratio": hole_radius_ratio,
    }

    if backend == "auto":
        if _try_import_fenics() is None:
            raise RuntimeError(
                "FEniCS backend unavailable. Install/run dolfinx environment and use --backend fenics."
            )
        backend = "fenics"

    if backend == "fenics":
        max_disp, max_vm = solve_plate_fenics(**params)
    else:
        max_disp, max_vm = solve_plate_proxy(**params)
    print("Max displacement:", max_disp)
    print("Max von Mises stress:", max_vm)
    print("Backend:", backend)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plate traction FEM simulation (single or batch).")
    parser.add_argument("--n", type=int, default=1, help="Number of simulations for batch mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, help="Output root path (enables batch parquet export)")
    parser.add_argument("--data-version", default="sim_v1")
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
    parser.add_argument("--hole-radius-ratio", type=float, default=0.15, help="Fixed hole radius ratio in single mode")
    parser.add_argument(
        "--hole-radius-range",
        default="0.05,0.20",
        help="Hole radius ratio range in batch mode. Format: min,max and both in (0, 0.5).",
    )
    args = parser.parse_args()

    if not (0.0 < args.hole_radius_ratio < 0.5):
        raise ValueError("--hole-radius-ratio must be in (0, 0.5)")
    if args.mesh_nx < 8 or args.mesh_ny < 4:
        raise ValueError("--mesh-nx must be >= 8 and --mesh-ny must be >= 4")
    hole_radius_range = _parse_range(args.hole_radius_range, float)
    if not (0.0 < hole_radius_range[0] < 0.5 and 0.0 < hole_radius_range[1] < 0.5):
        raise ValueError("--hole-radius-range bounds must both be in (0, 0.5)")

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
        "hole_radius_ratio": hole_radius_range,
    }

    if args.out is None:
        run_single_default_case(args.backend, args.hole_radius_ratio, args.mesh_nx, args.mesh_ny)
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
