import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path


GEOMETRY_CONFIG = {
    "with_hole": {
        "script": "fenics_projet/traction_plate_with_hole.py",
        "default_base": Path("data/raw/sim_v1"),
        "data_version": "sim_v1",
    },
    "without_hole": {
        "script": "fenics_projet/traction_plate_without_hole.py",
        "default_base": Path("data/raw/sim_v1_without_hole"),
        "data_version": "sim_v1_without_hole",
    },
    "with_hole_moving": {
        "script": "fenics_projet/traction_plate_moving_hole.py",
        "default_base": Path("data/raw/sim_v2_moving_hole"),
        "data_version": "sim_v2_moving_hole",
    },
}


def _resolve_generator_runner(mode: str, backend: str) -> str:
    if mode in {"local", "docker"}:
        return mode
    return "docker" if backend in {"fenics", "auto"} else "local"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily batch: generate -> validate -> build features -> optional MinIO upload."
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--geometry-type",
        choices=["with_hole", "without_hole", "with_hole_moving"],
        default="with_hole",
    )
    parser.add_argument("--base", type=Path, default=None, help="Raw output root (defaults by geometry type)")
    parser.add_argument("--backend", choices=["auto", "fenics", "proxy"], default="fenics")
    parser.add_argument(
        "--generator-runner",
        choices=["auto", "local", "docker"],
        default="auto",
        help="How to run generation script. 'auto' uses docker for fenics backend, local otherwise.",
    )
    parser.add_argument("--fenics-service", default="fenics", help="Docker Compose service name for FEniCS")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--upload-minio", action="store_true")
    parser.add_argument("--bucket", default=os.getenv("MINIO_BUCKET_RAW", "raw-simulations"))
    parser.add_argument("--processed-base", type=Path, default=Path("data/processed"))
    parser.add_argument("--features-base", type=Path, default=Path("data/features"))
    parser.add_argument("--processed-bucket", default=os.getenv("MINIO_BUCKET_PROCESSED", "processed-simulations"))
    parser.add_argument("--features-bucket", default=os.getenv("MINIO_BUCKET_FEATURES", "features"))
    parser.add_argument("--feature-group", default="stress_model")
    parser.add_argument("--feature-version", default="v1")
    args = parser.parse_args()

    config = GEOMETRY_CONFIG[args.geometry_type]
    base = args.base if args.base is not None else config["default_base"]
    runner = _resolve_generator_runner(args.generator_runner, args.backend)

    date_partition = dt.datetime.now(dt.timezone.utc).strftime("date=%Y-%m-%d")
    out_dir = base / date_partition
    processed_out_dir = args.processed_base / date_partition
    geometry_prefix = args.geometry_type
    features_out_dir = args.features_base / args.feature_group / args.feature_version / geometry_prefix / date_partition

    gen_cmd = [
        config["script"],
        "--n",
        str(args.n),
        "--seed",
        str(args.seed),
        "--out",
        str(base),
        "--data-version",
        config["data_version"],
        "--backend",
        args.backend,
        "--chunk-size",
        str(args.chunk_size),
    ]
    if runner == "docker":
        gen_cmd = ["docker", "compose", "exec", args.fenics_service, "python", *gen_cmd]
    else:
        gen_cmd = ["python", *gen_cmd]
    subprocess.run(gen_cmd, check=True)

    val_cmd = ["python", "-m", "src.processing.validate", "--input", str(out_dir)]
    subprocess.run(val_cmd, check=True)

    build_features_cmd = [
        "python",
        "-m",
        "src.processing.build_features",
        "--input",
        str(out_dir),
        "--out-dir",
        str(processed_out_dir),
        "--features-out-dir",
        str(features_out_dir),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(build_features_cmd, check=True)

    if args.upload_minio:
        raw_prefix = f"{geometry_prefix}/{base.name}/{date_partition}"
        raw_upload_cmd = [
            "python",
            "-m",
            "src.ingestion.upload_to_minio",
            "--local-path",
            str(out_dir),
            "--bucket",
            args.bucket,
            "--prefix",
            raw_prefix,
        ]
        subprocess.run(raw_upload_cmd, check=True)
        print(f"Uploaded raw batch to s3://{args.bucket}/{raw_prefix}")

        processed_prefix = f"{geometry_prefix}/{date_partition}"
        processed_upload_cmd = [
            "python",
            "-m",
            "src.ingestion.upload_to_minio",
            "--local-path",
            str(processed_out_dir),
            "--bucket",
            args.processed_bucket,
            "--prefix",
            processed_prefix,
        ]
        subprocess.run(processed_upload_cmd, check=True)
        print(f"Uploaded processed splits to s3://{args.processed_bucket}/{processed_prefix}")

        features_prefix = f"{args.feature_group}/{args.feature_version}/{geometry_prefix}/{date_partition}"
        features_upload_cmd = [
            "python",
            "-m",
            "src.ingestion.upload_to_minio",
            "--local-path",
            str(features_out_dir),
            "--bucket",
            args.features_bucket,
            "--prefix",
            features_prefix,
        ]
        subprocess.run(features_upload_cmd, check=True)
        print(f"Uploaded feature-store artifacts to s3://{args.features_bucket}/{features_prefix}")

    print(f"Batch completed. Raw output: {out_dir}")
    print(f"Batch completed. Processed output: {processed_out_dir}")
    print(f"Batch completed. Features output: {features_out_dir}")


if __name__ == "__main__":
    main()
