import argparse
import os
from pathlib import Path


def _build_endpoint_url(endpoint: str, secure: bool) -> str:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    scheme = "https" if secure else "http"
    return f"{scheme}://{endpoint}"


def upload_folder_to_minio(
    local_path: Path,
    bucket: str,
    prefix: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    region: str = "us-east-1",
    secure: bool = False,
) -> int:
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'boto3'. Install it to upload data to MinIO.") from exc

    if not local_path.exists():
        raise FileNotFoundError(f"Local path not found: {local_path}")
    if not local_path.is_dir():
        raise NotADirectoryError(f"Expected a directory for --local-path, got: {local_path}")

    endpoint_url = _build_endpoint_url(endpoint, secure)
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as exc:
        raise RuntimeError(
            f"Bucket '{bucket}' does not exist or is inaccessible at {endpoint_url}."
        ) from exc

    files = sorted(local_path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {local_path}")

    uploaded = 0
    clean_prefix = prefix.strip("/")
    for file_path in files:
        rel = file_path.relative_to(local_path).as_posix()
        key = f"{clean_prefix}/{rel}" if clean_prefix else rel
        try:
            s3.upload_file(str(file_path), bucket, key)
        except Exception as exc:
            raise RuntimeError(f"Failed to upload '{file_path}' to s3://{bucket}/{key}") from exc
        uploaded += 1

    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload parquet files to MinIO/S3.")
    parser.add_argument("--local-path", required=True, type=Path, help="Local folder with parquet files")
    parser.add_argument("--bucket", default=os.getenv("MINIO_BUCKET_RAW", "raw-simulations"))
    parser.add_argument(
        "--prefix",
        default=os.getenv("MINIO_PREFIX", ""),
        help="Object key prefix in bucket (e.g. sim_v1/date=2026-02-10)",
    )
    parser.add_argument("--endpoint", default=os.getenv("MINIO_ENDPOINT", "localhost:9000"))
    parser.add_argument("--access-key", default=os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
    parser.add_argument("--secret-key", default=os.getenv("MINIO_SECRET_KEY", "minioadmin"))
    parser.add_argument("--region", default=os.getenv("MINIO_REGION", "us-east-1"))
    parser.add_argument("--secure", action="store_true", help="Use HTTPS when endpoint has no scheme")
    args = parser.parse_args()

    uploaded = upload_folder_to_minio(
        local_path=args.local_path,
        bucket=args.bucket,
        prefix=args.prefix,
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        region=args.region,
        secure=args.secure,
    )

    endpoint_url = _build_endpoint_url(args.endpoint, args.secure)
    clean_prefix = args.prefix.strip("/")
    target = f"s3://{args.bucket}/{clean_prefix}" if clean_prefix else f"s3://{args.bucket}"
    print(f"Uploaded {uploaded} parquet files to {target} via {endpoint_url}")


if __name__ == "__main__":
    main()
