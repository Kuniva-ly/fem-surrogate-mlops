"""Upload data/raw/ to the MinIO raw-simulations bucket.

Usage
-----
    python scripts/upload_raw_to_minio.py
    python scripts/upload_raw_to_minio.py --raw-dir data/raw --endpoint http://localhost:9000
"""
import argparse
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ENDPOINT    = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
ACCESS_KEY  = os.getenv("AWS_ACCESS_KEY_ID",      "minioadmin")
SECRET_KEY  = os.getenv("AWS_SECRET_ACCESS_KEY",  "minioadmin")
BUCKET_RAW  = os.getenv("MINIO_BUCKET_RAW",       "raw-simulations")


def upload_raw(raw_dir: Path, endpoint: str) -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )

    # Ensure bucket exists
    try:
        s3.head_bucket(Bucket=BUCKET_RAW)
    except ClientError:
        s3.create_bucket(Bucket=BUCKET_RAW)
        print(f"Bucket '{BUCKET_RAW}' créé.")

    files = list(raw_dir.rglob("*.parquet"))
    if not files:
        print(f"Aucun fichier Parquet trouvé dans {raw_dir}")
        return

    print(f"{len(files)} fichier(s) à uploader vers s3://{BUCKET_RAW}/\n")
    for f in files:
        key = f.relative_to(raw_dir).as_posix()
        s3.upload_file(str(f), BUCKET_RAW, key)
        print(f"  ✓ {key}")

    print(f"\nUpload terminé — {len(files)} fichier(s) dans s3://{BUCKET_RAW}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload data/raw/ vers MinIO")
    parser.add_argument("--raw-dir",  type=Path, default=Path("data/raw"))
    parser.add_argument("--endpoint", default=ENDPOINT)
    args = parser.parse_args()
    upload_raw(args.raw_dir, args.endpoint)


if __name__ == "__main__":
    main()
