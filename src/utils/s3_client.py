"""Centralized MinIO / S3 client factory.

All S3 credentials are read exclusively from environment variables.
Import this module instead of instantiating boto3.client() inline.

Usage
-----
    from src.utils.s3_client import get_s3_client, BUCKET_FEATURES

    s3 = get_s3_client()
    s3.upload_file(local_path, BUCKET_FEATURES, key)
"""
from __future__ import annotations

import os

import boto3
from botocore.client import BaseClient

# Endpoint — defaults to localhost for local dev without Docker.
_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

# Bucket names — configurable via environment
BUCKET_RAW       = os.getenv("MINIO_BUCKET_RAW",       "raw-simulations")
BUCKET_PROCESSED = os.getenv("MINIO_BUCKET_PROCESSED", "processed-simulations")
BUCKET_FEATURES  = os.getenv("MINIO_BUCKET_FEATURES",  "features")
BUCKET_ARTIFACTS = os.getenv("MINIO_BUCKET_ARTIFACTS", "ml-artifacts")


def get_s3_client(endpoint: str | None = None) -> BaseClient:
    """Return a boto3 S3 client configured for MinIO.

    Credentials are read from AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
    environment variables. Both are required — set them in your .env file.

    Args:
        endpoint: Override the default endpoint URL (uses MLFLOW_S3_ENDPOINT_URL
                  env var if not provided).

    Raises:
        RuntimeError: If AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY are not set.
    """
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise RuntimeError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set. "
            "Copy .env.example to .env and fill in the MinIO credentials."
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint or _ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
