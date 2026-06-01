"""Attend que MinIO soit accessible. Exit 0 si prêt, exit 1 après timeout."""
import os
import sys
import time

import boto3

ENDPOINT   = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID",      "minioadmin")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY",  "minioadmin")
MAX_RETRIES = 30
INTERVAL    = 2


def main() -> None:
    s3 = boto3.client("s3", endpoint_url=ENDPOINT,
                      aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    for i in range(MAX_RETRIES):
        try:
            s3.list_buckets()
            print("MinIO pret.")
            sys.exit(0)
        except Exception:
            print(f"  Attente MinIO... ({i + 1}/{MAX_RETRIES})")
            time.sleep(INTERVAL)

    print(f"MinIO non disponible apres {MAX_RETRIES * INTERVAL}s.")
    sys.exit(1)


if __name__ == "__main__":
    main()
