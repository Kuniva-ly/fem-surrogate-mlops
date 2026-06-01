"""Spark-based ingestion pipeline for the FEM Surrogate Data Warehouse.

Reads raw simulation sources (local filesystem or MinIO/S3A), applies ETL
transformations and writes the cleaned Data Warehouse back.

Usage
-----
    # Local mode
    python -m src.processing.spark_ingest --raw-dir data/raw --wh-dir data/processed

    # MinIO mode (Docker stack running)
    python -m src.processing.spark_ingest --use-minio
"""
import argparse
import hashlib
import os
import platform
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

COLONNES_CRITIQUES = [
    "max_displacement_m",
    "max_von_mises_pa",
    "length_m",
    "height_m",
    "young_modulus_pa",
    "poisson_ratio",
    "traction_pa",
]

# MinIO / S3A — reads from environment (.env loaded by Docker or shell)
_MINIO_ENDPOINT   = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
_MINIO_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID",      "minioadmin")
_MINIO_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY",  "minioadmin")
_BUCKET_RAW       = os.getenv("MINIO_BUCKET_RAW",       "raw-simulations")
_BUCKET_PROCESSED = os.getenv("MINIO_BUCKET_PROCESSED", "processed-simulations")

# Hadoop-AWS connector — must match the Hadoop version bundled with PySpark 3.5.x (3.3.4)
_S3A_PACKAGES = (
    "org.apache.hadoop:hadoop-aws:3.3.4,"
    "com.amazonaws:aws-java-sdk-bundle:1.12.262"
)
# Exclude transitive Hadoop JARs — already bundled inside PySpark
_S3A_EXCLUDES = (
    "org.apache.hadoop:hadoop-common,"
    "org.apache.hadoop:hadoop-client-api,"
    "org.apache.hadoop:hadoop-client-runtime,"
    "org.apache.hadoop:hadoop-yarn-server-web-proxy"
)


# MinIO bucket provisioning

def _ensure_minio_buckets() -> None:
    """Create required MinIO buckets if they don't exist (idempotent)."""
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client(
        "s3",
        endpoint_url=_MINIO_ENDPOINT,
        aws_access_key_id=_MINIO_ACCESS_KEY,
        aws_secret_access_key=_MINIO_SECRET_KEY,
    )
    for bucket in (_BUCKET_RAW, _BUCKET_PROCESSED):
        try:
            s3.head_bucket(Bucket=bucket)
            print(f"[minio] bucket '{bucket}' : OK")
        except ClientError:
            s3.create_bucket(Bucket=bucket)
            print(f"[minio] bucket '{bucket}' : créé")


# Session

def build_spark_session(
    app_name: str = "FEM-Surrogate-Ingest",
    use_minio: bool = False,
) -> SparkSession:
    """Create a local SparkSession with optional S3A/MinIO connector."""
    if platform.system() == "Windows":
        os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
        hadoop_bin = Path(os.environ["HADOOP_HOME"]) / "bin"
        os.environ["PATH"] = str(hadoop_bin) + os.pathsep + os.environ.get("PATH", "")

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.showConsoleProgress", "false")
    )

    if use_minio:
        builder = (
            builder
            .config("spark.jars.packages", _S3A_PACKAGES)
            .config("spark.jars.excludes", _S3A_EXCLUDES)
            .config("spark.hadoop.fs.s3a.endpoint",          _MINIO_ENDPOINT)
            .config("spark.hadoop.fs.s3a.access.key",        _MINIO_ACCESS_KEY)
            .config("spark.hadoop.fs.s3a.secret.key",        _MINIO_SECRET_KEY)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.impl",
                    "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                    "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        )

    return builder.getOrCreate()


# Extract

def extract(spark: SparkSession, raw_path: str) -> DataFrame:
    """Read all raw Parquet sources and tag each row with its source name.

    Works with both local paths and s3a://bucket/ MinIO paths.
    """
    source_regex = rf"(?:{_BUCKET_RAW}/|raw[/\\])([^/\\]+)[/\\]"

    sdf = (
        spark.read
        .option("mergeSchema", "true")
        .option("recursiveFileLookup", "true")
        .parquet(raw_path)
        .withColumn(
            "source",
            F.regexp_extract(F.input_file_name(), source_regex, 1),
        )
    )
    n_partitions = sdf.rdd.getNumPartitions()
    print(f"[extract] {n_partitions} partitions Spark — {sdf.count():,} lignes brutes")
    return sdf


# Transform

@udf(StringType())
def _pseudonymize(uid: str):
    """SHA-256 hash — pseudonymisation RGPD irréversible."""
    if uid is None:
        return None
    return hashlib.sha256(uid.encode()).hexdigest()


def transform(sdf: DataFrame) -> DataFrame:
    """Clean, type-cast and pseudonymise the raw Spark DataFrame."""
    n_before = sdf.count()
    sdf_clean = (
        sdf
        .withColumn("timestamp", F.to_timestamp("timestamp"))
        .dropDuplicates(["simulation_id"])
        .dropna(subset=COLONNES_CRITIQUES)
        .withColumn("simulation_id", _pseudonymize(F.col("simulation_id")))
    )
    n_after = sdf_clean.count()
    print(f"[transform] {n_before:,} → {n_after:,} lignes  ({n_before - n_after} supprimées)")
    return sdf_clean


# Load

def load(sdf: DataFrame, wh_path: str) -> None:
    """Write cleaned DataFrame to Data Warehouse (local or s3a://)."""
    if not wh_path.startswith("s3a://"):
        Path(wh_path).parent.mkdir(parents=True, exist_ok=True)
    sdf.write.mode("overwrite").parquet(wh_path)
    print(f"[load] Data Warehouse écrit : {wh_path}")


# Validate

def validate_warehouse(spark: SparkSession, wh_path: str) -> None:
    """Read the Data Warehouse and assert basic quality constraints."""
    wh = spark.read.parquet(wh_path)

    total     = wh.count()
    distincts = wh.select("simulation_id").distinct().count()
    doublons  = total - distincts

    null_counts = (
        wh.select([
            F.sum(F.col(c).isNull().cast("int")).alias(c)
            for c in COLONNES_CRITIQUES
        ])
        .collect()[0]
        .asDict()
    )
    total_nulls = sum(null_counts.values())

    print(f"[validate] {total:,} lignes × {len(wh.columns)} colonnes")
    print(f"[validate] doublons       : {doublons}")
    print(f"[validate] valeurs nulles : {total_nulls}")

    if doublons > 0:
        raise ValueError(f"Data Warehouse contient {doublons} doublon(s) sur simulation_id")
    if total_nulls > 0:
        raise ValueError(f"Data Warehouse contient des nulls sur colonnes critiques : {null_counts}")

    print("[validate] OK — Data Warehouse valide")
    print("\n[validate] Répartition par source :")
    wh.groupBy("source").count().orderBy("count", ascending=False).show(truncate=False)


# Pipeline

def run_pipeline(
    raw_dir: Path | None = None,
    wh_dir: Path | None = None,
    use_minio: bool = False,
) -> None:
    """Orchestrate Extract → Transform → Load → Validate.

    - use_minio=False : local  (raw_dir → wh_dir/warehouse.parquet)
    - use_minio=True  : MinIO  (s3a://raw-simulations → s3a://processed-simulations)
    """
    spark = build_spark_session(use_minio=use_minio)
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark {spark.version} — {spark.sparkContext.defaultParallelism} cœurs\n")

    if use_minio:
        _ensure_minio_buckets()
        raw_path = f"s3a://{_BUCKET_RAW}/"
        wh_path  = f"s3a://{_BUCKET_PROCESSED}/warehouse.parquet"
        print(f"[pipeline] Mode MinIO\n  raw : {raw_path}\n  wh  : {wh_path}\n")
    else:
        raw_path = str(raw_dir)
        wh_path  = str(wh_dir / "warehouse.parquet")
        print(f"[pipeline] Mode local\n  raw : {raw_path}\n  wh  : {wh_path}\n")

    try:
        sdf_raw   = extract(spark, raw_path)
        sdf_clean = transform(sdf_raw)
        load(sdf_clean, wh_path)
        validate_warehouse(spark, wh_path)
    finally:
        spark.stop()
        print("\nSession Spark fermée.")


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spark ETL pipeline — FEM Surrogate Data Warehouse (RNCP Bloc 1)"
    )
    parser.add_argument(
        "--use-minio",
        action="store_true",
        help=(
            "Lire depuis MinIO (s3a://raw-simulations) et écrire vers MinIO "
            "(s3a://processed-simulations). Nécessite le Docker stack actif."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="[local] Répertoire des sources raw partitionnées (défaut: data/raw)",
    )
    parser.add_argument(
        "--wh-dir",
        type=Path,
        default=Path("data/processed"),
        help="[local] Répertoire de sortie du Data Warehouse (défaut: data/processed)",
    )
    args = parser.parse_args()
    run_pipeline(
        raw_dir=args.raw_dir,
        wh_dir=args.wh_dir,
        use_minio=args.use_minio,
    )


if __name__ == "__main__":
    main()
