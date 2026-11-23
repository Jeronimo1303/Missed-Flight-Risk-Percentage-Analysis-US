
import os
import io
import csv
import json
import time
import hashlib
import zipfile
import logging
import tempfile
import urllib.request
import boto3
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pyarrow.compute as pc

from datetime import datetime, timezone
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig


# ==================================================
# Configuration
# ==================================================

BUCKET = os.environ["BUCKET_NAME"]

BASE_URL = "https://www.transtats.bts.gov/PREZIP/"

DATASET = (
    "On_Time_Reporting_Carrier_On_Time_"
    "Performance_1987_present"
)

SCHEMA_VERSION = 3

s3 = boto3.client("s3")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

transfer_config = TransferConfig(
    multipart_threshold=8 * 1024**2,
    multipart_chunksize=8 * 1024**2,
    use_threads=False
)


# ==================================================
# Compression configuration
# ==================================================

def select_compression():

    for codec in ["snappy", "zstd", "gzip", "lz4", "brotli"]:

        logger.info(
            "Codec %s available: %s",
            codec,
            pa.Codec.is_available(codec)
        )

    for codec in ["snappy", "gzip"]:

        if pa.Codec.is_available(codec):

            logger.info("Selected compression: %s", codec)

            return codec

    logger.warning(
        "No preferred compression codec available. "
        "Using uncompressed Parquet."
    )

    return "NONE"


COMPRESSION = select_compression()


# ==================================================
# Bronze schema
# ==================================================

BRONZE_SCHEMA = pa.schema([
    ("DepDel15", pa.float64()),
    ("FlightDate", pa.date32()),
    ("Year", pa.int16()),
    ("Reporting_Airline", pa.string()),
    ("Quarter", pa.int16()),
    ("Month", pa.int16()),
    ("DayofMonth", pa.int16()),
    ("DayOfWeek", pa.int16()),
    ("Origin", pa.string()),
    ("Dest", pa.string()),
    ("OriginState", pa.string()),
    ("DestState", pa.string()),
    ("CRSDepTime", pa.int32()),
    ("DepTimeBlk", pa.string()),
    ("CRSArrTime", pa.int32()),
    ("CRSElapsedTime", pa.float64()),
    ("Distance", pa.float64()),
    ("Flight_Number_Reporting_Airline", pa.int32()),
    ("Tail_Number", pa.string()),
    ("Cancelled", pa.float64()),
    ("CancellationCode", pa.string()),
    ("Diverted", pa.float64()),
    ("DepTime", pa.int32()),
    ("DepDelay", pa.float64()),
    ("DepDelayMinutes", pa.float64()),
    ("DepartureDelayGroups", pa.float64()),
    ("TaxiOut", pa.float64()),
    ("WheelsOff", pa.int32()),
    ("ArrDelay", pa.float64()),
    ("ArrDel15", pa.float64()),
    ("CarrierDelay", pa.float64()),
    ("WeatherDelay", pa.float64()),
    ("NASDelay", pa.float64()),
    ("SecurityDelay", pa.float64()),
    ("LateAircraftDelay", pa.float64())
])

SELECTED_COLUMNS = BRONZE_SCHEMA.names

# BTS may represent integer values as decimals,
# for example 1200.00 or 1.0.

CSV_COLUMN_TYPES = {
    field.name: (
        pa.float64()
        if pa.types.is_integer(field.type)
        else field.type
    )
    for field in BRONZE_SCHEMA
}


# ==================================================
# Utility functions
# ==================================================

def get_object_metadata(key):

    try:

        return s3.head_object(
            Bucket=BUCKET,
            Key=key
        )

    except ClientError as error:

        code = error.response["Error"]["Code"]

        if code in ("404", "NoSuchKey", "NotFound"):
            return None

        raise


def load_manifest(key):

    try:

        response = s3.get_object(
            Bucket=BUCKET,
            Key=key
        )

        with response["Body"] as body:
            return json.loads(body.read())

    except ClientError as error:

        code = error.response["Error"]["Code"]

        if code in ("404", "NoSuchKey", "NotFound"):
            return None

        raise


def download_from_bts(url, destination):

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "BTS-Academic-Data-Ingestion/3.0"
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=60
    ) as response:

        with open(destination, "wb") as output:

            while True:

                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                output.write(chunk)


def calculate_sha256(path):

    checksum = hashlib.sha256()

    with open(path, "rb") as source:

        while True:

            chunk = source.read(1024 * 1024)

            if not chunk:
                break

            checksum.update(chunk)

    return checksum.hexdigest()


def validate_partition(table, year, month):

    for column, expected in [
        ("Year", year),
        ("Month", month)
    ]:

        values = table[column]

        if values.null_count > 0:

            raise ValueError(
                f"Null values found in {column}"
            )

        limits = pc.min_max(values).as_py()

        if (
            limits["min"] != expected
            or limits["max"] != expected
        ):

            raise ValueError(
                f"Unexpected values in {column}: {limits}"
            )


# ==================================================
# Lambda handler
# ==================================================

def lambda_handler(event, context):

    start = time.monotonic()

    # --------------------------------------------------
    # 1. Validate input parameters
    # --------------------------------------------------

    year = int(event["year"])
    month = int(event["month"])

    if year < 1987 or not 1 <= month <= 12:
        raise ValueError("Invalid year or month")

    period = f"year={year}/month={month:02d}"

    filename = f"{DATASET}_{year}_{month}.zip"

    url = BASE_URL + filename

    raw_key = (
        f"landing/raw/{period}/flights.zip"
    )

    parquet_key = (
        f"bronze/parquet/{period}/flights.parquet"
    )

    manifest_key = (
        f"control/manifests/{period}/ingestion.json"
    )

    logger.info("Processing period: %s", period)

    # --------------------------------------------------
    # 2. Check previous ingestion
    # --------------------------------------------------

    previous_manifest = load_manifest(manifest_key)

    raw_metadata = get_object_metadata(raw_key)

    parquet_metadata = get_object_metadata(parquet_key)

    if (
        previous_manifest
        and previous_manifest.get("status") == "SUCCESS"
        and previous_manifest.get("schema_version") == SCHEMA_VERSION
        and previous_manifest.get("selected_columns") == SELECTED_COLUMNS
        and previous_manifest.get("compression") == COMPRESSION
        and raw_metadata is not None
        and parquet_metadata is not None
        and raw_metadata["ContentLength"]
            == previous_manifest.get("zip_bytes")
        and parquet_metadata["ContentLength"]
            == previous_manifest.get("parquet_bytes")
    ):

        logger.info(
            "Period already ingested with current schema"
        )

        return {
            "status": "SKIPPED",
            "year": year,
            "month": month,
            "schema_version": SCHEMA_VERSION,
            "parquet_s3_uri": f"s3://{BUCKET}/{parquet_key}"
        }

    # --------------------------------------------------
    # 3. Create temporary workspace
    # --------------------------------------------------

    with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:

        zip_path = os.path.join(
            tmpdir,
            "flights.zip"
        )

        parquet_path = os.path.join(
            tmpdir,
            "flights.parquet"
        )

        # --------------------------------------------------
        # 4. Obtain original ZIP
        # --------------------------------------------------

        if raw_metadata is not None:

            logger.info("Reusing original ZIP from S3")

            s3.download_file(
                BUCKET,
                raw_key,
                zip_path,
                Config=transfer_config
            )

            download_source = "S3"

        else:

            logger.info("Downloading from BTS: %s", url)

            download_from_bts(url, zip_path)

            download_source = "BTS"

        zip_size = os.path.getsize(zip_path)

        zip_checksum = calculate_sha256(zip_path)

        logger.info(
            "ZIP size: %.2f MB",
            zip_size / 1024**2
        )

        # --------------------------------------------------
        # 5. Validate ZIP file
        # --------------------------------------------------

        with zipfile.ZipFile(zip_path, "r") as archive:

            invalid_file = archive.testzip()

            if invalid_file is not None:

                raise ValueError(
                    f"Corrupted ZIP member: {invalid_file}"
                )

            csv_files = [
                item
                for item in archive.infolist()
                if not item.is_dir()
                and item.filename.lower().endswith(".csv")
            ]

            if len(csv_files) != 1:

                raise ValueError(
                    f"Expected one CSV, found {len(csv_files)}"
                )

            csv_info = csv_files[0]

            original_csv_size = csv_info.file_size

            # --------------------------------------------------
            # 6. Validate original CSV columns
            # --------------------------------------------------

            with archive.open(csv_info) as source:

                with io.TextIOWrapper(
                    source,
                    encoding="utf-8-sig",
                    newline=""
                ) as text:

                    reader = csv.reader(text)

                    original_columns = next(reader, [])

            missing_columns = (
                set(SELECTED_COLUMNS)
                - set(original_columns)
            )

            if missing_columns:

                raise ValueError(
                    f"Missing columns: {sorted(missing_columns)}"
                )

            for column in SELECTED_COLUMNS:

                if original_columns.count(column) != 1:

                    raise ValueError(
                        f"Invalid duplicate column: {column}"
                    )

            logger.info(
                "Selecting %s of %s original columns",
                len(SELECTED_COLUMNS),
                len(original_columns)
            )

            # --------------------------------------------------
            # 7. Configure incremental CSV reading
            # --------------------------------------------------

            read_options = pacsv.ReadOptions(
                block_size=8 * 1024**2,
                encoding="utf8",
                use_threads=False
            )

            convert_options = pacsv.ConvertOptions(
                include_columns=SELECTED_COLUMNS,
                column_types=CSV_COLUMN_TYPES,
                include_missing_columns=False,
                strings_can_be_null=True,
                null_values=[
                    "",
                    "NaN",
                    "nan",
                    "NULL",
                    "null"
                ]
            )

            row_count = 0
            batch_count = 0

            logger.info(
                "Starting CSV to Parquet conversion"
            )

            logger.info(
                "Compression: %s",
                COMPRESSION
            )

            # --------------------------------------------------
            # 8. Convert CSV directly to Parquet
            # --------------------------------------------------

            with archive.open(csv_info) as source:

                with pacsv.open_csv(
                    source,
                    read_options=read_options,
                    convert_options=convert_options
                ) as reader:

                    with pq.ParquetWriter(
                        parquet_path,
                        schema=BRONZE_SCHEMA,
                        compression=COMPRESSION,
                        use_dictionary=True,
                        write_statistics=True
                    ) as writer:

                        for batch in reader:

                            if batch.num_rows == 0:
                                continue

                            table = pa.Table.from_batches(
                                [batch]
                            )

                            # BTS may change the order of columns in the ZIP.
                            # PyArrow Table.cast requires matching names AND order.
                            table = table.select(SELECTED_COLUMNS)
                            table = table.cast(
                                BRONZE_SCHEMA,
                                safe=True
                            )

                            validate_partition(
                                table,
                                year,
                                month
                            )

                            writer.write_table(table)

                            row_count += table.num_rows
                            batch_count += 1

                            logger.info(
                                "Batch %s | Total rows: %s",
                                batch_count,
                                row_count
                            )

            if row_count == 0:

                raise ValueError("Parquet contains no rows")

        # --------------------------------------------------
        # 9. Verify generated Parquet
        # --------------------------------------------------

        parquet_file = pq.ParquetFile(parquet_path)

        try:

            if parquet_file.metadata.num_rows != row_count:

                raise ValueError("Parquet row count mismatch")

            if not parquet_file.schema_arrow.equals(
                BRONZE_SCHEMA
            ):

                raise ValueError("Parquet schema mismatch")

            row_groups = parquet_file.metadata.num_row_groups

        finally:

            parquet_file.close()

        # Compare against previous ingestion when
        # the original ZIP has not changed.

        if (
            previous_manifest
            and previous_manifest.get("zip_sha256") == zip_checksum
            and previous_manifest.get("row_count") is not None
            and previous_manifest["row_count"] != row_count
        ):

            raise ValueError(
                "Row count differs from previous ingestion"
            )

        parquet_size = os.path.getsize(parquet_path)

        reduction_pct = round(
            (
                1 - parquet_size / original_csv_size
            ) * 100,
            2
        )

        logger.info("Total rows: %s", row_count)

        logger.info(
            "Original CSV: %.2f MB",
            original_csv_size / 1024**2
        )

        logger.info(
            "Parquet: %.2f MB",
            parquet_size / 1024**2
        )

        logger.info(
            "Storage reduction: %.2f%%",
            reduction_pct
        )

        # --------------------------------------------------
        # 10. Preserve original ZIP
        # --------------------------------------------------

        if download_source == "BTS":

            s3.upload_file(
                zip_path,
                BUCKET,
                raw_key,
                ExtraArgs={
                    "ContentType": "application/zip"
                },
                Config=transfer_config
            )

            logger.info("Original ZIP uploaded")

        # --------------------------------------------------
        # 11. Upload directly to Bronze
        # --------------------------------------------------

        s3.upload_file(
            parquet_path,
            BUCKET,
            parquet_key,
            ExtraArgs={
                "ContentType": "application/vnd.apache.parquet"
            },
            Config=transfer_config
        )

        logger.info(
            "Parquet uploaded: s3://%s/%s",
            BUCKET,
            parquet_key
        )

        # --------------------------------------------------
        # 12. Validate uploaded objects
        # --------------------------------------------------

        raw_object = s3.head_object(
            Bucket=BUCKET,
            Key=raw_key
        )

        parquet_object = s3.head_object(
            Bucket=BUCKET,
            Key=parquet_key
        )

        if raw_object["ContentLength"] != zip_size:

            raise ValueError("ZIP upload size mismatch")

        if parquet_object["ContentLength"] != parquet_size:

            raise ValueError("Parquet upload size mismatch")

        # --------------------------------------------------
        # 13. Generate ingestion manifest
        # --------------------------------------------------

        manifest = {
            "status": "SUCCESS",
            "schema_version": SCHEMA_VERSION,
            "source": "BTS",
            "download_source": download_source,
            "dataset": DATASET,
            "year": year,
            "month": month,
            "source_url": url,
            "original_filename": filename,
            "csv_filename": csv_info.filename,
            "row_count": row_count,
            "original_column_count": len(original_columns),
            "column_count": len(SELECTED_COLUMNS),
            "selected_columns": SELECTED_COLUMNS,
            "column_types": {
                field.name: str(field.type)
                for field in BRONZE_SCHEMA
            },
            "format": "parquet",
            "compression": COMPRESSION,
            "row_groups": row_groups,
            "batch_count": batch_count,
            "zip_bytes": zip_size,
            "original_csv_bytes": original_csv_size,
            "parquet_bytes": parquet_size,
            "parquet_reduction_pct": reduction_pct,
            "zip_sha256": zip_checksum,
            "raw_s3_uri": f"s3://{BUCKET}/{raw_key}",
            "parquet_s3_uri": f"s3://{BUCKET}/{parquet_key}",
            "ingested_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "duration_seconds": round(
                time.monotonic() - start,
                2
            )
        }

        s3.put_object(
            Bucket=BUCKET,
            Key=manifest_key,
            Body=json.dumps(
                manifest,
                indent=2
            ).encode("utf-8"),
            ContentType="application/json"
        )

        logger.info(
            "Ingestion completed: %s",
            json.dumps(manifest)
        )

        return manifest
