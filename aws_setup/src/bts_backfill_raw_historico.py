#!/usr/bin/env python3
"""Backfill SOLO de ZIP BTS en S3 landing/raw, sin tocar Bronze ni invocar Lambda/Glue.

Uso desde SageMaker (rol LabRole):
    python bts_backfill_raw_historico.py --start 2022-01 --end 2026-08
    python bts_backfill_raw_historico.py --start 2022-01 --end 2026-08 --check-only

El script es reanudable: omite objetos S3 existentes con tamaño positivo.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from boto3.s3.transfer import TransferConfig

BUCKET_DEFAULT = "bts-flights-analysis"
BASE_URL = "https://www.transtats.bts.gov/PREZIP/"
DATASET = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present"
CHUNK_SIZE = 1024 * 1024
TRANSFER = TransferConfig(multipart_threshold=8 * CHUNK_SIZE,
                          multipart_chunksize=8 * CHUNK_SIZE, use_threads=False)
log = logging.getLogger("bts-backfill")


def ym(value):
    try:
        dt = datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Usa YYYY-MM (ej. 2022-01)") from exc
    if dt.year < 1987:
        raise argparse.ArgumentTypeError("La serie BTS empieza en 1987")
    return (dt.year, dt.month)


def months(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1


def s3_head(s3, bucket, key):
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NotFound", "NoSuchKey"):
            return None
        raise


def download_and_validate(url, destination):
    request = urllib.request.Request(url, headers={
        "User-Agent": "BTS-Academic-Data-Ingestion/3.0"
    })
    hash_obj = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=90) as response, open(destination, "wb") as target:
        while True:
            data = response.read(CHUNK_SIZE)
            if not data:
                break
            target.write(data)
            hash_obj.update(data)
            total += len(data)
    if total == 0:
        raise ValueError("Descarga vacía")
    with zipfile.ZipFile(destination) as archive:
        members = [x for x in archive.infolist() if not x.is_dir() and x.filename.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Se esperaba un CSV dentro del ZIP; encontrados: {len(members)}")
        corrupted = archive.testzip()
        if corrupted is not None:
            raise ValueError(f"ZIP corrupto (CRC): {corrupted}")
    return total, hash_obj.hexdigest(), members[0].filename


def backfill_one(s3, bucket, year, month, attempts):
    filename = f"{DATASET}_{year}_{month}.zip"
    url = BASE_URL + filename
    key = f"landing/raw/year={year}/month={month:02d}/flights.zip"
    current = s3_head(s3, bucket, key)
    if current is not None and current.get("ContentLength", 0) > 0:
        log.info("SKIP    %04d-%02d  ya existe (%s bytes)", year, month, current["ContentLength"])
        return "SKIPPED"

    for attempt in range(1, attempts + 1):
        try:
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / filename
                log.info("GET     %04d-%02d  intento %d/%d", year, month, attempt, attempts)
                byte_count, digest, csv_name = download_and_validate(url, target)
                log.info("VALIDO  %04d-%02d  %.1f MB  SHA256=%s...",
                         year, month, byte_count / (1024 * 1024), digest[:12])
                # Nunca se modifica Bronze. Metadatos que reflejan origen real.
                s3.upload_file(str(target), bucket, key,
                               Config=TRANSFER,
                               ExtraArgs={
                                   "ContentType": "application/zip",
                                   "Metadata": {
                                       "source": "bts-prezip",
                                       "ingestion-method": "historical-backfill",
                                       "sha256": digest,
                                       "csv-name": os.path.basename(csv_name)[:300]
                                   }
                               })
                uploaded = s3_head(s3, bucket, key)
                if uploaded is None or uploaded.get("ContentLength") != byte_count:
                    raise RuntimeError(f"Verificación S3 de tamaño falló: {key}")
                log.info("SUBIDO  s3://%s/%s", bucket, key)
                return "UPLOADED"
        except (NoCredentialsError, PartialCredentialsError):
            log.error("Credenciales AWS no disponibles. Reinicia Learner Lab y vuelve a ejecutar.")
            raise
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("ExpiredToken", "InvalidAccessKeyId", "AccessDenied"):
                log.error("Error de credenciales/permisos: %s. Deteniendo backfill.", code)
                raise
            log.error("ERROR   %04d-%02d  %s", year, month, exc)
        except Exception as exc:
            log.error("ERROR   %04d-%02d  %s: %s", year, month, type(exc).__name__, exc)
        if attempt < attempts:
            time.sleep(min(30, 5 * (2 ** (attempt - 1))))
    return "FAILED"


def main():
    parser = argparse.ArgumentParser(description="Backfill histórico BTS Raw-only, reanudable")
    parser.add_argument("--start", type=ym, default=(2022, 1), metavar="YYYY-MM")
    parser.add_argument("--end", type=ym, default=(2026, 8), metavar="YYYY-MM")
    parser.add_argument("--bucket", default=BUCKET_DEFAULT)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--check-only", action="store_true", help="Solo comprobar la cobertura en S3")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar claves sin conectarse a AWS")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    if args.end < args.start:
        parser.error("--end debe ser posterior o igual a --start")
    periods = list(months(args.start, args.end))
    log.info("Cobertura: %s meses (%04d-%02d a %04d-%02d)", len(periods),
             *args.start, *args.end)

    if args.dry_run:
        for y, m in periods:
            print(f"s3://{args.bucket}/landing/raw/year={y}/month={m:02d}/flights.zip")
        return

    s3 = boto3.Session(region_name=args.region).client("s3")
    summary = {"UPLOADED": 0, "SKIPPED": 0, "FAILED": 0}
    if args.check_only:
        missing = []
        for y, m in periods:
            key = f"landing/raw/year={y}/month={m:02d}/flights.zip"
            existing = s3_head(s3, args.bucket, key)
            if existing is None or existing.get("ContentLength", 0) == 0:
                missing.append(f"{y}-{m:02d}")
        print(f"Presentes: {len(periods)-len(missing)}/{len(periods)}")
        print(f"Faltantes: {len(missing)}")
        if missing:
            print(", ".join(missing))
        return

    for y, m in periods:
        result = backfill_one(s3, args.bucket, y, m, args.attempts)
        summary[result] += 1
        log.info("AVANCE  %s", summary)
    print("\nRESUMEN:", json.dumps(summary, ensure_ascii=False))
    if summary["FAILED"]:
        raise SystemExit("Algunos meses fallaron. Reejecuta el mismo comando para completar faltantes.")
    print("Hecho. Se añadieron únicamente ZIP en landing/raw/. No se modificó Bronze ni Silver.")


if __name__ == "__main__":
    main()
