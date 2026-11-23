"""Glue (PySpark) — bronze -> silver del lakehouse BTS.

Replica a escala lo que hace el notebook `Ingesta_y_procesamiento_de_datos.ipynb`:
  * Lee el bronze tal como está:  bronze/parquet/year=YYYY/month=MM/flights.parquet
  * Valida (meses completos, columnas requeridas, volumen mínimo por mes).
  * Conserva target + features + identificadores (19 cols) y elimina las 16 columnas de fuga.
  * Filtra filas: DepDel15 no nulo y CRSElapsedTime > 0.
  * Escribe Parquet ZSTD particionado como el notebook: silver/flights/Year=YYYY/Month=M/

Es idempotente: solo procesa meses que están en bronze y faltan en silver (--FORCE true reprocesa todo).
Todo ocurre con DataFrames en el clúster; al driver solo llegan conteos agregados.
Cada corrida guarda métricas en s3://<bucket>/_metrics/ (evidencia de escalabilidad).

Parámetros del job (solo --BUCKET es obligatorio):
  --BUCKET --BRONZE_PREFIX (bronze/parquet/) --SILVER_PREFIX (silver/flights/) --FORCE (false)
  --EXPECTED_START (2022-01) --EXPECTED_END (2026-08) --MIN_ROWS_PER_MONTH (300000)
"""
import json
import re
import sys
import time
from datetime import datetime, timezone
from functools import reduce

import boto3
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F

# ---- Parámetros (los opcionales tienen valor por defecto) ------------------------------------
OPTIONAL = {
    "BRONZE_PREFIX": "bronze/parquet/",
    "SILVER_PREFIX": "silver/flights/",
    "FORCE": "false",
    "EXPECTED_START": "2022-01",
    "EXPECTED_END": "2026-08",
    "MIN_ROWS_PER_MONTH": "300000",
}
present = [k for k in OPTIONAL if f"--{k}" in sys.argv]
args = getResolvedOptions(sys.argv, ["JOB_NAME", "BUCKET"] + present)
for k, v in OPTIONAL.items():
    args.setdefault(k, v)


def _argv_value(name, default="n/a"):
    """Lee --NAME valor de sys.argv. JOB_RUN_ID no se pide a getResolvedOptions: Glue 6.0 ya lo
    registra internamente y pedirlo de nuevo provoca 'conflicting option string'."""
    flag = f"--{name}"
    if flag in sys.argv and sys.argv.index(flag) + 1 < len(sys.argv):
        return sys.argv[sys.argv.index(flag) + 1]
    return default


JOB_RUN_ID = _argv_value("JOB_RUN_ID")

BUCKET = args["BUCKET"]
BRONZE = args["BRONZE_PREFIX"]
SILVER = args["SILVER_PREFIX"]
FORCE = args["FORCE"].lower() == "true"
MIN_ROWS = int(args["MIN_ROWS_PER_MONTH"])

sc = SparkContext.getOrCreate()
glue_ctx = GlueContext(sc)
spark = glue_ctx.spark_session
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")  # solo reescribe particiones tocadas
job = Job(glue_ctx)
job.init(args["JOB_NAME"], args)
s3 = boto3.client("s3")

# ---- Columnas: misma auditoría de fuga del notebook de EDA -----------------------------------
TARGET = "DepDel15"
KEEP_COLS = [
    "DepDel15", "FlightDate", "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
    "Reporting_Airline", "Origin", "Dest", "OriginState", "DestState",
    "CRSDepTime", "DepTimeBlk", "CRSArrTime", "CRSElapsedTime", "Distance",
    "Flight_Number_Reporting_Airline", "Tail_Number",
]
LEAK_COLS = [
    "Cancelled", "CancellationCode", "Diverted", "DepTime", "DepDelay", "DepDelayMinutes",
    "DepartureDelayGroups", "TaxiOut", "WheelsOff", "ArrDelay", "ArrDel15",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]
STRING_COLS = {"Reporting_Airline", "Origin", "Dest", "OriginState", "DestState",
               "DepTimeBlk", "Tail_Number"}
SILVER_CASTS = {  # mismos tipos que el notebook
    TARGET: "tinyint", "Year": "smallint", "Quarter": "tinyint", "Month": "tinyint",
    "DayofMonth": "tinyint", "DayOfWeek": "tinyint", "CRSDepTime": "smallint",
    "CRSArrTime": "smallint", "Flight_Number_Reporting_Airline": "int",
}


# ---- Utilidades S3 ---------------------------------------------------------------------------
def list_prefixes(prefix):
    out, paginator = [], s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
        out += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
    return out


def partitions(root, year_key, month_key):
    """{(año, mes): prefijo} para carpetas <root><year_key>=YYYY/<month_key>=MM/."""
    found = {}
    for ypref in list_prefixes(root):
        ym = re.search(rf"{year_key}=(\d+)/$", ypref)
        if not ym:
            continue
        for mpref in list_prefixes(ypref):
            mm = re.search(rf"{month_key}=(\d+)/$", mpref)
            if mm:
                found[(int(ym.group(1)), int(mm.group(1)))] = mpref
    return found


def month_range(start, end):
    (sy, sm), (ey, em) = (map(int, start.split("-")), map(int, end.split("-")))
    out, y, m = [], sy, sm
    while (y, m) <= (ey, em):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def write_metrics(payload):
    payload.update({
        "stage": "bronze_to_silver",
        "job_run_id": JOB_RUN_ID,
        "executors": sc.getConf().get("spark.executor.instances", "n/a"),
        "executors_max_dynamic": sc.getConf().get("spark.dynamicAllocation.maxExecutors", "n/a"),
        "default_parallelism": sc.defaultParallelism,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    key = f"_metrics/bronze_to_silver/{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(payload, indent=2).encode())
    print("MÉTRICAS", json.dumps(payload))


# ---- Lógica ----------------------------------------------------------------------------------
def main():
    t0 = time.time()

    # 1) Descubrir y validar la cobertura del bronze
    bronze = partitions(BRONZE, "year", "month")
    expected = month_range(args["EXPECTED_START"], args["EXPECTED_END"])
    missing = [f"{y}-{m:02d}" for y, m in expected if (y, m) not in bronze]
    if missing:
        raise ValueError(f"Faltan meses en bronze ({len(missing)}): {missing}")

    done = set() if FORCE else set(partitions(SILVER, "Year", "Month"))
    pending = sorted(k for k in bronze if k not in done)
    if not pending:
        print("Silver ya está al día; nada que procesar")
        return write_metrics({"months_processed": [], "rows_in": 0, "rows_out": 0,
                              "seconds": round(time.time() - t0, 1)})

    # 2) Leer mes a mes (los esquemas pueden variar entre archivos), validar y tipar.
    #    Se lee la carpeta del mes: así Spark NO infiere year/month de la ruta y no choca con las
    #    columnas Year/Month que ya vienen dentro del Parquet.
    frames, rows_in_total, bad = [], 0, []
    for y, m in pending:
        df = spark.read.parquet(f"s3://{BUCKET}/{bronze[(y, m)]}")
        lacking = set(KEEP_COLS) - set(df.columns)
        if lacking:
            raise ValueError(f"{y}-{m:02d}: faltan columnas {sorted(lacking)}")
        n = df.count()  # conteo de metadatos del Parquet, barato
        if n < MIN_ROWS:
            bad.append((f"{y}-{m:02d}", n))
        rows_in_total += n

        # Filtrar ANTES de castear: en Spark, NaN -> tinyint daría 0 en silencio
        valid = df.filter(F.col(TARGET).cast("double").isin(0.0, 1.0) &
                          (F.col("CRSElapsedTime").cast("double") > 0))
        cols = []
        for c in KEEP_COLS:
            col = F.col(c)
            if c == "FlightDate":
                cols.append(F.to_date(col).alias(c))
            elif c in STRING_COLS:
                cols.append(col.cast("string").alias(c))
            elif c in SILVER_CASTS:
                cols.append(col.cast(SILVER_CASTS[c]).alias(c))
            else:
                cols.append(col.cast("double").alias(c))
        frames.append(valid.select(*cols))
    if bad:
        raise ValueError(f"Meses con menos de {MIN_ROWS:,} filas: {bad}")

    # 3) Escribir silver con el mismo layout que el notebook: Year=YYYY/Month=M/
    silver = reduce(lambda a, b: a.unionByName(b), frames).repartition("Year", "Month")
    silver.write.mode("overwrite").partitionBy("Year", "Month").option("compression", "zstd") \
        .parquet(f"s3://{BUCKET}/{SILVER}")

    # 4) Verificar lo escrito (equivale a los assert del notebook)
    paths = [f"s3://{BUCKET}/{SILVER}Year={y}/Month={m}/" for y, m in pending]
    out = spark.read.option("basePath", f"s3://{BUCKET}/{SILVER}").parquet(*paths)
    leaked = set(out.columns) & set(LEAK_COLS)
    if leaked:
        raise ValueError(f"Columnas con fuga en silver: {sorted(leaked)}")
    stats = out.agg(
        F.count(F.lit(1)).alias("n"),
        F.sum(F.col(TARGET).isNull().cast("int")).alias("null_target"),
        F.min("CRSElapsedTime").alias("min_elapsed"),
        F.avg(TARGET).alias("delay_rate"),
    ).first()
    if stats["n"] == 0 or stats["null_target"] > 0 or stats["min_elapsed"] <= 0:
        raise ValueError(f"Silver inválido: {stats.asDict()}")

    write_metrics({
        "months_processed": [f"{y}-{m:02d}" for y, m in pending],
        "rows_in": rows_in_total,
        "rows_out": stats["n"],
        "rows_dropped": rows_in_total - stats["n"],
        "delay_rate": round(float(stats["delay_rate"]), 4),
        "seconds": round(time.time() - t0, 1),
    })


main()
job.commit()