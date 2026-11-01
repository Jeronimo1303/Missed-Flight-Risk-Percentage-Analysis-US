"""
Convierte los CSV mensuales del BTS (On-Time Reporting Carrier) a parquet,
un archivo por mes, con estructura Hive:

    data/bronze/parquet/year=2026/month=01/flights.parquet

Por defecto conserva SOLO las 35 columnas del proyecto (lista COLUMNS, el mismo
orden que usaba src/year_aggregator.py) y NO filtra filas. Con --all-columns
conserva las ~109 columnas del CSV (conversión sin pérdida).

En ambos casos asigna tipos explícitos, para que todos los meses tengan el mismo
esquema, y descarta la columna vacía que genera la coma final de los CSV del BTS.

Uso (desde la raíz del repositorio):
    python src/utils/csv_to_parquet.py
    python src/utils/csv_to_parquet.py --force                 # reconvertir todo
    python src/utils/csv_to_parquet.py --expected-rows 32386436

Leer el resultado con poda de particiones (hive_types es necesario: sin él,
DuckDB infiere month="01" como texto):

    SELECT *
    FROM read_parquet('data/bronze/parquet/year=*/month=*/*.parquet',
                      hive_partitioning = true,
                      hive_types = {'year': 'SMALLINT', 'month': 'SMALLINT'})
    WHERE Year = 2025 AND Month BETWEEN 1 AND 6;
"""

from __future__ import annotations

import argparse
import csv as csvlib
import re
import sys
import time
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]

# Tipos explícitos para las columnas conocidas. Las columnas que no estén aquí
# (por ejemplo, si descargaste más campos del BTS) se infieren y se conservan.
COLUMN_TYPES: dict[str, str] = {
    "FlightDate": "DATE",
    "Year": "SMALLINT",
    "Quarter": "SMALLINT",
    "Month": "SMALLINT",
    "DayofMonth": "SMALLINT",
    "DayOfWeek": "SMALLINT",
    "Reporting_Airline": "VARCHAR",
    "Tail_Number": "VARCHAR",
    "Flight_Number_Reporting_Airline": "INTEGER",
    "Origin": "VARCHAR",
    "OriginState": "VARCHAR",
    "Dest": "VARCHAR",
    "DestState": "VARCHAR",
    "CRSDepTime": "INTEGER",
    "DepTime": "INTEGER",
    "DepDelay": "DOUBLE",
    "DepDelayMinutes": "DOUBLE",
    "DepDel15": "DOUBLE",
    "DepartureDelayGroups": "DOUBLE",
    "DepTimeBlk": "VARCHAR",
    "TaxiOut": "DOUBLE",
    "WheelsOff": "INTEGER",
    "CRSArrTime": "INTEGER",
    "ArrDelay": "DOUBLE",
    "ArrDel15": "DOUBLE",
    "Cancelled": "DOUBLE",
    "CancellationCode": "VARCHAR",
    "Diverted": "DOUBLE",
    "CRSElapsedTime": "DOUBLE",
    "Distance": "DOUBLE",
    "CarrierDelay": "DOUBLE",
    "WeatherDelay": "DOUBLE",
    "NASDelay": "DOUBLE",
    "SecurityDelay": "DOUBLE",
    "LateAircraftDelay": "DOUBLE",
}

# Las 35 columnas del proyecto, en el orden del agregador original.
COLUMNS: list[str] = [
    "DepDel15",  # objetivo
    "FlightDate",
    "Year",
    "Reporting_Airline",
    "Quarter",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "Origin",
    "Dest",
    "OriginState",
    "DestState",
    "CRSDepTime",
    "DepTimeBlk",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "Flight_Number_Reporting_Airline",
    "Tail_Number",  # predictoras
    "Cancelled",
    "CancellationCode",
    "Diverted",  # control
    "DepTime",
    "DepDelay",
    "DepDelayMinutes",
    "DepartureDelayGroups",
    "TaxiOut",
    "WheelsOff",
    "ArrDelay",
    "ArrDel15",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",  # no predictoras
]
assert set(COLUMNS) == set(COLUMN_TYPES) and len(COLUMNS) == 35

# Columnas de desvíos Div1..Div5 (vacías en casi todos los vuelos). Si no se
# fija su tipo, DuckDB infiere VARCHAR en los meses donde están vacías y
# BIGINT/DOUBLE en los que traen algún valor, y el esquema cambia entre meses.
DIV_COLUMN = re.compile(
    r"^Div[1-5](Airport|AirportID|AirportSeqID|WheelsOn|WheelsOff|TotalGTime|LongestGTime|TailNum)$"
)
DIV_SUFFIX_TYPES = {
    "Airport": "VARCHAR",
    "TailNum": "VARCHAR",
    "AirportID": "BIGINT",
    "AirportSeqID": "BIGINT",
    "WheelsOn": "BIGINT",
    "WheelsOff": "BIGINT",
    "TotalGTime": "DOUBLE",
    "LongestGTime": "DOUBLE",
}


def column_type(name: str) -> str | None:
    """Tipo explícito de una columna, o None si se deja inferir."""
    if name in COLUMN_TYPES:
        return COLUMN_TYPES[name]
    m = DIV_COLUMN.match(name)
    return DIV_SUFFIX_TYPES[m.group(1)] if m else None


# Nombres que DuckDB asigna a la columna sin encabezado creada por la coma final.
JUNK_COLUMN = re.compile(r"^(column\d+|unnamed.*|)$", re.IGNORECASE)
# ...On_Time_Performance_(1987_present)_2026_1.csv  ->  año 2026, mes 1
FILE_PATTERN = re.compile(r"_(\d{4})_(\d{1,2})\.csv$", re.IGNORECASE)


def sql_str(path: Path) -> str:
    """Ruta como literal SQL (con / y comillas escapadas)."""
    return "'" + path.as_posix().replace("'", "''") + "'"


def find_csvs(data_dir: Path, out_dir: Path) -> list[tuple[int, int, Path]]:
    """Busca los CSV mensuales y devuelve [(año, mes, ruta)] ordenado."""
    found = []
    for p in data_dir.rglob("*.csv"):
        if out_dir in p.parents:
            continue
        m = FILE_PATTERN.search(p.name)
        if not m:
            print(f"  [aviso] se ignora (nombre no reconocido): {p}")
            continue
        found.append((int(m.group(1)), int(m.group(2)), p))
    found.sort()
    # Windows limita las rutas a 260 caracteres (MAX_PATH). Si una ruta lo
    # supera, os.scandir la lista pero open() falla con FileNotFoundError.
    if sys.platform == "win32":
        too_long = [p for _, _, p in found if len(str(p)) >= 255]
        if too_long:
            sys.exit(
                f"{len(too_long)} CSV tienen una ruta de {len(str(too_long[0]))} caracteres "
                f"(límite de Windows: 260), p. ej.:\n  {too_long[0]}\n"
                "Acorta los nombres de las carpetas mensuales o mueve el repositorio a una "
                "ruta corta (p. ej. C:\\dev\\...) y vuelve a ejecutar."
            )
    keys = [(y, mo) for y, mo, _ in found]
    dups = {k for k in keys if keys.count(k) > 1}
    if dups:
        sys.exit(f"Hay más de un CSV para estos meses: {sorted(dups)}")
    return found


def convert_month(
    con, year: int, month: int, csv: Path, out_file: Path, all_columns: bool = False
) -> int:
    """Convierte un CSV a parquet y valida. Devuelve el número de filas."""
    with open(csv, newline="", encoding="utf-8-sig") as fh:
        header = next(csvlib.reader(fh))

    if all_columns:
        # DuckDB falla si 'types' menciona columnas que no están en el archivo,
        # así que se pasan solo los tipos de las columnas presentes en el encabezado.
        types = {c: t for c in header if (t := column_type(c))}
        read = (
            f"read_csv({sql_str(csv)}, header = true, sample_size = -1, "
            f"types = {types!r})"
        )
        cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {read}").fetchall()]
        keep = [c for c in cols if not JUNK_COLUMN.match(c)]
    else:
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            sys.exit(f"Faltan columnas en {csv.name}: {missing}")
        # Se declara el tipo de TODAS las columnas del archivo (las que no se
        # usan, como VARCHAR): así DuckDB no infiere tipos y lee solo lo necesario.
        schema = {}
        for k, c in enumerate(header):
            name = f"_extra{k}" if JUNK_COLUMN.match(c) or c in schema else c
            schema[name] = COLUMN_TYPES.get(name, "VARCHAR")
        read = (
            f"read_csv({sql_str(csv)}, header = true, delim = ',', quote = '\"', "
            f"columns = {schema!r})"
        )
        keep = COLUMNS
    select = ", ".join(f'"{c}"' for c in keep)

    tmp = out_file.with_suffix(".parquet.tmp")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY (SELECT {select} FROM {read}) TO {sql_str(tmp)} "
        f"(FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    # Validaciones: el contenido debe corresponder al año/mes del nombre.
    # (hive_partitioning = false: si no, DuckDB tomaría Year/Month de la ruta
    # y no del contenido del archivo, y la validación no probaría nada.)
    n_pq = con.execute(
        f"SELECT count(*) FROM read_parquet({sql_str(tmp)}, hive_partitioning = false)"
    ).fetchone()[0]
    n_csv = con.execute(f"SELECT count(*) FROM {read}").fetchone()[0]
    bad, nulls = con.execute(
        f"SELECT count(*) FILTER (WHERE Year <> {year} OR Month <> {month}), "
        f"count(*) FILTER (WHERE FlightDate IS NULL) "
        f"FROM read_parquet({sql_str(tmp)}, hive_partitioning = false)"
    ).fetchone()
    if n_pq != n_csv or bad or nulls:
        tmp.unlink(missing_ok=True)
        sys.exit(
            f"Validación fallida en {year}-{month:02d}: filas csv={n_csv:,} "
            f"parquet={n_pq:,}; Year/Month distintos al nombre={bad:,}; "
            f"FlightDate nulo={nulls:,}"
        )
    tmp.replace(out_file)  # reemplazo atómico: nunca queda un parquet a medias
    return n_pq


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data" / "bronze" / "parquet"
    )
    ap.add_argument("--temp-dir", type=Path, default=REPO_ROOT / "src" / "duckdb_tmp")
    ap.add_argument("--memory-limit", default="6GB")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument(
        "--all-columns",
        action="store_true",
        help="conserva todas las columnas del CSV en vez de las 35 del proyecto",
    )
    ap.add_argument(
        "--force", action="store_true", help="reconvierte aunque el parquet ya exista"
    )
    ap.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help="total esperado (p. ej. el del parquet combinado) para comparar",
    )
    args = ap.parse_args()

    data_dir, out_dir = args.data_dir.resolve(), args.out_dir.resolve()
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    csvs = find_csvs(data_dir, out_dir)
    if not csvs:
        sys.exit(f"No se encontraron CSV en {data_dir}. ¿Ya descomprimiste los ZIP?")
    print(
        f"{len(csvs)} CSV encontrados ({csvs[0][0]}-{csvs[0][1]:02d} → {csvs[-1][0]}-{csvs[-1][1]:02d})"
    )

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute(f"SET temp_directory = '{args.temp_dir.as_posix()}'")
    if args.threads:
        con.execute(f"SET threads = {args.threads}")

    total, converted, skipped = 0, 0, 0
    for year, month, csv in csvs:
        out_file = out_dir / f"year={year}" / f"month={month:02d}" / "flights.parquet"
        if out_file.exists() and not args.force:
            n = con.execute(
                f"SELECT count(*) FROM read_parquet({sql_str(out_file)}, hive_partitioning = false)"
            ).fetchone()[0]
            print(f"  {year}-{month:02d}  ya existe ({n:>9,} filas) → se omite")
            total += n
            skipped += 1
            continue
        t0 = time.time()
        n = convert_month(con, year, month, csv, out_file, args.all_columns)
        total += n
        converted += 1
        mb = out_file.stat().st_size / 1e6
        print(
            f"  {year}-{month:02d}  {n:>9,} filas  {mb:7.1f} MB  {time.time() - t0:5.1f}s"
        )

    print(
        f"\nConvertidos: {converted} | omitidos: {skipped} | filas totales: {total:,}"
    )
    if args.expected_rows is not None:
        ok = total == args.expected_rows
        print(
            f"Esperado: {args.expected_rows:,} → {'OK ✔' if ok else 'NO COINCIDE ✘ (revisa meses faltantes o duplicados)'}"
        )

    # Esquema consistente entre meses (si no, la lectura conjunta fallará o promoverá tipos).
    report_schemas(con, out_dir)


def report_schemas(con, out_dir: Path) -> None:
    """Agrupa los meses por esquema y muestra en qué difieren del más frecuente."""
    groups: dict[tuple, list[str]] = {}
    for p in sorted(out_dir.glob("year=*/month=*/flights.parquet")):
        rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({sql_str(p)}, hive_partitioning = false)"
        ).fetchall()
        label = f"{p.parent.parent.name[5:]}-{p.parent.name[6:]}"
        groups.setdefault(tuple((r[0], r[1]) for r in rows), []).append(label)

    if len(groups) == 1:
        print("Esquema idéntico en todos los meses ✔")
        return

    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    base_cols, base_months = ordered[0]
    base = dict(base_cols)
    print(
        f"⚠ Hay {len(groups)} esquemas distintos entre meses. Referencia: el más frecuente "
        f"({len(base_months)} meses, {len(base_cols)} columnas)."
    )
    for cols, months in ordered[1:]:
        cur = dict(cols)
        extra = [c for c in cur if c not in base]
        missing = [c for c in base if c not in cur]
        retyped = [
            f"{c}: {base[c]} → {cur[c]}" for c in cur if c in base and cur[c] != base[c]
        ]
        reordered = (
            not extra
            and not missing
            and not retyped
            and [c for c, _ in cols] != [c for c, _ in base_cols]
        )
        shown = ", ".join(months[:6]) + (
            f", … (+{len(months) - 6})" if len(months) > 6 else ""
        )
        print(f"\n  Grupo de {len(months)} mes(es): {shown}")
        if extra:
            print(f"    columnas extra: {extra}")
        if missing:
            print(f"    columnas ausentes: {missing}")
        if retyped:
            print(f"    tipos distintos: {retyped}")
        if reordered:
            print("    mismas columnas y tipos, pero en otro orden")


if __name__ == "__main__":
    main()
