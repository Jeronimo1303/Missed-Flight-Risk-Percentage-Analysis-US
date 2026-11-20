"""
Prepara las tablas pequeñas que consume el dashboard, a partir de la capa plata.

Por qué existe este script:
    La capa plata tiene ~31,8 millones de vuelos (unos 200 MB en Parquet).
    Un dashboard no necesita cada vuelo para mostrar tasas por mes o por
    aerolínea: necesita conteos ya agregados. Este script hace esa agregación
    una sola vez con DuckDB y guarda tablas de pocos MB en dashboard/data/.
    Así el dashboard carga en segundos y se puede desplegar en la nube.

    Guardamos conteos (vuelos y retrasados), nunca tasas: al filtrar, la tasa
    correcta de un grupo es sum(retrasados) / sum(vuelos). Promediar tasas de
    subgrupos daría un número equivocado.

Cómo se corre (desde la raíz del repositorio):
    python dashboard/preparar_datos.py

Entrada : data/silver/flights/Year=YYYY/Month=M/data_0.parquet
Salidas : dashboard/data/agg_mes.parquet       (mes x aerolínea x origen)
          dashboard/data/agg_hora.parquet      (mes x aerolínea x origen x hora de salida)
          dashboard/data/agg_dia_hora.parquet  (mes x aerolínea x día de la semana x hora)
"""

from pathlib import Path

import duckdb

RAIZ = Path(__file__).resolve().parents[1]
PLATA = RAIZ / "data" / "silver" / "flights" / "*" / "*" / "*.parquet"
SALIDA = Path(__file__).resolve().parent / "data"
SALIDA.mkdir(exist_ok=True)

con = duckdb.connect()
con.execute("SET enable_progress_bar = false")

# Vista base con las columnas que usa el dashboard.
# hora = hora PROGRAMADA de salida (CRSDepTime 1435 -> 14). 2400 se toma como 0.
con.execute(f"""
    CREATE VIEW vuelos AS
    SELECT
        CAST(year(FlightDate)  AS SMALLINT)        AS anio,
        CAST(month(FlightDate) AS TINYINT)         AS mes,
        Reporting_Airline                          AS aerolinea,
        Origin                                     AS origen,
        CAST(DayOfWeek AS TINYINT)                 AS dia_semana,   -- 1 = lunes ... 7 = domingo
        CAST((CRSDepTime // 100) % 24 AS TINYINT)  AS hora,
        DepDel15
    FROM read_parquet('{PLATA.as_posix()}', hive_partitioning = true)
""")

TABLAS = {
    # Página 1 y filtros de la barra lateral
    "agg_mes": "anio, mes, aerolinea, origen",
    # Página 2: tasa por hora de salida (respeta todos los filtros)
    "agg_hora": "anio, mes, aerolinea, origen, hora",
    # Página 2: mapa de calor día x hora (respeta periodo y aerolínea)
    "agg_dia_hora": "anio, mes, aerolinea, dia_semana, hora",
}

for nombre, llaves in TABLAS.items():
    destino = (SALIDA / f"{nombre}.parquet").as_posix()
    con.execute(f"""
        COPY (
            SELECT {llaves},
                   COUNT(*)                       AS vuelos,
                   CAST(SUM(DepDel15) AS INTEGER) AS retrasados
            FROM vuelos
            GROUP BY ALL
        ) TO '{destino}' (FORMAT parquet, COMPRESSION zstd)
    """)
    filas, vuelos, retrasados = con.execute(
        f"SELECT COUNT(*), SUM(vuelos), SUM(retrasados) FROM '{destino}'").fetchone()
    print(f"{nombre}.parquet -> {filas:,} filas, {vuelos:,} vuelos, {retrasados:,} retrasados")
