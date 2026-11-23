"""
Variables de modelado construidas a partir de la capa SILVER, con caché ("gold").

Este módulo NO repite nada de la ingesta: no limpia, no audita fuga y no elimina columnas
(eso ya lo hizo `Ingesta_y_procesamiento_de_datos.ipynb`, cuya salida es silver). Solo construye
las variables del modelo y las guarda para no recalcularlas en cada ejecución:

    data/gold/model_features/{train,val,test}.parquet  +  meta.json

Variables (ninguna usa el resultado del propio vuelo ni de vuelos posteriores):
  - estáticas: calendario, hora programada, distancia, categóricas, días a feriado federal;
  - patrones históricos: tasas de retraso por llave, solo con días ESTRICTAMENTE anteriores;
  - horario/rotación: congestión programada, tramo de la matrícula en el día, tiempo de espera;
  - estado del día: lo que ya se sabría `pred_horizon_min` minutos antes de la salida.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

TARGET = "DepDel15"
CAT_COLS = ["Reporting_Airline", "Origin", "Dest", "route", "OriginState", "DestState", "DepTimeBlk"]
NUM_STATIC = ["Distance", "CRSElapsedTime", "dep_min", "arr_min", "Month", "DayOfWeek", "DayofMonth",
              "day_of_year", "is_weekend", "days_to_holiday"]
HIST_KEYS = {"origin": "Origin", "route": "route", "airline": "Reporting_Airline",
             "tail": "tail", "origin_hr": "origin_hr"}
SCHED_COLS = ["origin_dep_load", "origin_day_load", "dest_arr_load", "tail_leg_num", "turnaround_min"]
STATUS_COLS = ["prev_leg_delayed", "tail_delayed_known", "tail_known_legs", "origin_known_today",
               "origin_delay_rate_today"]
FEATURE_VERSION = 1


@dataclass
class FeatureConfig:
    silver_glob: str = "../data/silver/flights/**/*.parquet"
    train_end: str = "2024-12-31"
    val_end: str = "2025-08-31"
    warmup_days: int = 60
    hist_window_days: int = 30
    hist_smooth: int = 50
    pred_horizon_min: int = 120
    seed: int = 42
    memory_limit: str = "8GB"
    temp_dir: str = "duckdb_tmp"


def hist_cols(cfg: FeatureConfig) -> list[str]:
    return [f"{k}_rate_{s}" for k in HIST_KEYS for s in ("all", f"{cfg.hist_window_days}d")]


def feature_lists(cfg: FeatureConfig) -> dict[str, list[str]]:
    static = NUM_STATIC + CAT_COLS
    hist = static + hist_cols(cfg)
    sched = hist + SCHED_COLS
    return {"static": static, "hist": hist, "sched": sched, "full": sched + STATUS_COLS}


def fit_cat_levels(train: pd.DataFrame) -> dict[str, list]:
    """Categorías fijadas con el entrenamiento: lo no visto queda como faltante."""
    return {c: sorted(train[c].dropna().unique()) for c in CAT_COLS}


def make_matrix(df: pd.DataFrame, features: list[str], cat_levels: dict[str, list]) -> pd.DataFrame:
    X = df[features].copy()
    for c in features:
        if c in cat_levels:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


# ----------------------------------------------------------------------------------------------
# Construcción (solo se ejecuta si no hay caché válida)
# ----------------------------------------------------------------------------------------------
def connect(cfg: FeatureConfig):
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{cfg.memory_limit}'")
    Path(cfg.temp_dir).mkdir(exist_ok=True)
    con.execute(f"SET temp_directory = '{cfg.temp_dir}'")
    con.execute(f"""
        CREATE OR REPLACE VIEW silver AS
        SELECT * FROM read_parquet('{cfg.silver_glob}', hive_partitioning = true,
                                   hive_types = {{'Year': 'SMALLINT', 'Month': 'TINYINT'}})""")
    d_min, d_max = con.sql("SELECT MIN(FlightDate), MAX(FlightDate) FROM silver").fetchall()[0]

    cal_dates = pd.date_range(pd.Timestamp(d_min) - pd.Timedelta(days=15), pd.Timestamp(d_max) + pd.Timedelta(days=15))
    hol = USFederalHolidayCalendar().holidays(start=cal_dates.min(), end=cal_dates.max()).values.astype("datetime64[D]")
    cal_np = cal_dates.values.astype("datetime64[D]")
    days_to_hol = np.abs(cal_np[:, None] - hol[None, :]).min(axis=1).astype(float).clip(max=10)
    cal_df = pd.DataFrame({"FlightDate": cal_dates.date, "days_to_holiday": days_to_hol})
    con.register("cal_df", cal_df)
    con.execute("CREATE OR REPLACE TABLE calendar AS SELECT CAST(FlightDate AS DATE) AS FlightDate, days_to_holiday FROM cal_df")

    con.execute(f"""
        CREATE OR REPLACE VIEW base AS
        SELECT FlightDate, CAST({TARGET} AS INTEGER) AS {TARGET},
               Reporting_Airline, Origin, Dest, OriginState, DestState, DepTimeBlk,
               Origin || '-' || Dest                                  AS route,
               NULLIF(TRIM(Tail_Number), '')                          AS tail,
               Origin || '_' || CAST(CRSDepTime // 100 AS VARCHAR)    AS origin_hr,
               CAST(Month AS INTEGER) AS Month, CAST(DayOfWeek AS INTEGER) AS DayOfWeek,
               CAST(DayofMonth AS INTEGER) AS DayofMonth,
               CAST(DayOfWeek IN (6, 7) AS INTEGER)                   AS is_weekend,
               DAYOFYEAR(FlightDate)                                  AS day_of_year,
               CAST(CRSDepTime // 100 AS INTEGER)                     AS dep_hour,
               CAST(CRSArrTime // 100 AS INTEGER)                     AS arr_hour,
               CAST((CRSDepTime // 100) * 60 + CRSDepTime % 100 AS INTEGER) AS dep_min,
               CAST((CRSArrTime // 100) * 60 + CRSArrTime % 100 AS INTEGER) AS arr_min,
               CRSElapsedTime, Distance
        FROM silver""")
    return con, d_min, d_max


def build_aux_tables(con, cfg: FeatureConfig) -> float:
    """Tablas de historial, congestión y estado del día. Devuelve el prior (tasa de entrenamiento)."""
    prior = con.sql(f"SELECT AVG({TARGET}) FROM base WHERE FlightDate <= DATE '{cfg.train_end}'").fetchall()[0][0]
    W = cfg.hist_window_days
    for name, col in HIST_KEYS.items():
        con.execute(f"""
            CREATE OR REPLACE TABLE hist_{name} AS
            WITH daily AS (
                SELECT {col} AS k, FlightDate, COUNT(*) AS n, SUM({TARGET}) AS d
                FROM base WHERE {col} IS NOT NULL GROUP BY 1, 2)
            SELECT k, FlightDate,
                   COALESCE(SUM(n) OVER w_all, 0) AS n_all, COALESCE(SUM(d) OVER w_all, 0) AS d_all,
                   COALESCE(SUM(n) OVER w_win, 0) AS n_win, COALESCE(SUM(d) OVER w_win, 0) AS d_win
            FROM daily
            WINDOW w_all AS (PARTITION BY k ORDER BY FlightDate ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                   w_win AS (PARTITION BY k ORDER BY FlightDate
                             RANGE BETWEEN INTERVAL {W} DAYS PRECEDING AND INTERVAL 1 DAYS PRECEDING)""")
    con.execute("CREATE OR REPLACE TABLE load_origin     AS SELECT Origin, FlightDate, dep_hour, COUNT(*) AS origin_dep_load FROM base GROUP BY ALL")
    con.execute("CREATE OR REPLACE TABLE load_origin_day AS SELECT Origin, FlightDate, COUNT(*) AS origin_day_load FROM base GROUP BY ALL")
    con.execute("CREATE OR REPLACE TABLE load_dest       AS SELECT Dest, FlightDate, arr_hour, COUNT(*) AS dest_arr_load FROM base GROUP BY ALL")
    con.execute("CREATE OR REPLACE TABLE legs AS SELECT tail, FlightDate, dep_min, arr_min, DepDel15 FROM base WHERE tail IS NOT NULL")
    con.execute("""
        CREATE OR REPLACE TABLE ap_cum AS
        WITH g AS (SELECT Origin, FlightDate, dep_min, COUNT(*) AS n, SUM(DepDel15) AS d FROM base GROUP BY ALL)
        SELECT Origin, FlightDate, dep_min, SUM(n) OVER w AS cum_n, SUM(d) OVER w AS cum_d
        FROM g WINDOW w AS (PARTITION BY Origin, FlightDate ORDER BY dep_min ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)""")
    return float(prior)


def build_split(con, cfg: FeatureConfig, prior: float, where: str, n_rows: int | None) -> pd.DataFrame:
    H, W = cfg.pred_horizon_min, cfg.hist_window_days
    q = f"SELECT b.*, c.days_to_holiday FROM base b JOIN calendar c USING (FlightDate) WHERE {where}"
    if n_rows:
        q = f"SELECT * FROM ({q}) USING SAMPLE reservoir({n_rows} ROWS) REPEATABLE ({cfg.seed})"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE smp AS
        SELECT ROW_NUMBER() OVER () AS rid, *, dep_min - {H} - 15 AS cut_min, dep_min - {H} AS cut_arr FROM ({q})""")
    con.execute("""CREATE OR REPLACE TEMP TABLE tail_feat AS
        SELECT s.rid, COUNT(l.dep_min) AS tail_prev_legs,
               s.dep_min - arg_max(l.arr_min, l.dep_min) AS turnaround_raw,
               SUM(l.DepDel15) FILTER (WHERE l.arr_min <= s.cut_arr) AS tail_delayed_known,
               COUNT(*)        FILTER (WHERE l.arr_min <= s.cut_arr) AS tail_known_legs,
               arg_max(l.DepDel15, l.dep_min) FILTER (WHERE l.arr_min <= s.cut_arr) AS prev_leg_delayed
        FROM smp s LEFT JOIN legs l ON l.tail = s.tail AND l.FlightDate = s.FlightDate AND l.dep_min < s.dep_min
        WHERE s.tail IS NOT NULL GROUP BY s.rid, s.dep_min""")
    con.execute("""CREATE OR REPLACE TEMP TABLE ap_feat AS
        SELECT s.rid, a.cum_n AS origin_known_today, a.cum_d AS origin_delayed_today
        FROM smp s ASOF LEFT JOIN ap_cum a
          ON s.Origin = a.Origin AND s.FlightDate = a.FlightDate AND s.cut_min >= a.dep_min""")
    joins, cols = [], []
    for name, col in HIST_KEYS.items():
        joins.append(f"LEFT JOIN hist_{name} h_{name} ON h_{name}.k = s.{col} AND h_{name}.FlightDate = s.FlightDate")
        for suffix, (n, d) in {"all": ("n_all", "d_all"), f"{W}d": ("n_win", "d_win")}.items():
            cols.append(f"(COALESCE(h_{name}.{d}, 0) + {cfg.hist_smooth} * CAST({prior} AS DOUBLE)) / "
                        f"(COALESCE(h_{name}.{n}, 0) + {cfg.hist_smooth}) AS {name}_rate_{suffix}")
    sql = f"""
        SELECT s.* EXCLUDE (cut_min, cut_arr, rid), {', '.join(cols)},
               lo.origin_dep_load, lod.origin_day_load, ld.dest_arr_load,
               tf.tail_prev_legs + 1 AS tail_leg_num,
               CASE WHEN tf.turnaround_raw >= 0 THEN tf.turnaround_raw END AS turnaround_min,
               tf.prev_leg_delayed, tf.tail_delayed_known, tf.tail_known_legs,
               COALESCE(af.origin_known_today, 0) AS origin_known_today,
               COALESCE(af.origin_delayed_today, 0) AS origin_delayed_today
        FROM smp s {' '.join(joins)}
        LEFT JOIN load_origin     lo  ON lo.Origin = s.Origin AND lo.FlightDate = s.FlightDate AND lo.dep_hour = s.dep_hour
        LEFT JOIN load_origin_day lod ON lod.Origin = s.Origin AND lod.FlightDate = s.FlightDate
        LEFT JOIN load_dest       ld  ON ld.Dest = s.Dest AND ld.FlightDate = s.FlightDate AND ld.arr_hour = s.arr_hour
        LEFT JOIN tail_feat tf ON tf.rid = s.rid
        LEFT JOIN ap_feat   af ON af.rid = s.rid"""
    df = con.sql(sql).df()
    df["origin_delay_rate_today"] = (df.origin_delayed_today + 10 * df.origin_rate_30d) / (df.origin_known_today + 10)
    return _numeric_clean(df)


def _numeric_clean(df: pd.DataFrame) -> pd.DataFrame:
    """DuckDB devuelve enteros con NULL como tipos anulables (pd.NA): se pasan a float (NaN)."""
    num = NUM_STATIC + SCHED_COLS + STATUS_COLS + ["origin_delayed_today"]
    for c in num + [c for c in df.columns if c.endswith(("_rate_all",)) or "_rate_" in c]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def verify_no_leakage(con, cfg: FeatureConfig, prior: float, df: pd.DataFrame, n: int = 8) -> None:
    """Recalcula a mano (consultando silver) variables de filas de `df`; falla si no coinciden."""
    H, W = cfg.pred_horizon_min, cfg.hist_window_days
    sample = df.dropna(subset=["tail"]).sample(n, random_state=cfg.seed)

    def brute_rate(col, value, date, window_days=None):
        lo = f"AND FlightDate >= DATE '{(date - pd.Timedelta(days=window_days)).date()}'" if window_days else ""
        c, d = con.execute(f"SELECT COUNT(*), COALESCE(SUM({TARGET}), 0) FROM base WHERE {col} = ? "
                           f"AND FlightDate < DATE '{date.date()}' {lo}", [value]).fetchall()[0]
        return (d + cfg.hist_smooth * prior) / (c + cfg.hist_smooth)

    for _, r in sample.iterrows():
        for key, feat in [("route", "route_rate_all"), ("tail", "tail_rate_all"),
                          ("route", f"route_rate_{W}d"), ("origin_hr", "origin_hr_rate_all")]:
            win = W if feat.endswith(f"{W}d") else None
            exp = brute_rate(HIST_KEYS[key], r[HIST_KEYS[key]], r.FlightDate, win)
            assert np.isclose(r[feat], exp, atol=1e-9), f"Historial no coincide: {feat} {r[feat]} vs {exp}"
        date = r.FlightDate.date()
        n_o, d_o = con.execute("SELECT COUNT(*), COALESCE(SUM(DepDel15), 0) FROM base WHERE Origin = ? "
                               "AND FlightDate = ? AND dep_min + 15 <= ?", [r.Origin, date, r.dep_min - H]).fetchone()
        legs_ = con.execute("SELECT dep_min, arr_min, DepDel15 FROM base WHERE tail = ? AND FlightDate = ? "
                            "AND dep_min < ? ORDER BY dep_min", [r["tail"], date, r.dep_min]).fetchall()
        known = [x for x in legs_ if x[1] <= r.dep_min - H]
        turn = r.dep_min - legs_[-1][1] if legs_ else np.nan
        expected = {
            "origin_known_today": n_o, "origin_delay_rate_today": (d_o + 10 * r.origin_rate_30d) / (n_o + 10),
            "tail_leg_num": len(legs_) + 1, "turnaround_min": turn if (legs_ and turn >= 0) else np.nan,
            "tail_known_legs": len(known),
            "tail_delayed_known": float(sum(x[2] for x in known)) if known else np.nan,
            "prev_leg_delayed": float(known[-1][2]) if known else np.nan,
        }
        for feat, exp in expected.items():
            assert np.isclose(r[feat], exp, atol=1e-9, equal_nan=True), f"Estado del día no coincide: {feat} {r[feat]} vs {exp}"


# ----------------------------------------------------------------------------------------------
# Caché
# ----------------------------------------------------------------------------------------------
def _signature(cfg: FeatureConfig, n_train, n_eval, n_silver, d_max) -> dict:
    sig = {k: v for k, v in asdict(cfg).items() if k not in ("silver_glob", "memory_limit", "temp_dir")}
    sig.update(n_train=n_train, n_eval=n_eval, silver_rows=int(n_silver), silver_max_date=str(d_max),
               feature_version=FEATURE_VERSION)
    return sig


def load_or_build(cfg: FeatureConfig, cache_dir, n_train: int | None, n_eval: int | None,
                  force: bool = False, verify: bool = True) -> dict:
    """Devuelve {'train','val','test','prior','from_cache'}. Reconstruye solo si cambió algo."""
    cache_dir = Path(cache_dir)
    con, d_min, d_max = connect(cfg)
    n_silver = con.sql("SELECT COUNT(*) FROM silver").fetchall()[0][0]
    sig = _signature(cfg, n_train, n_eval, n_silver, d_max)
    meta_p = cache_dir / "meta.json"
    files = {s: cache_dir / f"{s}.parquet" for s in ("train", "val", "test")}

    if not force and meta_p.exists() and all(p.exists() for p in files.values()):
        meta = json.loads(meta_p.read_text())
        if meta.get("signature") == sig:
            out = {s: _numeric_clean(con.sql(f"SELECT * FROM read_parquet('{p.as_posix()}')").df()) for s, p in files.items()}
            out.update(prior=meta["prior"], from_cache=True)
            return out
        print("La caché existe pero la configuración o silver cambió: se reconstruye.")

    t0 = time.time()
    prior = build_aux_tables(con, cfg)
    print(f"Tablas auxiliares listas en {time.time() - t0:,.0f}s | prior = {prior:.4f}")
    d0 = pd.Timestamp(d_min) + pd.Timedelta(days=cfg.warmup_days)
    wheres = {
        "train": (f"FlightDate <= DATE '{cfg.train_end}' AND FlightDate >= DATE '{d0.date()}'", n_train),
        "val":   (f"FlightDate > DATE '{cfg.train_end}' AND FlightDate <= DATE '{cfg.val_end}'", n_eval),
        "test":  (f"FlightDate > DATE '{cfg.val_end}'", n_eval),
    }
    out = {}
    for s, (w, n) in wheres.items():
        t1 = time.time()
        out[s] = build_split(con, cfg, prior, w, n)
        print(f"  {s}: {len(out[s]):,} filas en {time.time() - t1:,.0f}s")
    if verify:
        verify_no_leakage(con, cfg, prior, out["train"])
        print("✔ Verificación anti-fuga: variables históricas y de estado del día coinciden con el recálculo a mano.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    for s, df in out.items():
        con.register("df_tmp", df)
        con.execute(f"COPY df_tmp TO '{files[s].as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        con.unregister("df_tmp")
    meta_p.write_text(json.dumps({"signature": sig, "prior": prior}, indent=2))
    out.update(prior=prior, from_cache=False)
    return out
