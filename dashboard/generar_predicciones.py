"""
Genera las predicciones vuelo por vuelo que usa la página 4 del dashboard.

Qué hace:
    1. Ejecuta las celdas del notebook del equipo src/flight_delay_modeling.ipynb que
       construyen los datos, las variables (sin fuga: solo lo conocido 4 horas antes)
       y entrenan el LightGBM. Así se usa exactamente el mismo código del modelado.
    2. Puntúa TODOS los vuelos de un mes (por defecto agosto de 2026).
    3. Para los 300 vuelos más riesgosos de cada día calcula las 3 variables que más
       subieron su riesgo (contribuciones SHAP de LightGBM, pred_contrib=True).

Por qué no usa directamente la capa oro: la capa oro del notebook guarda tablas
agregadas (por mes, hora, aerolínea...), no la predicción de cada vuelo, y la página 4
necesita la lista de vuelos de un día.

Cómo se corre (desde la raíz del repositorio; tarda ~10 minutos):
    python dashboard/generar_predicciones.py
    python dashboard/generar_predicciones.py --desde 2026-08-01 --hasta 2026-08-31 --n-train 4000000

Salidas: dashboard/data/predicciones/predicciones_ago2026.parquet y meta.json
"""

import argparse
import gc
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "src" / "flight_delay_modeling.ipynb"
SALIDA = Path(__file__).resolve().parent / "data" / "predicciones"

ap = argparse.ArgumentParser()
ap.add_argument("--desde", default="2026-08-01")
ap.add_argument("--hasta", default="2026-08-31")
ap.add_argument("--n-train", type=int, default=1_500_000, help="filas de entrenamiento (el notebook usa 4.000.000)")
ap.add_argument("--memoria", default="3GB", help="límite de memoria de DuckDB")
ap.add_argument("--top-shap", type=int, default=300, help="vuelos por día con explicación SHAP")
args = ap.parse_args()

celdas = ["".join(c["source"]) for c in json.load(open(NOTEBOOK, encoding="utf-8"))["cells"]
          if c["cell_type"] == "code"]
g = {"display": print, "__name__": "__main__"}


def correr(i, reemplazos=()):
    src = celdas[i]
    for viejo, nuevo in reemplazos:
        assert viejo in src, f"La celda {i} del notebook cambió: no encuentro {viejo!r}"
        src = src.replace(viejo, nuevo)
    t = time.time()
    exec(compile(src, f"celda_{i}", "exec"), g)
    print(f"[celda {i}] {time.time() - t:,.0f}s", flush=True)


SALIDA.mkdir(parents=True, exist_ok=True)
tmp = SALIDA / "duckdb_tmp"
correr(0)                                                             # imports
correr(1, [('"../data/silver/flights/**/*.parquet"', repr((RAIZ / "data/silver/flights/**/*.parquet").as_posix())),
           ('Path("../outputs/outputs_modeling")', f'Path({str(SALIDA / "tmp_outputs")!r})'),
           ('"12GB"', repr(args.memoria)), ('"duckdb_tmp"', repr(str(tmp))),
           ("N_TRAIN   = 4_000_000", f"N_TRAIN   = {args.n_train}"), ("N_EVAL    = 1_000_000", "N_EVAL    = 300_000")])
correr(2)                                                             # conexión DuckDB y capa plata
correr(3, [("CRSElapsedTime, Distance\n    FROM silver",                # + número de vuelo para mostrarlo
            "CRSElapsedTime, Distance, Flight_Number_Reporting_Airline AS flight_num\n    FROM silver")])
for i in (4, 5, 6, 9, 10, 12):                                        # variables, muestras, LightGBM
    correr(i)

import numpy as np  # noqa: E402

m, feats = g["m_lgbm"], g["FULL_FEATURES"]
umbral = g["best_cost_threshold"](g["y_val"], g["results"]["LightGBM"]["val"])
ev = g["evaluate_model"]("LightGBM", g["y_test"], g["results"]["LightGBM"]["test"], umbral)
for k in ("train", "val", "test"):
    g.pop(k, None)
gc.collect()

mes = g["build_split"](f"FlightDate >= DATE '{args.desde}' AND FlightDate <= DATE '{args.hasta}'", None)
mes["prob"] = np.concatenate([m.booster_.predict(g["model_matrix"](mes.iloc[i:i + 100_000], feats))
                              for i in range(0, len(mes), 100_000)])

# Explicación SHAP solo para los vuelos que verá el equipo (los más riesgosos de cada día).
# Rotación imposible: el avión que voló estaba programado para llegar de su vuelo anterior después de esta
# salida (o con < 20 min). Solo pasa si la aerolínea cambió el avión ese día: posible fuga de información.
mes["rotacion_imposible"] = (((mes["tail_leg_num"] > 1) & mes["turnaround_min"].isna())
                             | (mes["turnaround_min"] < 20)).astype(int)
mes["rank_dia"] = mes.groupby("FlightDate")["prob"].rank(ascending=False, method="first")
mes["rank_dia_limpio"] = (mes[mes["rotacion_imposible"] == 0]
                          .groupby("FlightDate")["prob"].rank(ascending=False, method="first"))
top = mes.index[(mes["rank_dia"] <= args.top_shap) | (mes["rank_dia_limpio"] <= args.top_shap)]
contrib = m.booster_.predict(g["model_matrix"](mes.loc[top], feats), pred_contrib=True)[:, :-1]
orden = np.argsort(-contrib, axis=1)[:, :3]
for j in range(3):
    mes[f"motivo{j + 1}"], mes[f"aporte{j + 1}"] = None, np.nan
    mes.loc[top, f"motivo{j + 1}"] = np.array(feats)[orden[:, j]]
    mes.loc[top, f"aporte{j + 1}"] = contrib[np.arange(len(top)), orden[:, j]]

cols = ["FlightDate", "Reporting_Airline", "flight_num", "tail", "Origin", "Dest", "dep_min", "arr_min",
        "Distance", "prob", "DepDel15", "prev_leg_delayed", "tail_known_legs", "tail_delayed_known",
        "tail_leg_num", "turnaround_min", "origin_delay_rate_today", "origin_known_today",
        "route_rate_30d", "origin_hr_rate_30d", "tail_rate_30d", "airline_rate_30d", "origin_dep_load",
        "motivo1", "aporte1", "motivo2", "aporte2", "motivo3", "aporte3", "rotacion_imposible"]
mes[[c for c in cols if c in mes.columns]].to_parquet(SALIDA / "predicciones_ago2026.parquet", index=False)
json.dump({"umbral_costo": umbral,
           "metricas_prueba": {k: float(ev[k]) for k in ("roc_auc", "pr_auc", "ece", "precision", "recall")},
           "n_train": args.n_train, "horizonte_min": 240},
          open(SALIDA / "meta.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"Listo: {len(mes):,} vuelos puntuados en {SALIDA}")
