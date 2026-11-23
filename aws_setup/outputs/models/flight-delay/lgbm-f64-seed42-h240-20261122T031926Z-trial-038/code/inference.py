"""Contrato de inferencia SageMaker para LightGBM BTS.

Las variables ya deben estar calculadas para 240 minutos antes de la salida.
No acepta filas crudas de BTS, ni genera historia/ASOF automáticamente.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def model_fn(model_dir):
    root = Path(model_dir)
    with (root / "feature_schema.json").open(encoding="utf-8") as f:
        schema = json.load(f)
    with (root / "metadata.json").open(encoding="utf-8") as f:
        metadata = json.load(f)
    booster = lgb.Booster(model_file=str(root / "model.txt"))
    assert booster.feature_name() == schema["feature_names"], "Orden de variables incompatible"
    return {"booster": booster, "schema": schema, "metadata": metadata}


def input_fn(request_body, request_content_type):
    if request_content_type.split(";")[0].strip().lower() != "application/json":
        raise ValueError("Usa ContentType=application/json")
    payload = json.loads(request_body)
    if not isinstance(payload, dict):
        raise ValueError("Se espera un objeto JSON")
    if "features" in payload:
        rows = [payload["features"]]
    elif "instances" in payload:
        rows = payload["instances"]
    else:
        raise ValueError("Usa {\"features\": {...}} o {\"instances\": [{...}, ...]}")
    if not isinstance(rows, list) or not rows or len(rows) > 100:
        raise ValueError("instances debe contener de 1 a 100 registros")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Cada registro debe ser un objeto JSON")
    return rows


def predict_fn(input_data, model):
    schema = model["schema"]
    names = schema["feature_names"]
    cats = schema["cat_cols"]
    levels = schema["cat_levels"]
    cleaned = []
    for row in input_data:
        missing = set(names) - set(row)
        extra = set(row) - set(names)
        if missing or extra:
            raise ValueError(f"Variables faltantes: {sorted(missing)}. No admitidas: {sorted(extra)}")
        cleaned.append({k: row[k] for k in names})
    frame = pd.DataFrame(cleaned, columns=names)
    for col in names:
        if col in cats:
            frame[col] = pd.Categorical(frame[col], categories=levels[col])
        else:
            frame[col] = pd.to_numeric(frame[col], errors="raise").astype("float64")
    p = np.asarray(model["booster"].predict(frame), dtype=float)
    metadata = model["metadata"]
    threshold = float(metadata["decision_threshold"])
    return {
        "model_version": metadata["model_version"],
        "prediction_horizon_min": int(metadata["prediction_horizon_minutes"]),
        "predictions": [
            {"probability_delay_15min": float(x), "alert": bool(x >= threshold)}
            for x in p
        ],
        "decision_threshold": threshold,
    }


def output_fn(prediction, accept):
    if accept.split(";")[0].strip().lower() not in ("application/json", "*/*"):
        raise ValueError("El endpoint responde application/json")
    return json.dumps(prediction, ensure_ascii=False, allow_nan=False), "application/json"
