"""Métricas y utilidades de evaluación del modelo (probabilidades de retraso)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, log_loss,
                             roc_auc_score)


def best_cost_threshold(y_true, y_prob, c_fp=1.0, c_fn=3.0) -> float:
    """Umbral que minimiza c_fp*FP + c_fn*FN (se usa sobre VALIDACIÓN). Alerta si p >= umbral."""
    y_true = np.asarray(y_true)
    order = np.argsort(-y_prob)
    ps, ys = y_prob[order], y_true[order]
    cum_tp, cum_fp = np.cumsum(ys), np.cumsum(1 - ys)
    P = ys.sum()
    cost = c_fp * cum_fp + c_fn * (P - cum_tp)
    cost = np.where(np.r_[ps[1:] != ps[:-1], True], cost, np.inf)
    i = int(np.argmin(cost))
    return float(ps[i]) if cost[i] < c_fn * P else 1.01


def expected_calibration_error(y_true, y_prob, bins: int = 20) -> float:
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bin"] = pd.qcut(df.p, bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
    return float((g.n / g.n.sum() * (g.obs - g.pred).abs()).sum())


def evaluate_model(name, y_true, y_prob, threshold, c_fp=1.0, c_fn=3.0) -> dict:
    y_true = np.asarray(y_true)
    prev = y_true.mean()
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "modelo": name, "umbral": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob), "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision, "recall": recall, "f1": f1, "lift_precision": precision / prev,
        "alertas_%": (tp + fp) / len(y_true), "costo_x1000": (c_fp * fp + c_fn * fn) / len(y_true) * 1000,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def paired_bootstrap_diff(y_true, p_a, p_b, metric_fn, n_boot=100, rows=300_000, seed=42):
    """metric(A) − metric(B) con IC 95 % (bootstrap pareado sobre una submuestra)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    sub = rng.choice(len(y_true), size=min(rows, len(y_true)), replace=False)
    ys, a, b = y_true[sub], p_a[sub], p_b[sub]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(ys), len(ys))
        diffs[i] = metric_fn(ys[idx], a[idx]) - metric_fn(ys[idx], b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(metric_fn(ys, a) - metric_fn(ys, b)), float(lo), float(hi)


def top_k_table(y_true, y_prob, ks=(0.01, 0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    order = np.argsort(-y_prob)
    ys = np.asarray(y_true)[order]
    prev = ys.mean()
    rows = []
    for k in ks:
        m = int(len(ys) * k)
        prec = ys[:m].mean()
        rows.append({"top_k_%": k, "vuelos": m, "precision": prec, "lift": prec / prev,
                     "recall_capturado": ys[:m].sum() / ys.sum()})
    return pd.DataFrame(rows)
