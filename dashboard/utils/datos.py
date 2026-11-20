"""Carga de datos y filtros compartidos por todas las páginas."""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data"


def _con_periodo(df: pd.DataFrame) -> pd.DataFrame:
    df["periodo"] = pd.to_datetime(dict(year=df["anio"], month=df["mes"], day=1))
    return df


@st.cache_data
def cargar_agg_mes() -> pd.DataFrame:
    """Vuelos y retrasados por año, mes, aerolínea y aeropuerto de origen."""
    return _con_periodo(pd.read_parquet(DATA / "agg_mes.parquet"))


@st.cache_data
def cargar_agg_hora() -> pd.DataFrame:
    """Lo mismo, abierto además por hora programada de salida (0 a 23)."""
    return _con_periodo(pd.read_parquet(DATA / "agg_hora.parquet"))


@st.cache_data
def cargar_agg_dia_hora() -> pd.DataFrame:
    """Vuelos y retrasados por año, mes, aerolínea, día de la semana (1 = lunes) y hora.
    No tiene aeropuerto: con él la tabla pasaría de 3,6 millones de filas."""
    return _con_periodo(pd.read_parquet(DATA / "agg_dia_hora.parquet"))


@st.cache_data
def cargar_modelo(nombre: str) -> pd.DataFrame:
    """Resultados de la prueba del modelo (copiados de outputs/outputs_modeling del repo).
    Prueba: 1.000.000 de vuelos de sep-2025 a ago-2026 que el modelo nunca vio."""
    return pd.read_csv(DATA / "modelo" / f"{nombre}.csv")


def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica los filtros de la barra lateral guardados en st.session_state.
    Lista vacía de aerolíneas o aeropuertos = todos.
    Si la tabla no tiene columna 'origen', el filtro de aeropuerto no se aplica."""
    ini, fin = st.session_state.get("f_periodo", (df["periodo"].min(), df["periodo"].max()))
    out = df[(df["periodo"] >= ini) & (df["periodo"] <= fin)]
    aer = st.session_state.get("f_aerolineas", [])
    if aer:
        out = out[out["aerolinea"].isin(aer)]
    org = st.session_state.get("f_origenes", [])
    if org and "origen" in out.columns:
        out = out[out["origen"].isin(org)]
    return out


def tasa(g: pd.DataFrame) -> float:
    """Tasa de retraso correcta de un grupo: total retrasados / total vuelos.
    (Promediar tasas de subgrupos daría un resultado equivocado.)"""
    v = g["vuelos"].sum()
    return g["retrasados"].sum() / v if v else float("nan")


def tasa_por(df: pd.DataFrame, col: str | list) -> pd.DataFrame:
    """Suma vuelos y retrasados por una o varias columnas y calcula la tasa."""
    return (df.groupby(col, as_index=False)[["vuelos", "retrasados"]].sum()
              .assign(tasa=lambda d: d["retrasados"] / d["vuelos"]))
