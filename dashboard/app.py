"""
Dashboard — Riesgo de retraso en vuelos comerciales de EE. UU.
Proyecto Integrador 1, Maestría en Ciencia de Datos y Analítica, EAFIT. Grupo 12.

Cómo correrlo (desde la raíz del repositorio):
    pip install -r dashboard/requirements.txt
    streamlit run dashboard/app.py

Este archivo es la puerta de entrada: configura la página, dibuja los filtros
de la barra lateral (los mismos para todas las páginas) y carga la página elegida.
"""

import pandas as pd
import streamlit as st

from utils.datos import cargar_agg_mes
from utils.estilo import AEROLINEAS, CSS, MESES

st.set_page_config(
    page_title="Riesgo de retraso en vuelos",
    page_icon="✈️",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)

df = cargar_agg_mes()

# ---------------------------------------------------------------------------
# Barra lateral: filtros globales (regla del curso: los controles van a la
# izquierda y el panel principal queda solo para la narrativa visual).
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filtros")

    periodos = [pd.Timestamp(p) for p in sorted(df["periodo"].unique())]
    etiqueta = lambda p: f"{MESES[p.month - 1]} {p.year}"
    st.select_slider(
        "Periodo",
        options=periodos,
        value=(periodos[0], periodos[-1]),
        format_func=etiqueta,
        key="f_periodo",
    )

    orden_aer = (df.groupby("aerolinea")["vuelos"].sum().sort_values(ascending=False).index)
    st.multiselect(
        "Aerolínea",
        options=list(orden_aer),
        format_func=lambda c: f"{AEROLINEAS.get(c, c)} ({c})",
        placeholder="Todas",
        key="f_aerolineas",
    )

    orden_org = (df.groupby("origen")["vuelos"].sum().sort_values(ascending=False).index)
    st.multiselect(
        "Aeropuerto de origen",
        options=list(orden_org),
        placeholder="Todos (ordenados por número de vuelos)",
        key="f_origenes",
    )

    st.caption("Sin selección = se incluyen todas las aerolíneas y todos los aeropuertos.")

# ---------------------------------------------------------------------------
# Navegación entre páginas. Se irán agregando las demás páginas aquí.
# ---------------------------------------------------------------------------
paginas = [
    st.Page("paginas/1_resumen.py", title="1. Resumen del problema", icon="📈", default=True, url_path="resumen"),
    st.Page("paginas/2_riesgo.py", title="2. Dónde está el riesgo", icon="📍", url_path="riesgo"),
    st.Page("paginas/3_modelo.py", title="3. ¿Podemos confiar en el modelo?", icon="🎯", url_path="modelo"),
    st.Page("paginas/4_priorizacion.py", title="4. Priorización del día", icon="📋", url_path="priorizacion"),
    st.Page("paginas/5_limites.py", title="5. Dónde no confiar tanto", icon="⚠️", url_path="limites"),
]
st.navigation(paginas).run()
