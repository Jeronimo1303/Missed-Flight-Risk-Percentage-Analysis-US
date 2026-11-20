"""
Página 5 — ¿Dónde no confiar tanto en el modelo? (cierre honesto de la RESOLUCIÓN).

Pregunta de negocio: ¿en qué aerolíneas, aeropuertos, horas o temporadas el modelo
acierta menos o se equivoca de forma sistemática?
Audiencia: gerencia de operaciones y jurado.
Acción esperada: usar la alerta con más cautela en esos grupos y saber qué mejorar después.

Datos: análisis por subgrupos de la prueba (outputs/outputs_modeling/subgrupos_*.csv).
Los filtros de la barra lateral no aplican aquí.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.datos import cargar_modelo
from utils.estilo import (AEROLINEAS, AZUL, GRIS, GRIS_CLARO, ROJO, TEXTO_SUAVE, layout_base)

st.title("¿Dónde no confiar tanto en el modelo?")
st.caption("Ningún modelo acierta igual en todos los casos. Esta página muestra en qué grupos de vuelos "
           "el modelo funciona peor, para usar sus alertas con criterio.")
st.info("Resultados de la prueba del modelo (sep-2025 a ago-2026): los filtros de la izquierda no afectan esta página.",
        icon="ℹ️")

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
comp = cargar_modelo("model_comparison_full")
comp = comp[comp["umbral_tipo"] == "costo (validación)"].set_index("modelo")
roc_global = float(comp.loc["LightGBM", "roc_auc"])

fuentes = {
    "Aerolínea": ("subgrupos_aerolinea", "Aerolínea"),
    "Aeropuerto (20 más grandes)": ("subgrupos_aeropuerto_de_origen", "Aeropuerto de origen"),
    "Franja horaria": ("subgrupos_franja_horaria", "Franja horaria"),
    "Temporada": ("subgrupos_temporada", "Temporada"),
}
señalados = cargar_modelo("subgrupos_senalados")

vista = st.segmented_control("Ver el desempeño del modelo por:", options=list(fuentes),
                             default="Aerolínea", key="vista_limites") or "Aerolínea"
archivo, col = fuentes[vista]
g = cargar_modelo(archivo).rename(columns={col: "grupo", "sesgo (pred/real)": "sesgo"})
if vista == "Aerolínea":
    g["nombre"] = g["grupo"].map(lambda c: f"{AEROLINEAS.get(c, c)} ({c})")
else:
    g["nombre"] = g["grupo"]
marcados_roc = set(señalados.loc[(señalados["dimensión"] == col)
                                 & señalados["motivo"].str.contains("ROC-AUC"), "subgrupo"])

col_a, col_b = st.columns(2, gap="large")

# ---------------------------------------------------------------------------
# Gráfica 1: capacidad de distinguir (ROC-AUC) por grupo
# ---------------------------------------------------------------------------
with col_a:
    t = g[["grupo", "nombre", "roc_auc", "vuelos"]].copy()
    t = pd.concat([t, pd.DataFrame([{"grupo": "_global", "nombre": "Global (todos los vuelos)",
                                     "roc_auc": roc_global, "vuelos": None}])], ignore_index=True)
    t = t.sort_values("roc_auc", ascending=False)      # el peor queda abajo... y lo resaltamos en rojo
    peor = t[t["grupo"] != "_global"].iloc[-1]
    roc_txt = lambda v: f"{v:.2f}".replace(".", ",")
    titulo_1 = (f"Donde el modelo distingue peor: {peor['nombre']}<br>"
                f"(ROC-AUC {roc_txt(peor['roc_auc'])} frente a {roc_txt(roc_global)} global)")
    colores = [GRIS if r.grupo == "_global" else (ROJO if r.grupo in marcados_roc
                                                     else GRIS_CLARO) for r in t.itertuples()]
    fig1 = go.Figure(go.Bar(
        y=t["nombre"], x=t["roc_auc"] - 0.5, base=0.5, orientation="h", marker_color=colores,
        text=[roc_txt(v) for v in t["roc_auc"]], textposition="outside", cliponaxis=False,
        constraintext="none", textfont=dict(size=11, color=TEXTO_SUAVE),
        hovertemplate="%{y}<br>ROC-AUC <b>%{customdata:.3f}</b><extra></extra>", customdata=t["roc_auc"],
    ))
    layout_base(fig1, titulo_1, alto=max(380, 27 * len(t) + 120))
    fig1.update_xaxes(range=[0.5, max(t["roc_auc"]) + 0.06], title="ROC-AUC (0,5 = adivinar · 1 = perfecto)",
                      showgrid=True, gridcolor="#f1f5f9", tickformat=".2f")
    fig1.update_yaxes(autorange="reversed", showgrid=False, title=None)
    fig1.update_layout(bargap=0.35, margin=dict(t=80))
    st.plotly_chart(fig1, width="stretch")
    st.caption("ROC-AUC = qué tan bien separa el modelo los vuelos que se retrasan de los que no. "
               "Las barras empiezan en 0,5 (adivinar). En rojo, los grupos que el equipo marcó por separar peor que el resto.")

# ---------------------------------------------------------------------------
# Gráfica 2: sesgo (¿predice más o menos riesgo del real?)
# ---------------------------------------------------------------------------
with col_b:
    s = g[["grupo", "nombre", "sesgo", "tasa_real", "tasa_predicha"]].copy()
    s["dif"] = s["sesgo"] - 1                                  # -0,21 = predice 21 % menos de lo real
    s = s.sort_values("dif")
    extremo = s.loc[s["dif"].abs().idxmax()]
    sentido = "subestima" if extremo["dif"] < 0 else "sobreestima"
    pct_ext = f"{abs(extremo['dif']) * 100:.0f}"
    titulo_2 = (f"{extremo['nombre']}: el modelo predice<br>"
                f"{pct_ext} % {'menos' if extremo['dif'] < 0 else 'más'} riesgo del que hubo")
    colores2 = [ROJO if abs(d) >= 0.10 else (AZUL if abs(d) >= 0.05 else GRIS_CLARO) for d in s["dif"]]
    fig2 = go.Figure()
    for r, c in zip(s.itertuples(), colores2):                  # "piruletas" desde el 0
        fig2.add_trace(go.Scatter(x=[0, r.dif], y=[r.nombre] * 2, mode="lines",
                                  line=dict(color=c, width=3), hoverinfo="skip", showlegend=False))
    fig2.add_trace(go.Scatter(
        x=s["dif"], y=s["nombre"], mode="markers", marker=dict(color=colores2, size=12),
        customdata=s[["tasa_real", "tasa_predicha"]].values, showlegend=False,
        hovertemplate="%{y}<br>Real %{customdata[0]:.1%} · Predicho %{customdata[1]:.1%}"
                      "<br>Diferencia <b>%{x:+.0%}</b><extra></extra>",
    ))
    fig2.add_vline(x=0, line_color=GRIS, line_width=1.5)
    lim = max(0.25, s["dif"].abs().max() * 1.25)
    fig2.add_annotation(x=-lim, y=1.02, yref="paper", xanchor="left", text="← predice menos riesgo",
                        showarrow=False, font=dict(color=TEXTO_SUAVE, size=11))
    fig2.add_annotation(x=lim, y=1.02, yref="paper", xanchor="right", text="predice más riesgo →",
                        showarrow=False, font=dict(color=TEXTO_SUAVE, size=11))
    layout_base(fig2, titulo_2, alto=max(380, 27 * len(s) + 120))
    fig2.update_xaxes(range=[-lim, lim], tickformat="+.0%", title="Diferencia entre lo predicho y lo real",
                      showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    fig2.update_yaxes(showgrid=False, title=None)
    fig2.update_layout(margin=dict(t=95))
    st.plotly_chart(fig2, width="stretch")
    st.caption("0 % = el modelo acierta en promedio el nivel de riesgo del grupo. En rojo, desvíos de 10 % o más; "
               "en azul, entre 5 % y 10 %. Subestimar es lo más delicado: el modelo dice que hay menos riesgo del real.")

# ---------------------------------------------------------------------------
# Tabla de grupos señalados por el equipo
# ---------------------------------------------------------------------------
st.subheader("Grupos señalados en el análisis del modelo")
tabla = señalados.copy()
tabla["subgrupo"] = [f"{AEROLINEAS.get(x, x)} ({x})" if d == "Aerolínea" else x
                     for d, x in zip(tabla["dimensión"], tabla["subgrupo"])]
tabla["motivo"] = (tabla["motivo"].str.replace("sesgo", "predicho/real")
                   .str.replace("subestima más que el resto", "predice menos riesgo del real")
                   .str.replace("sobreestima más que el resto", "predice más riesgo del real")
                   .str.replace("lift PR-AUC", "mejora vs. azar")
                   .str.replace(r"(\d)\.(\d)", r"\1,\2", regex=True))
tabla["vuelos"] = tabla["vuelos"].map(lambda v: f"{v:,.0f}".replace(",", "."))
st.dataframe(tabla.rename(columns={"dimensión": "Dimensión", "subgrupo": "Grupo", "vuelos": "Vuelos en prueba",
                                   "motivo": "Por qué se señala"}),
             hide_index=True, width="stretch", height=35 * (len(tabla) + 1) + 3)

with st.expander("¿Por qué casi todos los grupos tienen ROC-AUC más bajo que el global?"):
    st.markdown("""
Es esperado y no es un error. El número global se beneficia de diferencias **entre** grupos: por ejemplo,
separar vuelos de la mañana (pocos retrasos) de vuelos de la noche (muchos) es fácil. Dentro de un solo grupo
esa ayuda desaparece y queda la parte difícil: distinguir entre vuelos parecidos. Por eso lo que se compara es
qué grupo queda **mucho** más abajo que los demás. La madrugada es el caso claro: tiene pocos vuelos y
pocos datos del día anteriores a la salida.
""")

st.markdown(
    """<div class="mensaje">
    <b>Qué nos dice esta página y qué haría la aerolínea</b><br><br>
    1. <b>Madrugada:</b> el modelo distingue poco. Ahí la alerta se toma como orientativa, no como decisión.<br><br>
    2. <b>Hawaiian, Republic y Boston:</b> el modelo predice menos riesgo del real. Conviene bajar el umbral de
    alerta para esos grupos o revisarlos aparte.<br><br>
    3. <b>Charlotte (CLT):</b> pasa lo contrario, predice más riesgo del real: habrá más falsas alarmas.<br><br>
    4. <b>Siguiente paso:</b> la tasa de retraso subió en 2026, así que el modelo debe reentrenarse cada cierto tiempo
    y vigilarse (fase de retroalimentación de ASUM-DM). Agregar el pronóstico del clima es la mejora natural.
    </div>""",
    unsafe_allow_html=True,
)
st.markdown('<div class="fuente">Fuente: análisis por subgrupos de la prueba del modelo, notebook '
            'flight_delay_modeling.ipynb del equipo (outputs/outputs_modeling). Datos BTS.</div>',
            unsafe_allow_html=True)
