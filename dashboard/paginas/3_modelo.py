"""
Página 3 — ¿Podemos confiar en el modelo? (inicio de la RESOLUCIÓN).

Pregunta de negocio: si la aerolínea usa el modelo para decidir qué vuelos vigilar,
¿qué tan seguido acierta y cuánto mejor es que no usarlo?
Audiencia: gerencia de operaciones (y el jurado).
Acción esperada: confiar en la lista de riesgo y decidir cuántos vuelos revisar al día.

Datos: resultados de la prueba del notebook flight_delay_modeling.ipynb
(outputs/outputs_modeling). Prueba = 1.000.000 de vuelos de sep-2025 a ago-2026
que el modelo nunca vio. Los filtros de la barra lateral no aplican aquí.
"""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.datos import DATA, cargar_modelo
from utils.estilo import (AZUL, DORADO, GRIS, GRIS_CLARO, TEXTO_SUAVE, kpi, layout_base,
                          pct)

st.title("¿Podemos confiar en el modelo?")
st.caption("Prueba con 1.000.000 de vuelos de septiembre de 2025 a agosto de 2026 que el modelo "
           "nunca vio durante su entrenamiento. Predicción hecha 4 horas antes de cada salida.")
st.info("Esta página muestra el resultado de la prueba del modelo: los filtros de la izquierda no la afectan.",
        icon="ℹ️")

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
topk = cargar_modelo("top_k_priorizacion")                       # 1, 5, 10, 20, 30 %
comp = cargar_modelo("model_comparison_full")
# La corrida final en AWS omitió el estudio de ablación (memoria); se conserva el de la corrida local,
# que usa el mismo LightGBM (PR-AUC 0,545 en ambas).
_abl = DATA / "modelo" / "ablacion_corrida_local.csv"
if _abl.exists() and not comp["modelo"].str.startswith("[").any():
    comp = pd.concat([comp, pd.read_csv(_abl)], ignore_index=True)
comp = comp[comp["umbral_tipo"] == "costo (validación)"].set_index("modelo")
config = json.loads((DATA / "modelo" / "run_config.json").read_text(encoding="utf-8"))
azar = float((topk["precision"] / topk["lift"]).mean())          # tasa base del tramo de prueba
franja = cargar_modelo("subgrupos_franja_horaria")
ELEGIDO = "LightGBM"                      # modelo que el equipo puso en producción
# Tiempo de entrenamiento de la corrida final en AWS SageMaker (21-nov-2026, mismos datos para ambos)
TIEMPO_AWS_S = {"LightGBM": 716.6, "XGBoost": 2516.6}
temporada = cargar_modelo("subgrupos_temporada")

# ---------------------------------------------------------------------------
# Control: ¿cuántos vuelos puede revisar el equipo?
# ---------------------------------------------------------------------------
opciones = {f"{int(k * 100)} %": k for k in topk["top_k_%"]}
eleccion = st.segmented_control(
    "¿Qué porcentaje de los vuelos del día puede revisar tu equipo de operaciones?",
    options=list(opciones), default="10 %", key="presupuesto",
)
k = opciones.get(eleccion or "10 %")
fila = topk[topk["top_k_%"] == k].iloc[0]
de_100 = round(fila["precision"] * 100)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi(f"De los vuelos que revisas ({eleccion or '10 %'} más riesgoso)",
                f"{de_100} de cada 100", "sí salen con retraso", clase="enfasis"), unsafe_allow_html=True)
c2.markdown(kpi("Si escogieras vuelos al azar", f"{round(azar * 100)} de cada 100",
                "saldrían con retraso"), unsafe_allow_html=True)
c3.markdown(kpi("Retrasos que alcanzas a anticipar", pct(fila["recall_capturado"], 0),
                f"de todos los retrasos, revisando solo el {eleccion or '10 %'} de los vuelos"),
            unsafe_allow_html=True)
c4.markdown(kpi("Mejora frente al azar", f"{fila['lift']:.1f} veces".replace(".", ","),
                "más certero que escoger vuelos sin modelo"), unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------------------
# Gráfica 1: certeza según cuántos vuelos se revisan (la gráfica de decisión)
# ---------------------------------------------------------------------------
titulo_1 = (f"Si tu equipo revisa el {int(k * 100)} % de vuelos más riesgosos, {de_100} de cada 100 "
            f"sí se retrasan; al azar serían {round(azar * 100)}")
etiquetas = [f"{int(x * 100)} % más riesgoso" for x in topk["top_k_%"]]
fig1 = go.Figure(go.Bar(
    x=etiquetas, y=topk["precision"], width=0.55,
    marker_color=[DORADO if x == k else GRIS_CLARO for x in topk["top_k_%"]],
    text=[f"<b>{round(p * 100)} de cada 100</b><br>atrapa {pct(r, 0)} de los retrasos"
          for p, r in zip(topk["precision"], topk["recall_capturado"])],
    textposition="outside", textfont=dict(size=13, color=AZUL), cliponaxis=False,
    constraintext="none", customdata=topk["vuelos"],
    hovertemplate="%{x}: %{customdata:,.0f} vuelos revisados<br><b>%{y:.1%}</b> se retrasan<extra></extra>",
))
fig1.add_hline(y=azar, line_dash="dot", line_color=GRIS, line_width=1.5)
fig1.add_annotation(x=1, xref="paper", xanchor="left", y=azar, yref="y",
                    text=f"Al azar<br>{pct(azar)}", showarrow=False,
                    font=dict(color=TEXTO_SUAVE, size=12), align="left", xshift=6)
layout_base(fig1, titulo_1, alto=460)
fig1.update_layout(margin=dict(r=90, t=80), bargap=0.4)
fig1.update_yaxes(title="% de los vuelos revisados que sí se retrasan", tickformat=".0%", range=[0, 1.12])
fig1.update_xaxes(title="Vuelos que revisa el equipo (los de mayor riesgo según el modelo)")
st.plotly_chart(fig1, width="stretch")
st.caption("El intercambio: mientras menos vuelos revises, más certera es cada alerta, pero atrapas una "
           "parte menor del total de retrasos. La aerolínea escoge el punto según cuánta gente tenga.")
with st.expander("⚠️ Alerta de rigor: los porcentajes del 1 % y 5 % pueden estar inflados"):
    st.markdown("""
Al revisar la lista de vuelos más riesgosos encontramos que casi todos los del **1 % superior** tienen una
**rotación imposible**: el avión que voló estaba programado para llegar de su vuelo anterior *después* de esta
salida (o con menos de 20 minutos). Eso solo ocurre cuando la aerolínea cambia de avión durante el día, y ese
cambio **puede no conocerse 4 horas antes**: sería una fuga de información a través del número de avión.

Prueba de sensibilidad con las predicciones de agosto de 2026 (modelo reproducido con el código del equipo):

| | Todos los vuelos | Sin rotación imposible (2,5 % menos vuelos) |
|---|---|---|
| ROC-AUC | 0,771 | 0,755 |
| PR-AUC | 0,598 (base 0,268) | 0,535 (base 0,253) |
| Precisión del 1 % más riesgoso | 96 % | 86 % |
| Precisión del 10 % más riesgoso | 73 % | 65 % |

**Conclusión:** el modelo sigue siendo útil sin esos vuelos (2,6 veces mejor que el azar en el 10 % superior),
pero las cifras del 1 % y 5 % deben presentarse con esta advertencia. Recomendación al equipo: reentrenar
excluyendo o corrigiendo las variables de rotación cuando el margen es menor a 20 minutos o negativo.
""")

# ---------------------------------------------------------------------------
# Gráficas 2 y 3: ¿por qué este modelo? y ¿qué aprende?
# ---------------------------------------------------------------------------
col_a, col_b = st.columns(2, gap="large")

with col_a:
    candidatos = ["Regresión logística", "LightGBM", "XGBoost", "K-means (tasa por clúster)"]
    nombres = {"Regresión logística": "Regresión logística (base)", "LightGBM": "LightGBM",
               "XGBoost": "XGBoost", "K-means (tasa por clúster)": "K-means"}
    m = comp.loc[candidatos, ["pr_auc"]].reset_index()
    m["nombre"] = m["modelo"].map(nombres)
    m = pd.concat([m, pd.DataFrame([{"nombre": "Adivinar al azar", "pr_auc": azar}])],
                  ignore_index=True).sort_values("pr_auc")
    mejor = comp.loc[candidatos, "pr_auc"].idxmax()
    dif = comp.loc[mejor, "pr_auc"] - comp.loc[ELEGIDO, "pr_auc"]
    veces = comp.loc[ELEGIDO, "pr_auc"] / azar
    lento = TIEMPO_AWS_S["XGBoost"] / TIEMPO_AWS_S["LightGBM"]
    if mejor == ELEGIDO:
        titulo_2 = (f"{ELEGIDO} fue el mejor de 4 modelos:<br>encuentra los retrasos "
                    f"{veces:.1f} veces mejor que adivinar".replace(".", ","))
    else:   # decisión del equipo: casi el mismo acierto con mucho menos tiempo de entrenamiento
        titulo_2 = (f"{mejor} acierta apenas {dif:.3f} más, pero tarda<br>"
                    f"{lento:.1f} veces más en entrenar: escogimos {ELEGIDO}".replace(".", ","))
    fig2 = go.Figure(go.Bar(
        y=m["nombre"], x=m["pr_auc"], orientation="h",
        marker_color=[DORADO if n == ELEGIDO else (GRIS if n == "Adivinar al azar" else GRIS_CLARO)
                      for n in m["nombre"]],
        text=[f"{v:.3f}".replace(".", ",") for v in m["pr_auc"]], textposition="outside",
        textfont=dict(size=12, color=TEXTO_SUAVE), cliponaxis=False,
        hovertemplate="%{y}<br>PR-AUC <b>%{x:.3f}</b><extra></extra>",
    ))
    layout_base(fig2, titulo_2, alto=380)
    fig2.update_xaxes(range=[0, m["pr_auc"].max() * 1.3], showgrid=False, showticklabels=False,
                      showline=False, title="PR-AUC (más es mejor)")
    fig2.update_yaxes(showgrid=False, title=None)
    fig2.update_layout(bargap=0.4, margin=dict(t=80))
    st.plotly_chart(fig2, width="stretch")
    dif_xgb = f"{abs(comp.loc['LightGBM', 'pr_auc'] - comp.loc['XGBoost', 'pr_auc']):.3f}".replace(".", ",")
    seg = {m_: f"{v:,.0f}".replace(",", ".") for m_, v in TIEMPO_AWS_S.items()}
    lento_txt = f"{lento:.1f}".replace(".", ",")
    st.caption("PR-AUC mide qué tan bien el modelo pone los vuelos que se retrasan arriba de la lista; "
               f"adivinar da la tasa base. LightGBM y XGBoost quedan casi empatados ({dif_xgb} de diferencia). "
               f"En la corrida final en AWS, XGBoost tardó {seg['XGBoost']} s en entrenar y LightGBM "
               f"{seg['LightGBM']} s ({lento_txt} veces menos). Por una diferencia tan pequeña, el equipo "
               "escogió el modelo más rápido de entrenar y actualizar.")

PASOS_ABLACION = [
    ("Solo datos<br>fijos", "[ablación] LightGBM estáticas"),
    ("+ historial<br>de retrasos", "[ablación] LightGBM + historial"),
    ("+ rotación<br>del avión", "[ablación] LightGBM + horario/rotación"),
    ("+ cómo va<br>el día", "LightGBM"),
]
hay_ablacion = all(c in comp.index for _, c in PASOS_ABLACION)

with col_b:
    if not hay_ablacion:   # en producción (AWS) se omitió para ahorrar memoria
        st.info("Esta corrida del modelo no incluye el estudio de ablación (se omitió en producción para "
                "ahorrar memoria en AWS). En la corrida completa, lo que más mejoró la predicción fue "
                "conocer la rotación del avión.", icon="ℹ️")
    else:
        pasos = PASOS_ABLACION
        vals = [comp.loc[c, "pr_auc"] for _, c in pasos]
        saltos = [None] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        i_max = max(range(1, len(vals)), key=lambda i: saltos[i])
        titulo_3 = ("Lo que más mejora la predicción es conocer<br>"
                    "la rotación del avión y cómo viene su día")
        fig3 = go.Figure(go.Bar(
            x=[p for p, _ in pasos], y=vals, width=0.55,
            marker_color=[DORADO if i == i_max else (AZUL if i == len(vals) - 1 else GRIS_CLARO)
                          for i in range(len(vals))],
            text=[f"{v:.3f}".replace(".", ",") + ("" if s is None else f"<br>(+{s:.3f})".replace(".", ","))
                  for v, s in zip(vals, saltos)],
            textposition="outside", textfont=dict(size=12, color=TEXTO_SUAVE), cliponaxis=False,
            constraintext="none",
            hovertemplate="%{x}<br>PR-AUC <b>%{y:.3f}</b><extra></extra>",
        ))
        fig3.add_hline(y=azar, line_dash="dot", line_color=GRIS, line_width=1.5)
        fig3.add_annotation(x=0, xref="paper", xanchor="left", y=azar, yref="y", yshift=10,
                            text=f"Adivinar: {azar:.3f}".replace(".", ","), showarrow=False,
                            font=dict(color=TEXTO_SUAVE, size=11))
        layout_base(fig3, titulo_3, alto=380)
        fig3.update_yaxes(title="PR-AUC", range=[0, max(vals) * 1.3], showgrid=False, showticklabels=False)
        fig3.update_xaxes(tickfont=dict(size=11), tickangle=0)
        fig3.update_layout(margin=dict(t=80))
        st.plotly_chart(fig3, width="stretch")
        st.caption("Estudio de ablación (corrida local del equipo): el mismo LightGBM entrenado agregando un grupo de variables a la vez. "
                   "Datos fijos = ruta, aerolínea, hora y fecha. Historial = tasas de retraso pasadas por ruta, "
                   "aeropuerto, aerolínea y avión. Rotación = congestión programada del aeropuerto, número de tramo "
                   "del avión en el día y tiempo entre tramos. "
                   "Cómo va el día = lo que ya pasó ese día en el avión y el aeropuerto. Todo se conoce 4 horas "
                   "antes de la salida.")

# ---------------------------------------------------------------------------
# Gráficas 4 y 5: ¿las probabilidades dicen la verdad? Una idea por gráfica:
# primero por hora del día y luego por temporada (antes estaban juntas en una sola).
# ---------------------------------------------------------------------------
def pesas(df: pd.DataFrame, col: str, titulo: str, x_max: float) -> go.Figure:
    """Gráfico de pesas: punto azul = lo que pasó, rombo dorado = lo que el modelo esperaba."""
    d = df.rename(columns={col: "grupo"}).iloc[::-1].reset_index(drop=True)
    fig = go.Figure()
    for _, r in d.iterrows():
        fig.add_trace(go.Scatter(x=[r["tasa_real"], r["tasa_predicha"]], y=[r["grupo"]] * 2, mode="lines",
                                 line=dict(color=GRIS_CLARO, width=4), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=d["tasa_real"], y=d["grupo"], mode="markers", name="Real",
                             marker=dict(color=AZUL, size=13),
                             hovertemplate="%{y}<br>Real: <b>%{x:.1%}</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=d["tasa_predicha"], y=d["grupo"], mode="markers", name="Predicho",
                             marker=dict(color=DORADO, size=13, symbol="diamond"),
                             hovertemplate="%{y}<br>Predicho: <b>%{x:.1%}</b><extra></extra>"))
    layout_base(fig, titulo, alto=340)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, xanchor="left"),
                      margin=dict(t=110))
    fig.update_xaxes(title="% de vuelos con retraso", tickformat=".0%", showgrid=True, gridcolor="#f1f5f9",
                     range=[0, x_max])
    fig.update_yaxes(showgrid=False, title=None)
    return fig


def dif_max(df: pd.DataFrame) -> str:
    return f"{(df['tasa_predicha'] - df['tasa_real']).abs().max() * 100:.1f}".replace(".", ",")


st.subheader("¿Las probabilidades dicen la verdad?")
st.caption("Si el modelo dice 30 %, deberían retrasarse cerca de 30 de cada 100 vuelos. "
           "Azul = lo que pasó; dorado = lo que el modelo esperaba en promedio.")
x_max = max(franja[["tasa_real", "tasa_predicha"]].max().max(),
            temporada[["tasa_real", "tasa_predicha"]].max().max()) * 1.15   # misma escala en ambas
col_c, col_d = st.columns(2, gap="large")
with col_c:
    st.plotly_chart(pesas(franja, "Franja horaria",
                          f"Por hora del día: lo predicho y lo real<br>difieren a lo sumo {dif_max(franja)} puntos",
                          x_max), width="stretch")
with col_d:
    st.plotly_chart(pesas(temporada, "Temporada",
                          f"Por temporada: lo predicho y lo real<br>difieren a lo sumo {dif_max(temporada)} puntos",
                          x_max), width="stretch")
ece_txt = f"{comp.loc[ELEGIDO, 'ece'] * 100:.2f}".replace(".", ",")
st.caption("Puntos casi encima uno del otro = probabilidades confiables. "
           f"Error de calibración global (ECE): {ece_txt} %.")

# ---------------------------------------------------------------------------
# Glosario para la sustentación y mensaje
# ---------------------------------------------------------------------------
ponderado_txt = (f" (ECE de {comp.loc['[experimento] LightGBM ponderado', 'ece']:.3f} con pesos vs "
                 f"{comp.loc['LightGBM', 'ece']:.3f} sin pesos)"
                 if "[experimento] LightGBM ponderado" in comp.index else "")
with st.expander("¿Qué significa cada métrica? (explicación sencilla)"):
    st.markdown(f"""
- **Precisión del top-k:** de los vuelos que el modelo marcó como más riesgosos, cuántos sí se retrasaron.
  Es la métrica que le importa al equipo que revisa la lista.
- **Cobertura (recall):** de todos los vuelos que se retrasaron, qué parte estaba en la lista revisada.
- **PR-AUC:** resume en un número qué tan bien el modelo pone arriba los vuelos que se retrasan, sin escoger
  un umbral. Adivinar al azar da la tasa base ({azar:.3f}). Se usa en vez de *accuracy* porque solo 23 de cada
  100 vuelos se retrasan: un modelo que diga siempre "no se retrasa" tendría 77 % de accuracy y no serviría.
- **Calibración (ECE):** si el modelo dice 30 %, ¿se retrasan 30 de cada 100? Un ECE cercano a 0 dice que sí.
  Por eso no se balancearon las clases: hacerlo deja el PR-AUC igual pero daña la calibración{ponderado_txt}.
- **Umbral por costo:** se supuso que no anticipar un retraso cuesta {config['cost_fn']} veces lo que una
  falsa alarma; con eso el umbral queda en {comp.loc['LightGBM', 'umbral']:.2f} de probabilidad.
""".replace("0.", "0,"))

st.markdown(
    f"""<div class="mensaje">
    <b>Qué nos dice esta página</b><br><br>
    1. El modelo <b>sí sirve para priorizar</b>: revisando el 10 % de vuelos más riesgosos, 2 de cada 3 se
    retrasan, cuando al azar sería menos de 1 de cada 4.<br><br>
    2. Sus probabilidades <b>se pueden creer</b>: lo que predice y lo que pasa casi coinciden en cada franja y temporada.<br><br>
    3. No es perfecto: revisando el 10 % se anticipa cerca de 3 de cada 10 retrasos. La siguiente página muestra
    cómo se usaría en un día concreto.
    </div>""",
    unsafe_allow_html=True,
)
st.markdown('<div class="fuente">Fuente: prueba del modelo, notebook flight_delay_modeling.ipynb del equipo '
            '(outputs/outputs_modeling). Datos BTS.</div>', unsafe_allow_html=True)
