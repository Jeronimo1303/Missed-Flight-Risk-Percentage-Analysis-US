"""
Página 1 — Resumen del problema (CONTEXTO del arco narrativo).

Pregunta de negocio: ¿qué tan grande es el problema de los retrasos y cómo se comporta en el tiempo?
Audiencia: gerencia de operaciones de una aerolínea.
Acción esperada: entender que el retraso es frecuente, estacional y va en aumento,
y que por eso vale la pena anticiparlo (lo que hace el modelo, páginas siguientes).

Cada título es declarativo y se calcula con los datos filtrados, para que siga
siendo verdad cuando el usuario cambia los filtros.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.datos import aplicar_filtros, cargar_agg_mes, tasa
from utils.estilo import (AZUL, DORADO, FUENTE_DATOS, GRIS, GRIS_CLARO, MESES, MESES_LARGOS,
                          ROJO, TEXTO_SUAVE, VERDE, kpi, layout_base, miles, pct)

df = aplicar_filtros(cargar_agg_mes())

st.title("¿Qué tan grande es el problema de los retrasos?")
st.caption("Vuelos domésticos de EE. UU. · Un vuelo se cuenta como retrasado si sale 15 minutos o más "
           "después de su hora programada (definición oficial del Departamento de Transporte).")

if df.empty or df["vuelos"].sum() == 0:
    st.warning("No hay vuelos con esa combinación de filtros. Amplía el periodo o quita algún filtro.")
    st.stop()

# ---------------------------------------------------------------------------
# Cálculos
# ---------------------------------------------------------------------------
tasa_total = tasa(df)
mensual = (df.groupby("periodo", as_index=False)[["vuelos", "retrasados"]].sum()
             .assign(tasa=lambda d: d["retrasados"] / d["vuelos"]))

# Estacionalidad: tasa por mes del calendario (todos los años juntos)
por_mes = (df.groupby("mes", as_index=False)[["vuelos", "retrasados"]].sum()
             .assign(tasa=lambda d: d["retrasados"] / d["vuelos"]))
mes_pico = por_mes.loc[por_mes["tasa"].idxmax()]
mes_valle = por_mes.loc[por_mes["tasa"].idxmin()]

# Comparación justa entre años: solo los meses que tiene el último año del filtro.
ultimo_anio = int(df["anio"].max())
meses_ultimo = sorted(df.loc[df["anio"] == ultimo_anio, "mes"].unique())
mismo_tramo = (df[df["mes"].isin(meses_ultimo)]
               .groupby("anio", as_index=False)[["vuelos", "retrasados"]].sum()
               .assign(tasa=lambda d: d["retrasados"] / d["vuelos"]))
tramo_txt = (f"{MESES[meses_ultimo[0] - 1]}–{MESES[meses_ultimo[-1] - 1]}"
             if len(meses_ultimo) < 12 else "año completo")

# ---------------------------------------------------------------------------
# Fila de KPIs (arriba a la izquierda: lo primero que lee el ojo, patrón Z)
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi("Vuelos analizados", miles(df["vuelos"].sum()),
                f"{df['aerolinea'].nunique()} aerolíneas · {df['origen'].nunique()} aeropuertos"),
            unsafe_allow_html=True)
c2.markdown(kpi("Salen con 15+ min de retraso", pct(tasa_total),
                f"≈ 1 de cada {round(1 / tasa_total)} vuelos"),
            unsafe_allow_html=True)
c3.markdown(kpi("Mes más crítico", f"{MESES_LARGOS[int(mes_pico['mes']) - 1].capitalize()}",
                f"{pct(mes_pico['tasa'])} con retraso vs {pct(mes_valle['tasa'])} en "
                f"{MESES_LARGOS[int(mes_valle['mes']) - 1]}", clase="enfasis"),
            unsafe_allow_html=True)

if len(mismo_tramo) >= 2:
    actual = mismo_tramo.iloc[-1]
    previo = mismo_tramo.iloc[-2]
    delta_pp = (actual["tasa"] - previo["tasa"]) * 100
    signo = "+" if delta_pp >= 0 else "−"
    c4.markdown(kpi(f"{ultimo_anio} vs {int(previo['anio'])} ({tramo_txt})", pct(actual["tasa"]),
                    f"{signo}{abs(delta_pp):.1f} puntos frente a {pct(previo['tasa'])}".replace(".", ","),
                    clase="alerta" if delta_pp > 0 else ""),
                unsafe_allow_html=True)
else:
    c4.markdown(kpi("Comparación entre años", "—", "El periodo elegido tiene un solo año"),
                unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Gráfica 1: serie mensual (variable de tiempo -> línea)
# ---------------------------------------------------------------------------
titulo_1 = (f"Los retrasos tienen temporada: en {MESES_LARGOS[int(mes_pico['mes']) - 1]} sale tarde "
            f"{pct(mes_pico['tasa'], 0)} de los vuelos y en {MESES_LARGOS[int(mes_valle['mes']) - 1]} "
            f"solo {pct(mes_valle['tasa'], 0)}")

fig1 = go.Figure()
# Contexto: promedio del periodo como referencia
fig1.add_hline(y=tasa_total, line_dash="dot", line_color=GRIS, line_width=1.5)
fig1.add_annotation(x=1, xref="paper", xanchor="left", y=tasa_total, yref="y",
                    text=f"Promedio<br>{pct(tasa_total)}", showarrow=False,
                    font=dict(color=TEXTO_SUAVE, size=12), align="left", xshift=6)
# Datos: la serie en azul
fig1.add_trace(go.Scatter(
    x=mensual["periodo"], y=mensual["tasa"], mode="lines",
    line=dict(color=AZUL, width=2.5),
    customdata=mensual[["vuelos"]],
    hovertemplate="%{x|%b %Y}<br><b>%{y:.1%}</b> con retraso<br>%{customdata[0]:,.0f} vuelos<extra></extra>",
))
# Contraste: el pico de cada año en dorado
picos = mensual.loc[mensual.groupby(mensual["periodo"].dt.year)["tasa"].idxmax()]
fig1.add_trace(go.Scatter(
    x=picos["periodo"], y=picos["tasa"], mode="markers",
    marker=dict(color=DORADO, size=11, line=dict(color="white", width=1.5)),
    hovertemplate="Pico del año: %{x|%b %Y}<br><b>%{y:.1%}</b><extra></extra>",
))
# Anotación del máximo absoluto (storytelling directo sobre la gráfica)
maximo = mensual.loc[mensual["tasa"].idxmax()]
fig1.add_annotation(
    x=maximo["periodo"], y=maximo["tasa"],
    text=f"<b>{MESES[maximo['periodo'].month - 1]} {maximo['periodo'].year}: {pct(maximo['tasa'])}</b><br>"
         f"más de 1 de cada 4 vuelos tarde" if maximo["tasa"] > 0.25 else
         f"<b>{MESES[maximo['periodo'].month - 1]} {maximo['periodo'].year}: {pct(maximo['tasa'])}</b>",
    showarrow=True, arrowhead=2, arrowcolor=DORADO, ax=0, ay=-45,
    font=dict(color=AZUL, size=12), align="center",
)
layout_base(fig1, titulo_1, alto=440)
fig1.update_yaxes(title="% de vuelos con 15+ min de retraso", tickformat=".0%",
                  range=[0, max(0.35, mensual["tasa"].max() * 1.15)])
ticks = mensual["periodo"][mensual["periodo"].dt.month.isin([1, 7])]
fig1.update_xaxes(title=None, tickvals=ticks,
                  ticktext=[f"{MESES[t.month - 1]} {t.year}" for t in ticks])
fig1.update_layout(margin=dict(r=90))
st.plotly_chart(fig1, width="stretch")

# ---------------------------------------------------------------------------
# Gráfica 2: comparación justa entre años (mismos meses) + mensaje
# ---------------------------------------------------------------------------
col_g, col_t = st.columns([3, 2], gap="large")

with col_g:
    if len(mismo_tramo) >= 2:
        es_max = mismo_tramo["tasa"].idxmax() == mismo_tramo.index[-1]
        titulo_2 = (f"{ultimo_anio} va en el peor ritmo del periodo:<br>"
                    f"{pct(mismo_tramo.iloc[-1]['tasa'])} de vuelos tarde entre {tramo_txt.replace('–', ' y ')}"
                    if es_max else f"Tasa de retraso por año,<br>comparando solo {tramo_txt}")
        colores = [GRIS_CLARO] * (len(mismo_tramo) - 1) + [DORADO if es_max else AZUL]
        fig2 = go.Figure(go.Bar(
            x=mismo_tramo["anio"].astype(str), y=mismo_tramo["tasa"],
            marker_color=colores, width=0.6,
            text=[pct(t) for t in mismo_tramo["tasa"]], textposition="outside",
            textfont=dict(color=AZUL, size=13),
            hovertemplate="%{x}: <b>%{y:.1%}</b><extra></extra>",
        ))
        anteriores = mismo_tramo.iloc[:-1]
        ref = tasa(anteriores)
        fig2.add_hline(y=ref, line_dash="dot", line_color=GRIS, line_width=1.5)
        fig2.add_annotation(x=1, xref="paper", xanchor="left", y=ref, yref="y",
                            text=f"Años<br>anteriores<br>{pct(ref)}", showarrow=False,
                            font=dict(color=TEXTO_SUAVE, size=11), align="left", xshift=6)
        layout_base(fig2, titulo_2, alto=420)
        fig2.update_layout(margin=dict(r=80, t=80))
        fig2.update_yaxes(title=f"% de vuelos con retraso ({tramo_txt})", tickformat=".0%",
                          range=[0, mismo_tramo["tasa"].max() * 1.25])
        fig2.update_xaxes(title=None, type="category")
        st.plotly_chart(fig2, width="stretch")
        st.caption(f"Solo se comparan los meses {tramo_txt} de cada año porque el último año tiene datos "
                   "hasta ese mes y el verano concentra los retrasos: compararlo contra años completos "
                   "lo haría ver peor de lo que es.")
    else:
        st.info("Selecciona un periodo con al menos dos años para comparar entre años.")

with col_t:
    st.write("")
    st.write("")
    st.markdown(
        f"""<div class="mensaje">
        <b>Qué nos dice esta página</b><br><br>
        1. El retraso no es raro: <b>{pct(tasa_total)}</b> de los vuelos sale 15 minutos tarde o más.<br><br>
        2. Tiene temporada: <b>{MESES_LARGOS[int(mes_pico['mes']) - 1]}</b> es el peor mes y
        <b>{MESES_LARGOS[int(mes_valle['mes']) - 1]}</b> el mejor.<br><br>
        3. Por eso no basta con mirar el histórico: hay que <b>anticipar qué vuelo concreto</b>
        tiene más riesgo. Eso es lo que hace el modelo (páginas siguientes).
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown(f'<div class="fuente">{FUENTE_DATOS}</div>', unsafe_allow_html=True)
