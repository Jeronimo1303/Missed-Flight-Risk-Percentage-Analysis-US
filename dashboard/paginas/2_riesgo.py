"""
Página 2 — ¿Dónde se concentra el riesgo? (CONFLICTO del arco narrativo).

Pregunta de negocio: ¿a qué horas, en qué días, con qué aerolíneas y en qué aeropuertos
se concentran los retrasos?
Audiencia: gerencia de operaciones.
Acción esperada: reconocer que el riesgo no está repartido por igual y que hay
patrones claros (sobre todo la hora), que son justamente los que aprende el modelo.

Gramática visual usada (curso, módulo "Tipos de variables"):
- Hora de salida  -> variable ordinal  -> barras en su orden natural.
- Día x hora      -> dos variables cruzadas -> mapa de calor.
- Aerolínea y aeropuerto -> nominales -> barras horizontales ordenadas de mayor a menor.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.datos import (aplicar_filtros, cargar_agg_dia_hora, cargar_agg_hora, tasa,
                         tasa_por)
from utils.estilo import (AEROLINEAS, AZUL, DIAS, DORADO, FUENTE_DATOS, GRIS, GRIS_CLARO,
                          TEXTO_SUAVE, layout_base, miles, pct)

MIN_PART_HORA = 0.005    # horas con menos del 0,5 % de los vuelos se muestran como poco confiables
MIN_VUELOS_AER = 2_000   # aerolíneas con menos vuelos en el filtro no se comparan


def barras_h(tabla, col_nombre, destacar, prom, titulo, etiqueta_prom="Promedio general"):
    """Barras horizontales ordenadas de menor a mayor (la peor queda arriba).
    El promedio entra como una barra gris más, en su lugar del orden: así el lector
    ve de un vistazo quién está por encima y quién por debajo, sin una línea que
    se cruce con las etiquetas."""
    t = tabla[[col_nombre, "tasa", "vuelos"]].copy()
    t = pd.concat([t, pd.DataFrame([{col_nombre: etiqueta_prom, "tasa": prom, "vuelos": np.nan}])],
                  ignore_index=True).sort_values("tasa")
    colores = [DORADO if n == destacar else (GRIS if n == etiqueta_prom else GRIS_CLARO)
               for n in t[col_nombre]]
    textos = [f"<b>{pct(r.tasa)}</b>" if r[col_nombre] == etiqueta_prom
              else f"{pct(r.tasa)} · {compacto(r.vuelos)}" for _, r in t.iterrows()]
    fig = go.Figure(go.Bar(
        y=t[col_nombre], x=t["tasa"], orientation="h", marker_color=colores,
        text=textos, textposition="outside", textfont=dict(size=11, color=TEXTO_SUAVE),
        cliponaxis=False,
        hovertemplate="%{y}<br><b>%{x:.1%}</b> con retraso<extra></extra>",
    ))
    layout_base(fig, titulo, alto=max(420, 27 * len(t) + 110))
    fig.update_xaxes(range=[0, t["tasa"].max() * 1.5], showgrid=False, showticklabels=False,
                     showline=False)
    fig.update_yaxes(showgrid=False, title=None)
    fig.update_layout(bargap=0.35, margin=dict(t=80, r=10))
    return fig


def compacto(v):
    """1.234.567 -> '1,2 M vuelos'; 868.122 -> '868 mil vuelos'."""
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}".replace(".", ",") + " M vuelos"
    return f"{v / 1000:.0f} mil vuelos"


st.title("¿Dónde se concentra el riesgo?")
st.caption("El retraso no está repartido por igual. Esta página muestra en qué horas, días, "
           "aerolíneas y aeropuertos se concentra.")

df_h = aplicar_filtros(cargar_agg_hora())
if df_h.empty or df_h["vuelos"].sum() == 0:
    st.warning("No hay vuelos con esa combinación de filtros. Amplía el periodo o quita algún filtro.")
    st.stop()

promedio = tasa(df_h)
hhmm = lambda h: f"{int(h):02d}:00"

# ---------------------------------------------------------------------------
# 1. Hora del día (ordinal -> barras en orden natural)
# ---------------------------------------------------------------------------
por_hora = tasa_por(df_h, "hora").set_index("hora").reindex(range(24)).fillna(0).reset_index()
por_hora["participacion"] = por_hora["vuelos"] / por_hora["vuelos"].sum()
confiables = por_hora[por_hora["participacion"] >= MIN_PART_HORA]
h_pico = confiables.loc[confiables["tasa"].idxmax()]
h_valle = confiables.loc[confiables["tasa"].idxmin()]
veces = h_pico["tasa"] / h_valle["tasa"] if h_valle["tasa"] > 0 else np.nan

titulo_hora = (f"El retraso se acumula durante el día: a las {hhmm(h_pico['hora'])} sale tarde "
               f"{pct(h_pico['tasa'], 0)} de los vuelos; a las {hhmm(h_valle['hora'])}, solo "
               f"{pct(h_valle['tasa'], 0)}")

colores = []
for _, r in por_hora.iterrows():
    if r["participacion"] < MIN_PART_HORA:
        colores.append("#eef2f6")      # casi sin vuelos: no se interpreta
    elif r["hora"] == h_pico["hora"]:
        colores.append(DORADO)         # el hallazgo
    else:
        colores.append(AZUL)

fig_h = go.Figure(go.Bar(
    x=[hhmm(h) for h in por_hora["hora"]], y=por_hora["tasa"], marker_color=colores, width=0.62,
    customdata=np.stack([por_hora["vuelos"], por_hora["participacion"]], axis=1),
    hovertemplate="Salida programada %{x}<br><b>%{y:.1%}</b> con retraso"
                  "<br>%{customdata[0]:,.0f} vuelos (%{customdata[1]:.1%} del total)<extra></extra>",
))
fig_h.add_hline(y=promedio, line_dash="dot", line_color=GRIS, line_width=1.5)
fig_h.add_annotation(x=1, xref="paper", xanchor="left", y=promedio, yref="y",
                     text=f"Promedio<br>{pct(promedio)}", showarrow=False,
                     font=dict(color=TEXTO_SUAVE, size=12), align="left", xshift=6)
fig_h.add_annotation(x=hhmm(h_pico["hora"]), y=h_pico["tasa"],
                     text=(f"<b>{pct(h_pico['tasa'])}</b><br>1 de cada {round(1 / h_pico['tasa'])}" if h_pico['tasa'] > 0 else f"<b>{pct(h_pico['tasa'])}</b>"),
                     showarrow=True, arrowhead=2, arrowcolor=DORADO, ax=0, ay=-38,
                     font=dict(color=AZUL, size=12))
fig_h.add_annotation(x=hhmm(h_valle["hora"]), y=h_valle["tasa"],
                     text=(f"<b>{pct(h_valle['tasa'])}</b><br>1 de cada {round(1 / h_valle['tasa'])}" if h_valle['tasa'] > 0 else f"<b>{pct(h_valle['tasa'])}</b>"),
                     showarrow=True, arrowhead=2, arrowcolor=GRIS, ax=0, ay=-38,
                     font=dict(color=TEXTO_SUAVE, size=12))
layout_base(fig_h, titulo_hora, alto=430)
fig_h.update_layout(margin=dict(r=90), bargap=0.35)
fig_h.update_yaxes(title="% de vuelos con 15+ min de retraso", tickformat=".0%",
                   range=[0, max(confiables["tasa"].max() * 1.3, promedio * 1.5)])
fig_h.update_xaxes(title="Hora programada de salida (hora local del aeropuerto)", tickangle=0,
                   tickvals=[hhmm(h) for h in range(0, 24, 2)])
st.plotly_chart(fig_h, width="stretch")
veces_txt = f"{veces:.1f}".replace(".", ",")
frase_veces = (f"Entre la hora más tranquila y la más crítica, el riesgo se multiplica por {veces_txt}. "
               if np.isfinite(veces) else "")
st.caption(frase_veces +
           f"Las barras casi blancas (madrugada) tienen menos del {pct(MIN_PART_HORA)} de los vuelos: "
           "con tan pocos datos su porcentaje no es confiable y no se usa para las conclusiones.")

# ---------------------------------------------------------------------------
# 2. Día de la semana x hora (dos variables cruzadas -> mapa de calor)
# ---------------------------------------------------------------------------
df_dh = aplicar_filtros(cargar_agg_dia_hora())
horas_ok = confiables["hora"].astype(int).tolist()
mat = tasa_por(df_dh[df_dh["hora"].isin(horas_ok)], ["dia_semana", "hora"])
piv = mat.pivot(index="dia_semana", columns="hora", values="tasa").reindex(index=range(1, 8))
piv_v = mat.pivot(index="dia_semana", columns="hora", values="vuelos").reindex(index=range(1, 8))
celda = mat.loc[mat["tasa"].idxmax()]
dia_pico, hora_celda = DIAS[int(celda["dia_semana"]) - 1], hhmm(celda["hora"])

titulo_calor = (f"La franja más crítica de la semana: {dia_pico} a las {hora_celda}, "
                f"con {pct(celda['tasa'], 0)} de vuelos tarde")

fig_c = go.Figure(go.Heatmap(
    z=piv.values, x=[hhmm(h) for h in piv.columns], y=[d.capitalize() for d in DIAS],
    customdata=piv_v.values,
    colorscale=[[0, "#f8fafc"], [0.5, "#94a3b8"], [1, AZUL]],
    zmin=float(np.nanmin(piv.values)), zmax=float(np.nanmax(piv.values)),
    xgap=2, ygap=2,
    colorbar=dict(title="% tarde", tickformat=".0%", thickness=12, len=0.8),
    hovertemplate="%{y} %{x}<br><b>%{z:.1%}</b> con retraso<br>%{customdata:,.0f} vuelos<extra></extra>",
))
# Contraste: la celda crítica enmarcada en dorado
fig_c.add_shape(type="rect", xref="x", yref="y",
                x0=list(piv.columns).index(celda["hora"]) - 0.5,
                x1=list(piv.columns).index(celda["hora"]) + 0.5,
                y0=int(celda["dia_semana"]) - 1.5, y1=int(celda["dia_semana"]) - 0.5,
                line=dict(color=DORADO, width=3))
layout_base(fig_c, titulo_calor, alto=400)
fig_c.update_yaxes(autorange="reversed", showgrid=False, title=None)
fig_c.update_xaxes(title="Hora programada de salida", showline=False)
st.plotly_chart(fig_c, width="stretch")
if st.session_state.get("f_origenes"):
    st.caption("Nota: este mapa respeta el periodo y la aerolínea, pero no el filtro de aeropuerto "
               "(se calcula sobre todos los aeropuertos).")

# ---------------------------------------------------------------------------
# 3. Aerolíneas y aeropuertos (nominales -> barras horizontales ordenadas)
#    Drill-down: al hacer clic en una aerolínea, la gráfica de aeropuertos
#    se recalcula solo para esa aerolínea.
# ---------------------------------------------------------------------------
st.write("")
col_a, col_b = st.columns(2, gap="large")

with col_a:
    aer = tasa_por(df_h, "aerolinea")
    aer = aer[aer["vuelos"] >= MIN_VUELOS_AER].sort_values("tasa")
    if len(aer) >= 2:
        aer["nombre"] = aer["aerolinea"].map(lambda c: f"{AEROLINEAS.get(c, c)} ({c})")
        peor, mejor = aer.iloc[-1], aer.iloc[0]
        titulo_aer = (f"{AEROLINEAS.get(peor['aerolinea'], peor['aerolinea'])} retrasa "
                      f"{pct(peor['tasa'], 0)} de sus vuelos;<br>"
                      f"{AEROLINEAS.get(mejor['aerolinea'], mejor['aerolinea'])}, {pct(mejor['tasa'], 0)}")
        fig_a = barras_h(aer, "nombre", peor["nombre"], promedio, titulo_aer)
        evento = st.plotly_chart(fig_a, width="stretch", on_select="rerun",
                                 selection_mode="points", key="sel_aerolinea")
        st.caption("Haz clic en una aerolínea para ver sus aeropuertos a la derecha. "
                   f"Solo se comparan aerolíneas con al menos {miles(MIN_VUELOS_AER)} vuelos en el filtro.")
    else:
        evento = None
        st.info("Con los filtros actuales hay menos de dos aerolíneas para comparar.")

# Aerolínea elegida con clic (drill-down)
elegida = None
try:
    puntos = evento.selection.points if evento else []
    if puntos:
        nombre = puntos[0].get("y")
        elegida = aer.loc[aer["nombre"] == nombre, "aerolinea"].iloc[0]
except Exception:
    elegida = None

with col_b:
    n_aerop = st.session_state.get("n_aerop", 15)
    base_ap = df_h[df_h["aerolinea"] == elegida] if elegida else df_h
    ap = tasa_por(base_ap, "origen").sort_values("vuelos", ascending=False).head(n_aerop)
    if len(ap) >= 2:
        ap = ap.sort_values("tasa")
        peor_ap = ap.iloc[-1]
        prom_ap = tasa(base_ap)
        quien = f" de {AEROLINEAS.get(elegida, elegida)}" if elegida else ""
        titulo_ap = (f"{peor_ap['origen']} es el aeropuerto más crítico{quien}:<br>"
                     f"{pct(peor_ap['tasa'], 0)} de vuelos tarde (top {len(ap)} por volumen)")
        fig_ap = barras_h(ap, "origen", peor_ap["origen"], prom_ap, titulo_ap,
                          etiqueta_prom=f"Promedio {AEROLINEAS.get(elegida, elegida)}" if elegida else "Promedio general")
        st.plotly_chart(fig_ap, width="stretch")
        st.select_slider("Aeropuertos a comparar (los de más vuelos)", options=[10, 15, 20, 30],
                         value=15, key="n_aerop")
        if elegida:
            st.caption(f"Mostrando solo vuelos de {AEROLINEAS.get(elegida, elegida)}. "
                       "Haz clic de nuevo en la barra (o doble clic en el fondo) para quitar la selección.")
    else:
        st.info("Con los filtros actuales hay menos de dos aeropuertos para comparar.")

# ---------------------------------------------------------------------------
# Mensaje de la página (storytelling: conflicto -> puente a la resolución)
# ---------------------------------------------------------------------------
st.markdown(
    f"""<div class="mensaje">
    <b>Qué nos dice esta página</b><br><br>
    1. <b>La hora es el patrón más fuerte</b>: el riesgo pasa de {pct(h_valle['tasa'], 0)} a primera hora
    a {pct(h_pico['tasa'], 0)} en la noche, porque los retrasos de la mañana se van encadenando en los
    siguientes vuelos del mismo avión.<br><br>
    2. La aerolínea y el aeropuerto también pesan, pero ninguno por sí solo dice qué vuelo concreto se va a retrasar.<br><br>
    3. Por eso el modelo <b>combina</b> estos patrones con cómo viene el día de cada avión y cada aeropuerto,
    4 horas antes de la salida. La siguiente página muestra qué tan bien lo hace.
    </div>""",
    unsafe_allow_html=True,
)
st.markdown(f'<div class="fuente">{FUENTE_DATOS}</div>', unsafe_allow_html=True)
