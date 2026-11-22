"""
Página 4 — Priorización del día: ¿qué vuelos vigilar? (RESOLUCIÓN y CASO APLICADO).

Pregunta de negocio: hoy, 4 horas antes de cada salida, ¿qué vuelos debe vigilar el
centro de control de operaciones y por qué?
Audiencia: coordinador del centro de control de una aerolínea.
Acción esperada: revisar primero los vuelos de la lista (tripulación de reserva,
reacomodar conexiones, avisar a pasajeros).

Datos: predicción vuelo por vuelo de TODO agosto de 2026 con el LightGBM del notebook
del equipo (mismo código y variables; ver generar_predicciones.py). Como agosto ya pasó,
se puede comparar cada alerta con lo que realmente ocurrió.
Filtros que aplican: aerolínea y aeropuerto de origen (el periodo no, aquí se elige el día).
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.datos import DATA
from utils.estilo import (AEROLINEAS, AZUL, DORADO, GRIS, GRIS_CLARO, ROJO, TEXTO_SUAVE, VERDE,
                          kpi, layout_base, pct)

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_predicciones() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "predicciones" / "predicciones_ago2026.parquet")
    df["fecha"] = pd.to_datetime(df["FlightDate"]).dt.date
    return df


pred = cargar_predicciones()
meta = json.loads((DATA / "predicciones" / "meta.json").read_text(encoding="utf-8"))
UMBRAL = meta["umbral_costo"]          # umbral por costo elegido en validación (~0,25)
ALTO = 0.50                            # desde aquí, más probable que se retrase a que no

hhmm = lambda m: f"{int(m) // 60 % 24:02d}:{int(m) % 60:02d}"


def ent(x) -> int:
    """Entero seguro: NaN o vacío -> 0."""
    return 0 if x is None or pd.isna(x) else int(x)


def nivel(p: float) -> str:
    return "🔴 Alto" if p >= ALTO else ("🟡 Medio" if p >= UMBRAL else "🟢 Bajo")


def motivo_texto(var: str, r: pd.Series) -> str:
    """Traduce el nombre técnico de la variable que más subió el riesgo a una frase de negocio."""
    o, d = r["Origin"], r["Dest"]
    v = lambda c: r.get(c, np.nan)
    textos = {
        "prev_leg_delayed": "El vuelo anterior de este avión salió tarde" if v("prev_leg_delayed") == 1
                            else "Estado del vuelo anterior del avión",
        "tail_delayed_known": f"El avión ya tuvo {ent(v('tail_delayed_known'))} tramo(s) con retraso hoy",
        "tail_known_legs": "Tramos que el avión ya voló hoy",
        "tail_leg_num": f"Es el vuelo n.º {ent(v('tail_leg_num'))} del avión en el día",
        "turnaround_min": (f"Solo {int(v('turnaround_min'))} min entre la llegada anterior y esta salida"
                           if pd.notna(v("turnaround_min")) else "Tiempo entre vuelos del avión"),
        "origin_delay_rate_today": f"{o} lleva {(pct(v('origin_delay_rate_today'), 0) if pd.notna(v('origin_delay_rate_today')) else '—')} de salidas tarde hoy",
        "origin_known_today": f"Cómo va el día en {o}",
        "origin_rate_all": f"Historial de retrasos de {o}", "origin_rate_30d": f"{o} viene con retrasos el último mes",
        "route_rate_all": f"Historial de la ruta {o}–{d}", "route_rate_30d": f"La ruta {o}–{d} viene con retrasos el último mes",
        "airline_rate_all": "Historial de la aerolínea", "airline_rate_30d": "La aerolínea viene con retrasos el último mes",
        "tail_rate_all": "Historial de este avión", "tail_rate_30d": "Este avión viene con retrasos el último mes",
        "origin_hr_rate_all": f"Historial de {o} a esa hora", "origin_hr_rate_30d": f"{o} a esa hora viene con retrasos",
        "origin_dep_load": f"Muchas salidas programadas en {o} a esa hora", "origin_day_load": f"Día muy cargado en {o}",
        "dest_arr_load": f"Muchas llegadas programadas en {d} a esa hora",
        "dep_min": f"Sale a las {hhmm(r['dep_min'])} (más tarde = más riesgo)", "DepTimeBlk": "Franja horaria de salida",
        "arr_min": "Hora de llegada programada",
        "Month": "Época del año", "day_of_year": "Época del año", "DayofMonth": "Día del mes",
        "DayOfWeek": "Día de la semana", "is_weekend": "Fin de semana", "days_to_holiday": "Cercanía a un festivo",
        "Reporting_Airline": "Aerolínea", "Origin": f"Aeropuerto de origen {o}", "OriginState": "Estado de origen",
        "Dest": f"Destino {d}", "DestState": "Estado de destino", "route": f"Ruta {o}–{d}",
        "Distance": "Distancia del vuelo", "CRSElapsedTime": "Duración programada del vuelo",
    }
    return textos.get(var, var)


# ---------------------------------------------------------------------------
# Encabezado y controles
# ---------------------------------------------------------------------------
st.title("Priorización del día: ¿qué vuelos vigilar?")
st.caption("Caso aplicado con todos los vuelos de agosto de 2026. El modelo calcula el riesgo 4 horas antes de cada "
           "salida; como agosto ya pasó, se puede comparar cada alerta con lo que realmente ocurrió.")

dias = sorted(pred["fecha"].unique())
c_dia, c_n = st.columns([1, 2])
# Por defecto: el día que el modelo anticipaba como más complicado (riesgo promedio más alto).
dia_default = pred.groupby("fecha")["prob"].mean().idxmax()
if "dia_prior" not in st.session_state:          # así el botón de los 3 casos puede cambiar el día
    st.session_state["dia_prior"] = dia_default
dia = c_dia.date_input("Día", min_value=dias[0], max_value=dias[-1], format="DD/MM/YYYY", key="dia_prior")
n_rev = c_n.slider("¿Cuántos vuelos puede revisar tu equipo ese día?", min_value=10, max_value=300,
                   value=50, step=10, key="n_rev")

limpiar = st.toggle(
    "Ocultar vuelos con rotación imposible (posible fuga de información)", value=True, key="sin_rotacion",
    help="Vuelos cuyo avión estaba programado para llegar de su vuelo anterior después de esta salida, o con menos "
         "de 20 minutos de margen. Eso solo pasa cuando la aerolínea cambia el avión durante el día, y ese cambio "
         "puede no conocerse 4 horas antes. Se ocultan para que la lista sea honesta.")
hoy = pred[pred["fecha"] == dia]
n_ocultos = int(hoy["rotacion_imposible"].sum())
if limpiar:
    hoy = hoy[hoy["rotacion_imposible"] == 0]
aer = st.session_state.get("f_aerolineas", [])
org = st.session_state.get("f_origenes", [])
if aer:
    hoy = hoy[hoy["Reporting_Airline"].isin(aer)]
if org:
    hoy = hoy[hoy["Origin"].isin(org)]
notas = []
if aer or org:
    notas.append("se aplican los filtros de aerolínea / aeropuerto de la barra lateral")
if limpiar:
    notas.append(f"se ocultan {n_ocultos:,} vuelos con rotación imposible".replace(",", "."))
if notas:
    st.caption("Nota: " + "; ".join(notas) + ".")
if hoy.empty:
    st.warning("No hay vuelos ese día con los filtros elegidos.")
    st.stop()

hoy = hoy.sort_values("prob", ascending=False).reset_index(drop=True)
n_rev = min(n_rev, len(hoy))
lista = hoy.head(n_rev)
tasa_dia = hoy["DepDel15"].mean()
aciertos = int(lista["DepDel15"].sum())
anticipados = aciertos / max(hoy["DepDel15"].sum(), 1)

# ---------------------------------------------------------------------------
# KPIs del día
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi("Vuelos programados ese día", f"{len(hoy):,}".replace(",", "."),
                f"se retrasó {pct(tasa_dia)} (1 de cada {round(1 / tasa_dia) if tasa_dia else '—'})"),
            unsafe_allow_html=True)
k2.markdown(kpi(f"De los {n_rev} vuelos de la lista", f"{aciertos} se retrasaron",
                f"{pct(aciertos / n_rev, 0)} de la lista · al azar serían ~{round(tasa_dia * n_rev)}",
                clase="enfasis"), unsafe_allow_html=True)
k3.markdown(kpi("Retrasos del día anticipados", f"{aciertos} de {int(hoy['DepDel15'].sum()):,}".replace(",", "."),
                f"{pct(anticipados, 1)} de los retrasos, revisando {pct(n_rev / len(hoy), 1)} de los vuelos"),
            unsafe_allow_html=True)
k4.markdown(kpi("Retrasos esperados vs reales", f"{hoy['prob'].sum():,.0f} vs {int(hoy['DepDel15'].sum()):,}".replace(",", "."),
                "suma de probabilidades del modelo frente a lo que pasó"), unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------------------
# Lista priorizada (tabla interactiva: clic en una fila = ver el detalle)
# ---------------------------------------------------------------------------
st.subheader(f"Los {n_rev} vuelos con más riesgo de retraso el {dia.strftime('%d/%m/%Y')}")
tabla = pd.DataFrame({
    "Riesgo": lista["prob"],
    "Nivel": lista["prob"].map(nivel),
    "Salida": lista["dep_min"].map(hhmm),
    "Aerolínea": lista["Reporting_Airline"].map(lambda c: f"{AEROLINEAS.get(c, c)} ({c})"),
    "Vuelo": lista["Reporting_Airline"] + " " + lista["flight_num"].astype("Int64").astype(str),
    "Ruta": lista["Origin"] + " → " + lista["Dest"],
    "Avión": lista["tail"].fillna("—"),
    "Qué pasó": lista["DepDel15"].map({1: "⏰ Salió tarde", 0: "✅ A tiempo"}),
    "Principal señal de riesgo": [motivo_texto(m, r) if isinstance(m, str) else "—"
                                  for m, (_, r) in zip(lista["motivo1"], lista.iterrows())],
})
evento = st.dataframe(
    tabla, hide_index=True, width="stretch", height=min(420, 35 * (len(tabla) + 1) + 3),
    on_select="rerun", selection_mode="single-row", key="tabla_prior",
    column_config={
        "Riesgo": st.column_config.ProgressColumn("Riesgo", format="percent", min_value=0, max_value=1, width="small"),
        "Principal señal de riesgo": st.column_config.TextColumn(width="large"),
    },
)
st.caption(f"Nivel: 🔴 Alto = 50 % o más · 🟡 Medio = desde {pct(UMBRAL, 0)} (umbral por costo) · 🟢 Bajo = menos. "
           "Haz clic en una fila para ver por qué el modelo marcó ese vuelo. La columna \"Qué pasó\" solo existe "
           "porque es un día pasado: en operación real no se conoce todavía.")

# ---------------------------------------------------------------------------
# Detalle del vuelo elegido (drill-down) con la explicación del modelo
# ---------------------------------------------------------------------------
filas = evento.selection.rows if evento and evento.selection else []
i_sel = filas[0] if filas else 0
r = lista.iloc[i_sel]

col_f, col_g = st.columns([2, 3], gap="large")
with col_f:
    st.markdown(f"#### ✈️ {r['Reporting_Airline']} {ent(r['flight_num'])} · {r['Origin']} → {r['Dest']}")
    st.markdown(f"**Salida programada:** {hhmm(r['dep_min'])} · **Avión:** {r['tail'] if pd.notna(r['tail']) else '—'}")
    st.markdown(kpi("Riesgo de salir 15+ min tarde", pct(r["prob"], 0), nivel(r["prob"]),
                    clase="alerta" if r["prob"] >= ALTO else ("enfasis" if r["prob"] >= UMBRAL else "")),
                unsafe_allow_html=True)
    st.write("")
    st.markdown("**Lo que el modelo sabía 4 horas antes:**")
    datos = []
    if pd.notna(r.get("tail_leg_num")):
        datos.append(f"- Es el vuelo n.º {int(r['tail_leg_num'])} de este avión en el día")
    if pd.notna(r.get("prev_leg_delayed")):
        datos.append("- Su vuelo anterior " + ("**salió tarde**" if r["prev_leg_delayed"] == 1 else "salió a tiempo"))
    if pd.notna(r.get("turnaround_min")):
        datos.append(f"- Tiempo entre la llegada anterior y esta salida: {int(r['turnaround_min'])} min")
    if pd.notna(r.get("origin_delay_rate_today")):
        datos.append(f"- En {r['Origin']}, hoy van {pct(r['origin_delay_rate_today'], 0)} de salidas tarde")
    if pd.notna(r.get("route_rate_30d")):
        datos.append(f"- La ruta tuvo {pct(r['route_rate_30d'], 0)} de retrasos el último mes")
    st.markdown("\n".join(datos) if datos else "—")
    st.markdown("**Qué pasó:** " + ("⏰ salió con 15+ min de retraso: la alerta acertó."
                                     if r["DepDel15"] == 1 else
                                     "✅ salió a tiempo: fue una falsa alarma (el costo de revisarlo fue bajo)."))

with col_g:
    if isinstance(r.get("motivo1"), str):
        mot = pd.DataFrame({"var": [r["motivo1"], r["motivo2"], r["motivo3"]],
                            "aporte": [r["aporte1"], r["aporte2"], r["aporte3"]]}).dropna()
        mot["texto"] = [motivo_texto(v, r) for v in mot["var"]]
        mot = mot.sort_values("aporte")
        fig = go.Figure(go.Bar(
            y=mot["texto"], x=mot["aporte"], orientation="h", width=0.5,
            marker_color=[GRIS_CLARO] * (len(mot) - 1) + [DORADO],
            text=[f"+{a:.2f}".replace(".", ",") for a in mot["aporte"]], textposition="outside",
            cliponaxis=False, textfont=dict(color=TEXTO_SUAVE),
            hovertemplate="%{y}<br>Aporte al riesgo: <b>%{x:+.2f}</b><extra></extra>",
        ))
        layout_base(fig, "Por qué el modelo marcó este vuelo:<br>las 3 señales que más subieron su riesgo", alto=320)
        fig.update_xaxes(title="Aporte al riesgo (SHAP)",
                         range=[0, mot["aporte"].max() * 1.35], showgrid=False)
        fig.update_yaxes(showgrid=False, title=None, automargin=True)
        fig.update_layout(margin=dict(t=80, l=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Las 3 variables que más subieron el riesgo de ESTE vuelo según el propio modelo "
                   "(contribuciones SHAP de LightGBM, en escala log-odds: más larga la barra, más pesa). "
                   "No es una regla fija: cada vuelo tiene sus razones.")

# ---------------------------------------------------------------------------
# El día completo: riesgo predicho vs lo que pasó, por hora
# ---------------------------------------------------------------------------
hoy["hora"] = (hoy["dep_min"] // 60).astype(int) % 24
ph = hoy.groupby("hora").agg(vuelos=("prob", "size"), pred=("prob", "mean"), real=("DepDel15", "mean")).reset_index()
ph = ph[ph["vuelos"] >= 20]
if len(ph) >= 3:
    h_max = ph.loc[ph["real"].idxmax()]
    fig_d = go.Figure()
    fig_d.add_trace(go.Bar(x=ph["hora"].map(lambda h: f"{h:02d}:00"), y=ph["real"], name="Lo que pasó",
                           marker_color=GRIS_CLARO, width=0.6,
                           hovertemplate="%{x}<br>Real: <b>%{y:.1%}</b><extra></extra>"))
    fig_d.add_trace(go.Scatter(x=ph["hora"].map(lambda h: f"{h:02d}:00"), y=ph["pred"], name="Lo que esperaba el modelo",
                               mode="lines+markers", line=dict(color=AZUL, width=2.5), marker=dict(size=6),
                               hovertemplate="%{x}<br>Predicho: <b>%{y:.1%}</b><extra></extra>"))
    layout_base(fig_d, f"Ese día, el modelo siguió el ritmo de los retrasos hora a hora "
                       f"(pico real: {int(h_max['hora']):02d}:00 con {pct(h_max['real'], 0)})", alto=360)
    fig_d.update_layout(showlegend=True, legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0), margin=dict(t=95))
    fig_d.update_yaxes(title="% de vuelos con retraso", tickformat=".0%")
    fig_d.update_xaxes(title="Hora programada de salida")
    st.plotly_chart(fig_d, width="stretch")

# ---------------------------------------------------------------------------
# Tres casos reales explicados: cuándo acierta el modelo, cuándo da falsa alarma y qué
# retraso no vio. Responde a la rúbrica: predicciones individuales, por qué acierta o falla
# y qué decisión toma el usuario. Valores leídos de las predicciones (no escritos a mano).
# ---------------------------------------------------------------------------
from datetime import date  # noqa: E402

DIA_CASOS = date(2026, 8, 9)   # domingo: día típico de mucha operación
pct_ = lambda v: pct(v, 0).replace(" ", "\u00a0")   # que "36 %" no se parta en dos líneas
CASOS = [
    {"tipo": "✅ Acierto", "clase": "enfasis", "aer": "WN", "vuelo": 3416, "origen": "LAS",
     "por_que": lambda r: (f"Era el vuelo n.º {ent(r['tail_leg_num'])} del avión ese día, su vuelo anterior ya había "
                           f"salido tarde, tenía solo {ent(r['turnaround_min'])} min entre llegar y volver a salir, y "
                           f"{r['Origin']} llevaba {pct_(r['origin_delay_rate_today'])} de salidas tarde. "
                           "Todas las señales apuntaban al mismo lado: efecto dominó."),
     "decision": "Alistar tripulación de reserva y reacomodar desde temprano a los pasajeros con conexión."},
    {"tipo": "⚠️ Falsa alarma", "clase": "", "aer": "WN", "vuelo": 545, "origen": "BWI",
     "por_que": lambda r: (f"El modelo vio que el avión venía atrasado, que era de noche y que {r['Origin']} llevaba "
                           f"{pct_(r['origin_delay_rate_today'])} de salidas tarde. Pero el avión tenía "
                           f"{ent(r['turnaround_min'])} min de margen entre vuelos y alcanzó a recuperar el retraso."),
     "decision": "Lo revisa, ve el margen amplio y lo deja en vigilancia. La falsa alarma costó minutos de revisión."},
    {"tipo": "❌ Retraso que no vio", "clase": "alerta", "aer": "UA", "vuelo": 2608, "origen": "LAS",
     "por_que": lambda r: ("Era el primer vuelo del avión en el día: no había vuelo anterior del cual «contagiarse». "
                           f"A esa hora, en {r['Origin']}, solo {pct_(r['origin_hr_rate_30d'])} de los vuelos salían "
                           "tarde el último mes. La causa estuvo en algo que el modelo no sabía 4 horas antes "
                           "(los datos no traen la causa del retraso)."),
     "decision": "No confiar solo en el modelo para los primeros vuelos del día: ahí revisar mantenimiento y "
                 "tripulación por aparte (ver página 5: la madrugada es donde el modelo es más débil)."},
]

casos = []
for c in CASOS:
    m = pred[(pred["fecha"] == DIA_CASOS) & (pred["Reporting_Airline"] == c["aer"])
             & (pred["flight_num"] == c["vuelo"]) & (pred["Origin"] == c["origen"])]
    if len(m):
        casos.append((c, m.iloc[0]))

if len(casos) == len(CASOS):
    st.subheader("Tres vuelos para entender al modelo: cuándo acierta y cuándo falla")
    dia_txt = DIA_CASOS.strftime("%d/%m/%Y")
    base_dia = pred.loc[(pred["fecha"] == DIA_CASOS) & (pred["rotacion_imposible"] == 0), "DepDel15"].mean()
    st.caption(f"Domingo {dia_txt}: ese día salieron tarde {round(base_dia * 100)} de cada 100 vuelos. "
               "Tres casos reales del mismo día, sin rotación imposible.")
    cols = st.columns(3, gap="medium")
    for col, (c, r) in zip(cols, casos):
        with col:
            st.markdown(f"#### {c['tipo']}")
            st.markdown(f"**{r['Reporting_Airline']} {ent(r['flight_num'])} · {r['Origin']} → {r['Dest']} · "
                        f"{hhmm(r['dep_min'])}**")
            st.markdown(kpi("Riesgo que calculó el modelo", pct(r["prob"], 0),
                            "⏰ salió con 15+ min de retraso" if r["DepDel15"] == 1 else "✅ salió a tiempo",
                            clase=c["clase"]), unsafe_allow_html=True)
            st.markdown(f"**Por qué:** {c['por_que'](r)}")
            st.markdown(f"**Qué hace el supervisor:** {c['decision']}")

    def _ir_al_dia():
        st.session_state["dia_prior"] = DIA_CASOS
    st.button(f"Ver la lista completa del {dia_txt}", on_click=_ir_al_dia, key="btn_dia_casos")
    st.write("")

st.markdown(
    f"""<div class="mensaje">
    <b>Cómo lo usaría el centro de control</b><br><br>
    1. Cada vuelo recibe su riesgo 4 horas antes de salir; el coordinador ve la lista ordenada y revisa
    primero los de arriba, hasta donde le alcance el equipo.<br><br>
    2. Para cada vuelo de la lista ve <b>por qué</b> está ahí (por ejemplo, el avión ya viene retrasado), y así decide
    la acción: tripulación o avión de reserva, reacomodar conexiones o avisar a los pasajeros.<br><br>
    3. Habrá falsas alarmas (vuelos de la lista que salen a tiempo): se aceptan porque revisar un vuelo cuesta
    mucho menos que no anticipar un retraso (supuesto del equipo: 3 veces menos).
    </div>""",
    unsafe_allow_html=True,
)
st.markdown(f'<div class="fuente">Predicciones: LightGBM entrenado con el código del notebook del equipo '
            f'(prueba: PR-AUC {meta["metricas_prueba"]["pr_auc"]:.3f}, ROC-AUC {meta["metricas_prueba"]["roc_auc"]:.3f}). '
            'Datos BTS, agosto de 2026.</div>'.replace("0.", "0,"), unsafe_allow_html=True)
