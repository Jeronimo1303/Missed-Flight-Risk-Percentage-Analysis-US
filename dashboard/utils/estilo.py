"""
Estilo visual del dashboard, tomado del curso de Visualización de Datos (EAFIT).

Regla del curso: Contexto + Datos + Contraste = Mensaje.
- El dato principal va en azul marino.
- Solo el hallazgo que queremos resaltar va en dorado (o en rojo si es alerta).
- Todo lo que es contexto va en grises.
- Nunca un color distinto por barra ("efecto arcoíris").
"""

import plotly.graph_objects as go

# Paleta (la misma de los ejemplos de código del curso)
AZUL = "#002855"       # dato principal
DORADO = "#F5A800"     # énfasis: el hallazgo
GRIS = "#94a3b8"       # contexto
GRIS_CLARO = "#cbd5e1" # contexto secundario
GRIS_GRILLA = "#e2e8f0"
TEXTO = "#334155"
TEXTO_SUAVE = "#64748b"
ROJO = "#ef4444"       # alerta
VERDE = "#22c55e"      # positivo

FUENTE_DATOS = (
    "Fuente: Bureau of Transportation Statistics (BTS), Reporting Carrier On-Time "
    "Performance, ene-2022 a ago-2026. Retraso = salida 15 minutos o más después de la hora programada."
)

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
MESES_LARGOS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
                "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

AEROLINEAS = {
    "9E": "Endeavor Air", "AA": "American Airlines", "AS": "Alaska Airlines",
    "B6": "JetBlue", "DL": "Delta Air Lines", "F9": "Frontier", "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines", "MQ": "Envoy Air", "NK": "Spirit Airlines",
    "OH": "PSA Airlines", "OO": "SkyWest", "QX": "Horizon Air", "UA": "United Airlines",
    "WN": "Southwest", "YV": "Mesa Airlines", "YX": "Republic Airways",
}


def pct(x: float, dec: int = 1) -> str:
    """Formatea una proporción como porcentaje con coma decimal: 0.215 -> '21,5 %'."""
    return f"{x * 100:.{dec}f}".replace(".", ",") + " %"


def miles(n: float) -> str:
    """31833031 -> '31,8 M'; 52000 -> '52.000'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}".replace(".", ",") + " M"
    return f"{n:,.0f}".replace(",", ".")


def layout_base(fig: go.Figure, titulo: str, alto: int = 420) -> go.Figure:
    """Aplica el layout limpio del curso: fondo blanco, sin bordes, grilla tenue,
    título declarativo arriba a la izquierda (recorrido visual en Z)."""
    fig.update_layout(
        template="plotly_white",
        height=alto,
        title=dict(text=titulo, x=0, xanchor="left", font=dict(size=19, color=AZUL)),
        margin=dict(l=10, r=20, t=70, b=40),
        font=dict(color=TEXTO, size=13),
        separators=",.",          # coma decimal, punto de miles
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_size=13),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=GRIS_CLARO, ticks="")
    fig.update_yaxes(showgrid=True, gridcolor=GRIS_GRILLA, zeroline=False)
    return fig


CSS = f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
h1 {{ color: {AZUL}; }}
.kpi {{
    background: white; border: 1px solid {GRIS_GRILLA}; border-radius: 10px;
    padding: 16px 18px; height: 100%;
}}
.kpi .etiqueta {{ color: {TEXTO_SUAVE}; font-size: 0.85rem; margin-bottom: 4px; }}
.kpi .valor {{ color: {AZUL}; font-size: 2rem; font-weight: 700; line-height: 1.1; }}
.kpi .valor.enfasis {{ color: {DORADO}; }}
.kpi .valor.alerta {{ color: {ROJO}; }}
.kpi .contexto {{ color: {TEXTO_SUAVE}; font-size: 0.85rem; margin-top: 6px; }}
.mensaje {{
    border-left: 4px solid {DORADO}; background: #fffbeb; padding: 14px 18px;
    border-radius: 6px; color: {TEXTO};
}}
.fuente {{ color: {TEXTO_SUAVE}; font-size: 0.8rem; margin-top: 8px; }}
</style>
"""


def kpi(etiqueta: str, valor: str, contexto: str = "", clase: str = "") -> str:
    """Tarjeta de indicador en HTML (st.metric no permite controlar los colores)."""
    return (f'<div class="kpi"><div class="etiqueta">{etiqueta}</div>'
            f'<div class="valor {clase}">{valor}</div>'
            f'<div class="contexto">{contexto}</div></div>')
