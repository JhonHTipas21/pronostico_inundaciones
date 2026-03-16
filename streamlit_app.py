# streamlit_app.py
"""
Dashboard interactivo — Hidrograma de Alerta Temprana.
Predicción de caudales pluviales urbanos, Santiago de Cali.

Ejecutar:  python3 -m streamlit run streamlit_app.py
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess
import sys

# ═══════════════════════════════════════════════════════════════════
# Configuración
# ═══════════════════════════════════════════════════════════════════
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Hidrograma de Alerta · Santiago de Cali",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .main .block-container { padding-top: 1.5rem; max-width: 1300px;
                             font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #142136 50%, #1a2d4a 100%);
    }
    [data-testid="stSidebar"] * { color: #c8d6e5 !important; }
    [data-testid="stSidebar"] h1 { color: #48dbfb !important; font-size: 1.15rem !important; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #74b9ff !important; }
    .metric-card {
        background: linear-gradient(135deg, #0c1a30 0%, #162d50 100%);
        border-radius: 12px; padding: 18px 22px; margin: 6px 0;
        border-left: 4px solid #0984e3; color: #fff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h3 { margin: 0; font-size: 0.8rem; color: #74b9ff;
                       text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.7rem; font-weight: 700; margin: 4px 0;
                          color: #dfe6e9; }
    .hero-header {
        background: linear-gradient(135deg, #0a1628 0%, #162d50 60%, #1e3a5f 100%);
        border-radius: 14px; padding: 28px 32px; margin-bottom: 24px;
        color: #fff; text-align: center;
        border: 1px solid rgba(9,132,227,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .hero-header h1 { margin: 0; font-size: 1.6rem; color: #dfe6e9; }
    .hero-header p { margin: 6px 0 0; opacity: 0.75; font-size: 0.88rem;
                     color: #b2bec3; line-height: 1.5; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 20px;
             font-size: 0.75rem; font-weight: 600; }
    .badge-ok   { background: #00b894; color: #fff; }
    .badge-warn { background: #fdcb6e; color: #2d3436; }
    .badge-bad  { background: #d63031; color: #fff; }
    .station-table { width: 100%; border-collapse: collapse;
                     font-size: 0.85rem; margin-top: 8px; }
    .station-table th { background: #0c1a30; color: #74b9ff; padding: 10px 14px;
                        text-align: left; font-weight: 600; }
    .station-table td { padding: 10px 14px; border-bottom: 1px solid #1e3a5f;
                        color: #dfe6e9; }
    .station-table tr:hover { background: rgba(9, 132, 227, 0.08); }
    .alert-card {
        border-radius: 10px; padding: 16px 20px; margin: 8px 0;
        font-size: 0.88rem;
    }
    .alert-ok { background: rgba(0,184,148,0.15); color: #55efc4;
                border-left: 4px solid #00b894; }
    .alert-danger { background: rgba(214,48,49,0.15); color: #ff7675;
                    border-left: 4px solid #d63031; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def api_get(path, timeout=10):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def api_post(path, data, timeout=1200):
    try:
        r = requests.post(f"{API_URL}{path}", json=data, timeout=timeout)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=30)
def api_health():
    try:
        r = requests.get(f"{API_URL.replace('/api/v1', '')}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def r2_badge(r2):
    if pd.isna(r2):
        return f'<span class="badge badge-bad">N/A</span>'
    if r2 < 0.0:
        return f'<span class="badge badge-warn">En calibración</span>'
    if r2 >= 0.80:
        return f'<span class="badge" style="background:#00e676; color:#0e1726; box-shadow: 0 0 8px #00e676;">R² = {r2:.3f}</span>'
    elif r2 >= 0.40:
        return f'<span class="badge badge-warn" style="background:#fdcb6e; color:#2d3436;">R² = {r2:.3f}</span>'
    return f'<span class="badge badge-bad">R² = {r2:.3f}</span>'

@st.cache_data
def load_holdout_data():
    p = Path("app/data/processed/holdout_preds.csv")
    return pd.read_csv(p, parse_dates=["fecha"]) if p.exists() else None

@st.cache_data
def load_comparison_data():
    p = Path("app/data/comparison/comparison_results.json")
    if p.exists():
        return json.load(open(p))
    return None

def ensure_comparison_data():
    """Auto-ejecuta compare_models.py si no existe el JSON."""
    p = Path("app/data/comparison/comparison_results.json")
    if not p.exists():
        try:
            subprocess.run(
                [sys.executable, "compare_models.py"],
                cwd=str(Path(__file__).parent),
                timeout=300, capture_output=True,
            )
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# Hidrograma de Ingeniería — Builder
# ═══════════════════════════════════════════════════════════════════

def build_hidrograma(forecast_data: dict, title_suffix: str = ""):
    """
    Construye un Hidrograma de Caudal Resultante profesional.
    Curvas spline suavizadas, eje X en horas, bandas IC 95%.
    """
    estacion = forecast_data["estacion"]
    q_max = forecast_data["q_max_canal_m3s"]
    historico = forecast_data.get("historico", [])
    pronostico = forecast_data.get("pronostico", [])

    if not pronostico:
        st.warning("No se generaron pronósticos.")
        return

    fig = go.Figure()

    # ── Preparar ejes temporales ──
    # Punto de referencia = último dato histórico (T=0)
    if historico:
        t_ref = pd.to_datetime(historico[-1]["fecha"])
    else:
        t_ref = pd.to_datetime(pronostico[0]["fecha"]) - pd.Timedelta(hours=3)

    # Fechas absolutas del pronóstico
    fechas_pred = [pd.to_datetime(p["fecha"]) for p in pronostico]
    vals_pred = [p["caudal_pred_m3s"] for p in pronostico]
    lower_pred = [p["lower_95"] for p in pronostico]
    upper_pred = [p["upper_95"] for p in pronostico]

    # Fechas absolutas del histórico
    fechas_hist = [pd.to_datetime(h["fecha"]) for h in historico] if historico else []
    vals_hist = [h["caudal_m3s"] for h in historico] if historico else []

    # ── 1. Banda de Confianza IC 95% (sombra del hidrograma) ──
    band_x = fechas_pred + fechas_pred[::-1]
    band_y = upper_pred + lower_pred[::-1]
    fig.add_trace(go.Scatter(
        x=band_x, y=band_y,
        fill="toself",
        fillcolor="rgba(9, 132, 227, 0.12)",
        line=dict(color="rgba(9, 132, 227, 0.25)", width=1),
        name="Intervalo de Confianza 95%",
        hoverinfo="skip",
        showlegend=True,
    ))

    # ── 2. Caudal Histórico (Línea azul sólida gruesa) ──
    if fechas_hist and vals_hist:
        fig.add_trace(go.Scatter(
            x=fechas_hist, y=vals_hist,
            mode="lines+markers",
            name="Caudal Real (Histórico)",
            line=dict(color="#0984e3", width=4, shape="spline"),
            marker=dict(size=7, color="#0984e3",
                        line=dict(width=1.5, color="#fff")),
        ))

    # ── 3. Caudal Predicho (Línea naranja punteada, spline) ──
    fig.add_trace(go.Scatter(
        x=fechas_pred, y=vals_pred,
        mode="lines+markers",
        name="Caudal Resultante (Pronóstico)",
        line=dict(color="#e17055", width=3, dash="dash", shape="spline"),
        marker=dict(size=5, color="#e17055",
                    line=dict(width=1, color="#fff")),
    ))

    # Marcadores especiales en T+3h y T+6h  
    special_hours = [3, 6]
    sp_x, sp_y, sp_text = [], [], []
    for p in pronostico:
        if p["hora_adelanto"] in special_hours:
            h_rel = pd.to_datetime(p["fecha"])
            sp_x.append(h_rel)
            sp_y.append(p["caudal_pred_m3s"])
            sp_text.append(f"T+{p['hora_adelanto']}h\n{p['caudal_pred_m3s']:.2f} m³/s")

    if sp_x:
        fig.add_trace(go.Scatter(
            x=sp_x, y=sp_y,
            mode="markers+text",
            name="Alerta T+3 / T+6",
            marker=dict(size=16, color="#fdcb6e", symbol="star-diamond",
                        line=dict(width=2, color="#2d3436")),
            text=sp_text,
            textposition="top center",
            textfont=dict(size=9, color="#ffeaa7"),
        ))

    # ── 4. Línea de Alerta Roja (Q_max capacidad del canal) ──
    all_dates = fechas_hist + fechas_pred
    d_min = min(all_dates) if all_dates else t_ref - pd.Timedelta(hours=24)
    d_max = max(all_dates) if all_dates else t_ref + pd.Timedelta(hours=48)
    d_min = d_min - pd.Timedelta(hours=2)
    d_max = d_max + pd.Timedelta(hours=2)

    fig.add_trace(go.Scatter(
        x=[d_min, d_max],
        y=[q_max, q_max],
        mode="lines",
        name=f"Capacidad Máxima Canal ({q_max:.1f} m³/s)",
        line=dict(color="#d63031", width=2.5, dash="dot"),
    ))

    # Anotación Q_max
    fig.add_annotation(
        x=d_max, y=q_max,
        text=f"Q_máx = {q_max:.1f} m³/s",
        showarrow=True, arrowhead=2, arrowcolor="#d63031",
        font=dict(size=10, color="#ff7675"),
        bgcolor="rgba(214,48,49,0.15)",
        bordercolor="#d63031", borderwidth=1,
        ax=-60, ay=-25,
    )

    # ── 5. Separadores diarios (líneas verticales tenues) ──
    current_day = d_min.floor('D')
    while current_day <= d_max:
        if current_day >= d_min:
            fig.add_shape(
                type="line",
                x0=current_day, x1=current_day,
                y0=0, y1=1, yref="paper",
                line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"),
            )
            fig.add_annotation(
                x=current_day, y=1.03, yref="paper",
                text=current_day.strftime("%b %d"),
                showarrow=False,
                font=dict(size=9, color="rgba(255,255,255,0.4)"),
            )
        current_day += pd.Timedelta(days=1)

    # ── Layout de Hidrograma Profesional ──
    fig.update_layout(
        title=dict(
            text=(f"Hidrograma de Caudal Resultante — {estacion}"
                  f"{title_suffix}"),
            font=dict(size=16, color="#dfe6e9", family="Inter"),
            x=0.5, xanchor="center",
        ),
        template="plotly_dark",
        height=550,
        margin=dict(l=70, r=40, t=90, b=80),
        xaxis=dict(
            title=dict(text="Tiempo", font=dict(size=13, color="#b2bec3")),
            dtick=10800000,
            tickformat="%H:%M\n%b %d",
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.2)",
            zerolinewidth=2,
            tickfont=dict(size=11, color="#b2bec3"),
            range=[d_min, d_max],
        ),
        yaxis=dict(
            title=dict(text="Caudal (m³/s)", font=dict(size=13, color="#b2bec3")),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.1)",
            tickfont=dict(size=11, color="#b2bec3"),
            rangemode="tozero",
        ),
        legend=dict(
            orientation="h", y=-0.18, x=0.5, xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color="#b2bec3"),
        ),
        plot_bgcolor="rgba(10, 22, 40, 0.95)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(20,33,54,0.95)", font_size=11),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Tabla de pronóstico ──
    with st.expander("📋 Tabla de Pronóstico Detallada", expanded=False):
        df_pron = pd.DataFrame(pronostico)
        df_pron["fecha"] = pd.to_datetime(df_pron["fecha"]).dt.strftime("%b %d, %H:%M")
        df_pron = df_pron.rename(columns={
            "fecha": "Fecha/Hora",
            "hora_adelanto": "T+ (h)",
            "caudal_pred_m3s": "Q pred (m³/s)",
            "lower_95": "IC inf",
            "upper_95": "IC sup",
        })
        df_pron["Estado"] = df_pron["Q pred (m³/s)"].apply(
            lambda v: "🚨 EXCEDE" if v > q_max else "✅ Normal"
        )
        st.dataframe(df_pron, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🌊 Canales Principales del Sur de Cali")
    st.markdown("---")

    backend_ok = api_health()
    if backend_ok:
        st.success("✅ Backend conectado", icon="🟢")
    else:
        st.error("❌ Backend offline", icon="🔴")

    st.markdown("---")

    vista = st.radio(
        "📊 Vista",
        ["Dashboard", "Hidrograma de Alerta",
         "Predicción CSV", "Ajuste Manual",
         "Análisis de Dispersión",
         "Comparación Modelos", "Reentrenamiento"],
        index=0,
    )

    st.markdown("---")

    stations_data = api_get("/stations")
    station_names = ([s["nombre"] for s in stations_data.get("estaciones", [])]
                     if stations_data else [
        "Canal Cañaveralejo", "Canal Ciudad Jardin",
        "Canal Interceptor Sur", "Quebrada Lili",
        "Quebrada Pance (urbana)", "Rio Melendez"])
    selected_station = st.selectbox("🏞️ Estación", station_names, index=0)

    # Q_max de la estación
    q_max_selected = 30.0
    if stations_data:
        for s in stations_data.get("estaciones", []):
            if s["nombre"] == selected_station:
                q_max_selected = s.get("caudal_max_m3s", 30.0)
                break

    horizon_opts = {
        "3 horas (1 paso)": 1,
        "6 horas (2 pasos)": 2,
    }
    horizon_label = st.selectbox("⏱️ Horizonte de Predicción",
                                  list(horizon_opts.keys()))
    horizon = horizon_opts[horizon_label]

    st.markdown("---")
    st.caption("Universidad Santiago de Cali")
    st.caption("Proyecto de Grado · 2026")


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <h1>🌊 Predicción de Caudales Pluviales</h1>
    <p>Prototipo de sistema predictivo, correspondiente a los canales pluviales urbanos,
    para la mejora en la toma de decisiones en la gestión preventiva de Inundaciones
    al sur de Santiago de Cali — <strong>Enfoque de Hidrograma de Diseño</strong></p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# VISTA: Dashboard
# ═══════════════════════════════════════════════════════════════════
if vista == "Dashboard":
    metrics = api_get("/metrics")
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><h3>Modelo</h3>'
                        f'<div class="value">{metrics.get("model_type","—").upper()}</div></div>',
                        unsafe_allow_html=True)
        with c2:
            r2_val = metrics.get("r2_cv_mean", 0)
            if r2_val < 0:
                color_r2 = "#fdcb6e"  # Amarillo
                glow_r2 = ""
                display_val = "Calibración"
            else:
                color_r2 = "#00e676" if r2_val >= 0.80 else ("#fdcb6e" if r2_val >= 0.40 else "#d63031")
                glow_r2 = f'text-shadow: 0 0 10px {color_r2};' if r2_val >= 0.80 else ''
                display_val = f"{r2_val:.4f}"
            
            st.markdown(f'<div class="metric-card"><h3>R² (CV)</h3>'
                        f'<div class="value" style="color:{color_r2}; {glow_r2}">{display_val}</div></div>',
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h3>RMSE (CV)</h3>'
                        f'<div class="value">{metrics.get("rmse_cv_mean",0):.4f}</div></div>',
                        unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><h3>Horizonte</h3>'
                        f'<div class="value">{metrics.get("horizon",1)*6}h</div></div>',
                        unsafe_allow_html=True)

    st.markdown("---")

    holdout = load_holdout_data()
    if holdout is not None:
        st.subheader(f"📈 Caudal Real vs Predicho — {selected_station}")
        df_est = holdout[holdout["estacion"] == selected_station].copy()
        if len(df_est) > 0:
            dias_vis = st.slider("📅 Días visibles", 3, 30, 7)
            cutoff = df_est["fecha"].max() - pd.Timedelta(days=dias_vis)
            df_est = df_est[df_est["fecha"] >= cutoff]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_est["fecha"], y=df_est["real"],
                mode="lines", name="Real",
                line=dict(color="#0984e3", width=3, shape="spline"),
                fill="tozeroy", fillcolor="rgba(9,132,227,0.06)"))
            fig.add_trace(go.Scatter(
                x=df_est["fecha"], y=df_est["pred"],
                mode="lines+markers", name="Predicho",
                line=dict(color="#e17055", width=2, shape="spline"),
                marker=dict(size=3)))

            if "lower_95" in df_est.columns and "upper_95" in df_est.columns:
                fig.add_trace(go.Scatter(
                    x=df_est["fecha"], y=df_est["upper_95"],
                    mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(
                    x=df_est["fecha"], y=df_est["lower_95"],
                    mode="lines", name="IC 95%",
                    fill="tonexty", fillcolor="rgba(225,112,85,0.1)",
                    line=dict(width=0)))

            fig.update_layout(
                template="plotly_dark", height=420,
                margin=dict(l=60, r=20, t=40, b=60),
                xaxis_title="Fecha", yaxis_title="Caudal (m³/s)",
                legend=dict(orientation="h", y=1.08),
                plot_bgcolor="rgba(10,22,40,0.95)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickformat="%b %d, %H:%M", nticks=12,
                           gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Calculando métricas... (procesando en background).")

    if metrics and metrics.get("por_estacion"):
        st.subheader("📊 Métricas por Estación")
        rows_html = ""
        for m in metrics["por_estacion"]:
            rows_html += (f'<tr><td>{m["estacion"]}</td>'
                          f'<td>{r2_badge(m["r2"])}</td>'
                          f'<td>{m["rmse"]:.4f}</td></tr>')
        st.markdown(
            f'<table class="station-table"><thead><tr>'
            f'<th>Estación</th><th>R²</th><th>RMSE</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# VISTA: Hidrograma de Alerta Temprana
# ═══════════════════════════════════════════════════════════════════
elif vista == "Hidrograma de Alerta":
    st.subheader("🌊 Hidrograma de Evento Crítico — Alerta Temprana (3-6h)")
    st.markdown("""
    Genera un **hidrograma resultante** con pronóstico recursivo a 48h
    en intervalos de 3 horas (T+3, T+6, ... T+48). Incluye **envolvente
    de confianza al 95%** y **capacidad máxima del canal** como
    referencia de diseño hidráulico.
    """)

    col1, col2 = st.columns(2)
    with col1:
        lluvia_forecast = st.slider(
            "🌧️ Lluvia esperada (mm/paso)", 0.0, 50.0, 5.0, 0.5,
            key="lluvia_hf")
    with col2:
        steps_forecast = st.slider(
            "⏱️ Pasos de 3h", 4, 24, 16,
            help="16 pasos = 48 horas de pronóstico", key="steps_hf")

    if st.button("🚀 Generar Hidrograma", type="primary", key="btn_hf"):
        if not backend_ok:
            st.error("⚠️ Backend no conectado.")
        else:
            with st.spinner(
                f"⏳ Generando hidrograma recursivo a "
                f"{steps_forecast * 3}h..."):
                result = api_post("/forecast-48h", {
                    "estacion": selected_station,
                    "lluvia_mm": lluvia_forecast,
                    "steps": steps_forecast,
                })

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                pronostico = result.get("pronostico", [])
                if pronostico:
                    max_pred = max(p["caudal_pred_m3s"] for p in pronostico)
                    q_max = result.get("q_max_canal_m3s", 30.0)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(
                            f'<div class="metric-card"><h3>Q pico predicho</h3>'
                            f'<div class="value">{max_pred:.2f} m³/s</div>'
                            f'</div>', unsafe_allow_html=True)
                    with c2:
                        color = "#d63031" if max_pred > q_max else "#00b894"
                        st.markdown(
                            f'<div class="metric-card"><h3>Capacidad canal</h3>'
                            f'<div class="value" style="color:{color}">'
                            f'{q_max:.1f} m³/s</div></div>',
                            unsafe_allow_html=True)
                    with c3:
                        t3 = next(
                            (p for p in pronostico
                             if p["hora_adelanto"] == 3), None)
                        t6 = next(
                            (p for p in pronostico
                             if p["hora_adelanto"] == 6), None)
                        t3v = f'{t3["caudal_pred_m3s"]:.2f}' if t3 else "—"
                        t6v = f'{t6["caudal_pred_m3s"]:.2f}' if t6 else "—"
                        st.markdown(
                            f'<div class="metric-card"><h3>T+3h / T+6h</h3>'
                            f'<div class="value">{t3v} / {t6v}</div></div>',
                            unsafe_allow_html=True)

                build_hidrograma(result)


# ═══════════════════════════════════════════════════════════════════
# VISTA: Predicción CSV
# ═══════════════════════════════════════════════════════════════════
elif vista == "Predicción CSV":
    st.subheader("📁 Subir CSV para Predicción")
    st.markdown(
        "Columnas: `fecha`, `lluvia_mm`, `temperatura_C`, "
        "`impermeabilidad_pct`, `caudal_m3s`, `estacion`")

    uploaded = st.file_uploader("Selecciona CSV", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.success(f"✅ {len(df_up)} registros cargados")
        st.dataframe(df_up.head(10), use_container_width=True)

        if st.button("🚀 Generar Predicción", type="primary"):
            if not backend_ok:
                st.error("Backend no conectado.")
            else:
                df_up["fecha"] = pd.to_datetime(
                    df_up["fecha"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
                result = api_post("/predict", {
                    "horizon": horizon,
                    "records": df_up.to_dict("records")})
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    preds = result.get("predictions", [])
                    if preds:
                        pdf = pd.DataFrame(preds)
                        pdf["fecha"] = pd.to_datetime(pdf["fecha"])
                        for est in pdf["estacion"].unique():
                            sub = pdf[pdf["estacion"] == est]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=sub["fecha"],
                                y=sub["caudal_pred_m3s"],
                                mode="lines+markers", name=est,
                                line=dict(shape="spline")))
                            if "lower_95" in sub.columns:
                                fig.add_trace(go.Scatter(
                                    x=sub["fecha"], y=sub["upper_95"],
                                    mode="lines", line=dict(width=0),
                                    showlegend=False))
                                fig.add_trace(go.Scatter(
                                    x=sub["fecha"], y=sub["lower_95"],
                                    fill="tonexty",
                                    fillcolor="rgba(9,132,227,0.15)",
                                    mode="lines", line=dict(width=0),
                                    name="IC 95%"))
                            fig.update_layout(
                                title=f"Predicción — {est}",
                                template="plotly_dark", height=350,
                                plot_bgcolor="rgba(10,22,40,0.95)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                xaxis=dict(tickformat="%b %d, %H:%M"))
                            st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(pdf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# VISTA: Ajuste Manual (vinculado al Hidrograma)
# ═══════════════════════════════════════════════════════════════════
elif vista == "Ajuste Manual":
    st.subheader("🎛️ Ajuste de Variables Climáticas — Hidrograma SCS-CN")
    st.markdown(
        "Ajusta los parámetros hidrológicos y presiona **Simular** para "
        "regenerar el hidrograma de diseño con tus valores de lluvia "
        "y saturación del suelo.")

    c1, c2 = st.columns(2)
    with c1:
        lluvia = st.slider("🌧️ Lluvia (mm)", 0.0, 100.0, 10.0, 0.5)
        temperatura = st.slider("🌡️ Temperatura (°C)", 15.0, 38.0, 24.0, 0.5)
    with c2:
        imp = st.slider("🏗️ Impermeabilidad (%)", 0.0, 100.0, 60.0, 1.0)
        caudal_prev = st.slider(
            "💧 Caudal previo (m³/s)", 0.0, 20.0, 0.5, 0.1)

    steps_manual = st.slider(
        "⏱️ Horizonte de pronóstico (pasos de 3h)", 4, 24, 16,
        key="steps_am")

    if st.button("⚡ Simular", type="primary"):
        if not backend_ok:
            st.error("Backend no conectado.")
        else:
            with st.spinner(
                f"Simulando hidrograma — lluvia={lluvia}mm, "
                f"T={temperatura}°C, imp={imp}%..."):
                result = api_post("/forecast-48h", {
                    "estacion": selected_station,
                    "lluvia_mm": lluvia,
                    "temperatura_C": temperatura,
                    "impermeabilidad_pct": imp,
                    "caudal_previo_m3s": caudal_prev,
                    "steps": steps_manual,
                })

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                pronostico = result.get("pronostico", [])
                if pronostico:
                    max_pred = max(
                        p["caudal_pred_m3s"] for p in pronostico)
                    q_max = result.get("q_max_canal_m3s", 30.0)

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(
                            f'<div class="metric-card"><h3>Lluvia</h3>'
                            f'<div class="value">{lluvia} mm</div></div>',
                            unsafe_allow_html=True)
                    with c2:
                        st.markdown(
                            f'<div class="metric-card"><h3>Impermeab.</h3>'
                            f'<div class="value">{imp}%</div></div>',
                            unsafe_allow_html=True)
                    with c3:
                        st.markdown(
                            f'<div class="metric-card"><h3>Q pico</h3>'
                            f'<div class="value">{max_pred:.2f}</div>'
                            f'</div>', unsafe_allow_html=True)
                    with c4:
                        alert = ("🚨 ALERTA" if max_pred > q_max
                                 else "✅ Normal")
                        st.markdown(
                            f'<div class="metric-card"><h3>Estado</h3>'
                            f'<div class="value">{alert}</div></div>',
                            unsafe_allow_html=True)

                build_hidrograma(
                    result,
                    f" (P={lluvia}mm, Imp={imp}%)")


# ═══════════════════════════════════════════════════════════════════
# VISTA: Comparación Modelos (auto-generado)
# ═══════════════════════════════════════════════════════════════════
elif vista == "Comparación Modelos":
    st.subheader("🔬 Comparación Ridge vs Lasso vs ElasticNet")

    # Auto-ejecutar sin intervención manual
    ensure_comparison_data()
    comp = load_comparison_data()

    if not comp:
        with st.spinner(
            "⏳ Ejecutando comparación de modelos "
            "automáticamente..."):
            ensure_comparison_data()
            comp = load_comparison_data()

    if comp:
        for m in comp:
            r2 = m.get("r2_global", 0)
            if r2 < 0:
                color = "#fdcb6e"
                r2_display = "En calibración"
            else:
                color = "#00e676" if r2 >= 0.80 else ("#fdcb6e" if r2 >= 0.40 else "#d63031")
                r2_display = f"{r2:.4f}"
            
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #0c1a30 0%, #162d50 100%); 
                        padding:20px; border-radius:12px; margin-bottom:15px; 
                        border-left: 5px solid {color}; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                <h3 style="margin-top:0; color:#74b9ff; font-size: 1.2rem;">{m["model"].upper()}</h3>
                <p style="font-size:0.95rem; color:#b2bec3; line-height: 1.5;">{m.get("description", "")}</p>
                <div style="display:flex; gap:30px; margin-top:15px;">
                    <div style="font-size: 1.1rem;"><strong>R² Global:</strong> <span style="color:{color}; font-weight:bold;">{r2_display}</span></div>
                    <div style="font-size: 1.1rem; color:#dfe6e9;"><strong>RMSE:</strong> {m["rmse_global"]:.4f} m³/s</div>
                    <div style="font-size: 1.1rem; color:#dfe6e9;"><strong>MAE:</strong> {m["mae_global"]:.4f} m³/s</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

        df_comp = pd.DataFrame([{
            "Modelo": m["model"].upper(),
            "R²": m["r2_global"],
        } for m in comp])
        fig = px.bar(
            df_comp, x="Modelo", y="R²", color="Modelo",
            color_discrete_map={
                "RIDGE": "#0984e3",
                "LASSO": "#e17055"},
            title="R² Global (Holdout)")
        
        # Linea de meta 0.8
        fig.add_hline(y=0.80, line_dash="dash", line_color="#00e676", annotation_text="Meta 0.80")
        
        fig.update_layout(
            template="plotly_dark", height=350,
            plot_bgcolor="rgba(10,22,40,0.95)",
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se pudieron generar los resultados.")


# ═══════════════════════════════════════════════════════════════════
# VISTA: Análisis de Dispersión (Validation Plot)
# ═══════════════════════════════════════════════════════════════════
elif vista == "Análisis de Dispersión":
    st.subheader("📊 Análisis de Dispersión")
    st.markdown("""
    Gráfico de dispersión **Caudal Real vs Caudal Predicho**.
    La cercanía de los puntos a la **línea de identidad** (y = x) de 45 grados define la precisión y exactitud del modelo.
    """)

    holdout_scatter = load_holdout_data()
    if holdout_scatter is not None:
        df_sc = holdout_scatter[
            holdout_scatter["estacion"] == selected_station].copy()

        if len(df_sc) > 5:
            real = df_sc["real"].values
            pred = df_sc["pred"].values

            # Calcular métricas
            ss_res = ((real - pred) ** 2).sum()
            ss_tot = ((real - real.mean()) ** 2).sum()
            r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            rmse_val = np.sqrt(((real - pred) ** 2).mean())
            mae_val = np.abs(real - pred).mean()

            r2_display = f"{r2_val:.4f}" if r2_val >= 0 else "Calibración"
            color_r2 = "#00e676" if r2_val >= 0.80 else ("#fdcb6e" if r2_val >= 0.40 else "#d63031")
            if r2_val < 0: color_r2 = "#fdcb6e"

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="metric-card"><h3>R²</h3>'
                    f'<div class="value" style="color:{color_r2}">{r2_display}</div></div>',
                    unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f'<div class="metric-card"><h3>RMSE</h3>'
                    f'<div class="value">{rmse_val:.4f} m³/s</div></div>',
                    unsafe_allow_html=True)
            with c3:
                st.markdown(
                    f'<div class="metric-card"><h3>MAE</h3>'
                    f'<div class="value">{mae_val:.4f} m³/s</div></div>',
                    unsafe_allow_html=True)

            # Scatter Plot
            fig = go.Figure()

            # Puntos de dispersión
            fig.add_trace(go.Scatter(
                x=real, y=pred,
                mode="markers",
                name="Observaciones",
                marker=dict(
                    size=6, color="#0984e3", opacity=0.6,
                    line=dict(width=0.5, color="#74b9ff")),
            ))

            # Ejes proporcionales (Mismo Máximo y Mínimo global)
            v_min = min(real.min(), pred.min()) * 0.95
            v_max = max(real.max(), pred.max()) * 1.05
            
            # Línea de identidad (y = x)
            fig.add_trace(go.Scatter(
                x=[v_min, v_max], y=[v_min, v_max],
                mode="lines",
                name="Línea de Identidad (y = x)",
                line=dict(color="#d63031", width=2, dash="dash"),
            ))

            # Línea de regresión
            z = np.polyfit(real, pred, 1)
            p_line = np.poly1d(z)
            x_reg = np.linspace(v_min, v_max, 100)
            fig.add_trace(go.Scatter(
                x=x_reg, y=p_line(x_reg),
                mode="lines",
                name=f"Regresión (m={z[0]:.3f})",
                line=dict(color="#00b894", width=2),
            ))

            fig.update_layout(
                title=dict(
                    text=f"Dispersión Real vs Predicho — {selected_station}",
                    font=dict(size=15, color="#dfe6e9"),
                    x=0.5, xanchor="center"),
                template="plotly_dark",
                height=520,
                xaxis=dict(
                    title="Caudal Real (m³/s)",
                    gridcolor="rgba(255,255,255,0.06)",
                    range=[v_min, v_max],
                    constrain="domain",
                ),
                yaxis=dict(
                    title="Caudal Predicho (m³/s)",
                    gridcolor="rgba(255,255,255,0.06)",
                    range=[v_min, v_max],
                    scaleanchor="x", scaleratio=1,
                ),
                plot_bgcolor="rgba(10,22,40,0.95)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(
                    orientation="h", y=-0.15, x=0.5,
                    xanchor="center",
                    font=dict(size=10, color="#b2bec3")),
            )

            # Anotación R²
            r2_text = f"R² = {r2_val:.4f}" if r2_val >= 0 else "R² < 0 (Calibración)"
            fig.add_annotation(
                x=0.05, y=0.95, xref="paper", yref="paper",
                text=f"{r2_text}<br>RMSE = {rmse_val:.4f}",
                showarrow=False,
                font=dict(size=12, color="#dfe6e9"),
                bgcolor="rgba(10,22,40,0.8)",
                bordercolor="#0984e3", borderwidth=1,
                borderpad=8,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Distribución de residuales
            with st.expander("📉 Distribución de Residuales",
                              expanded=False):
                residuals = pred - real
                fig_res = go.Figure()
                fig_res.add_trace(go.Histogram(
                    x=residuals,
                    nbinsx=40,
                    marker_color="#0984e3",
                    opacity=0.7,
                    name="Residuales",
                ))
                fig_res.add_vline(
                    x=0, line=dict(color="#d63031", width=2, dash="dash"))
                fig_res.update_layout(
                    title="Distribución de Residuales (Pred - Real)",
                    template="plotly_dark", height=300,
                    xaxis_title="Residual (m³/s)",
                    yaxis_title="Frecuencia",
                    plot_bgcolor="rgba(10,22,40,0.95)",
                    paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_res, use_container_width=True)

        else:
            st.warning(
                f"Datos insuficientes para {selected_station}.")
    else:
        st.info("Datos de holdout no disponibles. Reentrena el modelo.")


# ═══════════════════════════════════════════════════════════════════
# VISTA: Reentrenamiento
# ═══════════════════════════════════════════════════════════════════
elif vista == "Reentrenamiento":
    st.subheader("🔧 Reentrenamiento del Modelo")
    c1, c2 = st.columns(2)
    with c1:
        model_type = st.selectbox(
            "Modelo", ["lasso", "ridge", "elasticnet"])
    with c2:
        retrain_horizon = st.selectbox(
            "Horizonte", list(horizon_opts.keys()))

    csv_path = st.text_input("CSV", value="app/data/dataset_6h.csv")

    if st.button("🚀 Reentrenar", type="primary"):
        if not backend_ok:
            st.error("Backend no conectado.")
        else:
            h = horizon_opts[retrain_horizon]
            with st.spinner(
                f"Entrenando {model_type.upper()} ({h*6}h)..."):
                result = api_post("/retrain", {
                    "csv_path": csv_path,
                    "model_type": model_type,
                    "horizon": h,
                }, timeout=600)
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success("✅ Modelo re-entrenado exitosamente")
                st.json(result)
