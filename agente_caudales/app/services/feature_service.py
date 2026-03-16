# app/services/feature_service.py
"""
Pipeline de ingeniería de features para predicción de caudales pluviales.

Construye features temporales (lags, rolling, estacionalidad),
geoespaciales (CN, escorrentía, área de cuenca) e interacciones
para series de 6 horas.
"""
import pandas as pd
import numpy as np
import re

from app.services.geo_service import enrich_dataframe as geo_enrich

# ─── Features base ────────────────────────────────────────────────
# Un "step" = 6 horas.  Ventanas: 4=24h, 8=48h, 20≈5d

BASE_NUMERICS = [
    "lluvia_mm",
    "temperatura_C",
    "impermeabilidad_pct",

    # Lluvia acumulada / medias (rolling 6H)
    "api_4", "api_8", "api_20",
    "rain_ma_4", "rain_ma_8", "rain_ema_8",

    # Temperatura suavizada
    "temp_ma_4", "temp_ma_8",

    # Lags de caudal (6H,12H,24H,48H) + lag del horizonte
    "lag_q_1", "lag_q_2", "lag_q_4", "lag_q_8", "lag_q_h",

    # Lags de lluvia
    "lag_rain_1", "lag_rain_2", "lag_rain_4",

    # Interacciones globales
    "rain_x_imp", "api4_x_imp",

    # Estacionalidad mensual/horaria
    "month_sin", "month_cos", "hour_sin", "hour_cos",

    # ── Nuevas features geoespaciales (estáticas por estación) ──
    "cn_number",
    "area_km2",
    "pendiente_media",
    "pct_impermeable",
    "tiempo_concentracion_h",
    "longitud_cauce_km",

    # ── Nuevas features dinámicas geoespaciales ──
    "coef_escorrentia",
    "escorrentia_mm",
    "saturacion_suelo",

    # ── Nuevas features de intensidad y tendencia ──
    "lluvia_max_4",        # máxima lluvia en ventana 24h
    "delta_q",             # cambio de caudal entre pasos
    "q_rolling_max_4",     # caudal máximo en ventana 24h
    "rain_intensity_idx",  # índice de intensidad = lluvia / tc
    "q_trend_4",           # pendiente de regresión lineal del caudal (4 pasos)
]

CATEGORICS = ["estacion"]

# Interacciones num×estación para variables clave
STATION_INTERACT_VARS = ["lluvia_mm", "api_4", "lag_q_1", "escorrentia_mm"]


def _sanitize_token(s: str) -> str:
    """Normaliza nombre de estación a token seguro para columna."""
    s = s.lower().strip()
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"^_+|_+$", "", s)
    return s or "est"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas y tipos de datos."""
    ren = {
        "Fecha": "fecha", "Lluvia_mm": "lluvia_mm",
        "Temperatura_C": "temperatura_C",
        "Impermeabilidad_pct": "impermeabilidad_pct",
        "Caudal_m3s": "caudal_m3s",
        "Estacion": "estacion", "Estación": "estacion",
        "estación": "estacion",
    }
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})

    req = ["fecha", "lluvia_mm", "temperatura_C",
           "impermeabilidad_pct", "caudal_m3s", "estacion"]
    mis = [c for c in req if c not in df.columns]
    if mis:
        raise ValueError(f"Faltan columnas requeridas: {mis}")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    for c in ["lluvia_mm", "temperatura_C", "impermeabilidad_pct", "caudal_m3s"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Rangos razonables
    df["lluvia_mm"] = df["lluvia_mm"].clip(lower=0)
    mx = df["impermeabilidad_pct"].max(skipna=True)
    if pd.notna(mx) and mx <= 1.0:
        df["impermeabilidad_pct"] = df["impermeabilidad_pct"] * 100.0
    df["impermeabilidad_pct"] = df["impermeabilidad_pct"].clip(0, 100)
    df["estacion"] = df["estacion"].astype(str)

    return df


def _add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features sinusoidales de estacionalidad."""
    month = df["fecha"].dt.month
    hour = df["fecha"].dt.hour
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return df


def _linear_trend(arr: np.ndarray) -> float:
    """Pendiente de regresión lineal simple sobre un array."""
    n = len(arr)
    if n < 2 or np.isnan(arr).any():
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = arr.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (arr - y_mean)).sum() / denom)


def build_features(df: pd.DataFrame, horizon: int, horizon_units: str = "6H"):
    """
    Construye todas las features para series 6H.

    Parameters
    ----------
    df : DataFrame con columnas estándar.
    horizon : pasos de 6H (1 paso = 6 horas).
    horizon_units : unidad temporal (para referencia).

    Returns
    -------
    dfm : DataFrame filtrado (sin NaN en features/target).
    feats_num : lista de nombres de features numéricas.
    feats_cat : lista de nombres de features categóricas.
    """
    df = standardize_columns(df.copy())
    df = df.sort_values(["estacion", "fecha"]).reset_index(drop=True)

    g = df.groupby("estacion", group_keys=False)

    # ── Rolling de lluvia ──
    df["api_4"] = g["lluvia_mm"].transform(
        lambda s: s.rolling(4, min_periods=1).sum())    # 24h
    df["api_8"] = g["lluvia_mm"].transform(
        lambda s: s.rolling(8, min_periods=1).sum())    # 48h
    df["api_20"] = g["lluvia_mm"].transform(
        lambda s: s.rolling(20, min_periods=1).sum())   # ~5d
    df["rain_ma_4"] = g["lluvia_mm"].transform(
        lambda s: s.rolling(4, min_periods=1).mean())
    df["rain_ma_8"] = g["lluvia_mm"].transform(
        lambda s: s.rolling(8, min_periods=1).mean())
    df["rain_ema_8"] = g["lluvia_mm"].apply(
        lambda s: s.ewm(span=8, adjust=False).mean()
    ).reset_index(level=0, drop=True)

    # ── Temperatura suavizada ──
    df["temp_ma_4"] = g["temperatura_C"].transform(
        lambda s: s.rolling(4, min_periods=1).mean())
    df["temp_ma_8"] = g["temperatura_C"].transform(
        lambda s: s.rolling(8, min_periods=1).mean())

    # ── Lags de caudal ──
    df["lag_q_1"] = g["caudal_m3s"].shift(1)
    df["lag_q_2"] = g["caudal_m3s"].shift(2)
    df["lag_q_4"] = g["caudal_m3s"].shift(4)
    df["lag_q_8"] = g["caudal_m3s"].shift(8)
    df["lag_q_h"] = g["caudal_m3s"].shift(horizon)

    # ── Lags de lluvia ──
    df["lag_rain_1"] = g["lluvia_mm"].shift(1)
    df["lag_rain_2"] = g["lluvia_mm"].shift(2)
    df["lag_rain_4"] = g["lluvia_mm"].shift(4)

    # ── Interacciones globales ──
    df["rain_x_imp"] = df["lluvia_mm"] * df["impermeabilidad_pct"]
    df["api4_x_imp"] = df["api_4"] * df["impermeabilidad_pct"]

    # ── Estacionalidad ──
    df = _add_seasonality(df)

    # ── Enriquecer con features geoespaciales (SCS-CN) ──
    df = geo_enrich(df)

    # ── Features de intensidad y tendencia ──
    df["lluvia_max_4"] = g["lluvia_mm"].transform(
        lambda s: s.rolling(4, min_periods=1).max())

    df["delta_q"] = g["caudal_m3s"].diff()
    df["delta_q"] = df["delta_q"].fillna(0.0)

    df["q_rolling_max_4"] = g["caudal_m3s"].transform(
        lambda s: s.rolling(4, min_periods=1).max())

    # Índice de intensidad = lluvia / tiempo_concentración
    tc = df["tiempo_concentracion_h"].replace(0, 1.0)
    df["rain_intensity_idx"] = df["lluvia_mm"] / tc

    # Tendencia lineal del caudal (ventana 4 pasos)
    df["q_trend_4"] = g["caudal_m3s"].transform(
        lambda s: s.rolling(4, min_periods=2).apply(_linear_trend, raw=True))
    df["q_trend_4"] = df["q_trend_4"].fillna(0.0)

    # ── Objetivo adelantado ──
    df["y_target"] = g["caudal_m3s"].shift(-horizon)

    # ── Interacciones num×estación ──
    feats_num = BASE_NUMERICS.copy()
    uniq = df["estacion"].dropna().unique().tolist()

    for st in uniq:
        m = (df["estacion"] == st).astype(int)
        t = _sanitize_token(st)
        for v in STATION_INTERACT_VARS:
            col = f"{v}__est_{t}"
            df[col] = df[v] * m
            feats_num.append(col)

    feats_cat = CATEGORICS.copy()

    # ── Anti-fuga: eliminar filas sin lags o target ──
    lag_cols = [c for c in feats_num if c.startswith("lag_")]
    need = ["y_target"] + lag_cols
    dfm = df.dropna(subset=need).copy()

    # Limpiar NaN/Inf restantes en numéricas
    dfm = dfm.dropna(subset=feats_num).copy()
    for c in feats_num:
        vals = dfm[c].to_numpy()
        if not np.isfinite(vals).all():
            dfm = dfm[np.isfinite(dfm[c])].copy()

    keep = ["fecha", "estacion", "y_target"] + feats_num
    dfm = dfm[keep]

    return dfm, feats_num, feats_cat
