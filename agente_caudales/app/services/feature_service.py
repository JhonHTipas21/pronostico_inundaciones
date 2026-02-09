# app/services/feature_service.py
import pandas as pd
import numpy as np
import re

# Un "step" = 6 horas (dataset 6H). Ventanas en pasos (4=24h, 8=48h, 20≈5 días)
BASE_NUMERICS = [
    "lluvia_mm",
    "temperatura_C",
    "impermeabilidad_pct",
    # Lluvia acumulada y medias (6H)
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
    "month_sin","month_cos","hour_sin","hour_cos",
]
CATEGORICS = ["estacion"]

# Interacciones num×estación para variables clave (sigue siendo lineal)
STATION_INTERACT_VARS = ["lluvia_mm", "api_4", "lag_q_1"]

def _sanitize_token(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"^_+|_+$", "", s)
    return s or "est"

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {"Fecha":"fecha","Lluvia_mm":"lluvia_mm","Temperatura_C":"temperatura_C",
           "Impermeabilidad_pct":"impermeabilidad_pct","Caudal_m3s":"caudal_m3s",
           "Estacion":"estacion","Estación":"estacion","estación":"estacion"}
    df = df.rename(columns={k:v for k,v in ren.items() if k in df.columns})
    req = ["fecha","lluvia_mm","temperatura_C","impermeabilidad_pct","caudal_m3s","estacion"]
    mis = [c for c in req if c not in df.columns]
    if mis:
        raise ValueError(f"Faltan columnas requeridas: {mis}")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    for c in ["lluvia_mm","temperatura_C","impermeabilidad_pct","caudal_m3s"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # rangos
    df["lluvia_mm"] = df["lluvia_mm"].clip(lower=0)
    mx = df["impermeabilidad_pct"].max(skipna=True)
    if pd.notna(mx) and mx <= 1.0:
        df["impermeabilidad_pct"] = df["impermeabilidad_pct"]*100.0
    df["impermeabilidad_pct"] = df["impermeabilidad_pct"].clip(0,100)
    df["estacion"] = df["estacion"].astype(str)
    return df

def _add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    month = df["fecha"].dt.month
    hour  = df["fecha"].dt.hour
    df["month_sin"] = np.sin(2*np.pi*month/12.0)
    df["month_cos"] = np.cos(2*np.pi*month/12.0)
    df["hour_sin"]  = np.sin(2*np.pi*hour/24.0)
    df["hour_cos"]  = np.cos(2*np.pi*hour/24.0)
    return df

def build_features(df: pd.DataFrame, horizon: int, horizon_units: str = "6H"):
    """
    Construye features para serie 6H:
      - horizon: pasos de 6H (1 paso = 6 horas).
    Devuelve: dfm, feats_num, feats_cat
    """
    df = standardize_columns(df.copy())
    df = df.sort_values(["estacion","fecha"]).reset_index(drop=True)

    g = df.groupby("estacion", group_keys=False)

    # Lluvia acumulada y medias (6H)
    df["api_4"]  = g["lluvia_mm"].transform(lambda s: s.rolling(4,  min_periods=1).sum())  # 24h
    df["api_8"]  = g["lluvia_mm"].transform(lambda s: s.rolling(8,  min_periods=1).sum())  # 48h
    df["api_20"] = g["lluvia_mm"].transform(lambda s: s.rolling(20, min_periods=1).sum())  # ~5d
    df["rain_ma_4"]  = g["lluvia_mm"].transform(lambda s: s.rolling(4,  min_periods=1).mean())
    df["rain_ma_8"]  = g["lluvia_mm"].transform(lambda s: s.rolling(8,  min_periods=1).mean())
    df["rain_ema_8"] = g["lluvia_mm"].apply(lambda s: s.ewm(span=8, adjust=False).mean()).reset_index(level=0, drop=True)

    # Temperatura suavizada (6H)
    df["temp_ma_4"] = g["temperatura_C"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["temp_ma_8"] = g["temperatura_C"].transform(lambda s: s.rolling(8, min_periods=1).mean())

    # Lags de caudal (en pasos 6H)
    df["lag_q_1"] = g["caudal_m3s"].shift(1)   # +6h
    df["lag_q_2"] = g["caudal_m3s"].shift(2)   # +12h
    df["lag_q_4"] = g["caudal_m3s"].shift(4)   # +24h
    df["lag_q_8"] = g["caudal_m3s"].shift(8)   # +48h
    df["lag_q_h"] = g["caudal_m3s"].shift(horizon)

    # Lags lluvia
    df["lag_rain_1"] = g["lluvia_mm"].shift(1)
    df["lag_rain_2"] = g["lluvia_mm"].shift(2)
    df["lag_rain_4"] = g["lluvia_mm"].shift(4)

    # Interacciones globales
    df["rain_x_imp"] = df["lluvia_mm"] * df["impermeabilidad_pct"]
    df["api4_x_imp"] = df["api_4"]     * df["impermeabilidad_pct"]

    # Objetivo adelantado (horizon pasos de 6H)
    df["y_target"] = g["caudal_m3s"].shift(-horizon)

    # Estacionalidad mensual/horaria
    df = _add_seasonality(df)

    # Interacciones num×estación (solo 3 variables clave)
    feats_num = BASE_NUMERICS.copy()
    uniq = df["estacion"].dropna().unique().tolist()
    # normalizar nombres de estación para columnas
    def _tok(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
        s = re.sub(r"^_+|_+$", "", s)
        return s or "est"
    for st in uniq:
        m = (df["estacion"] == st).astype(int)
        t = _tok(st)
        for v in STATION_INTERACT_VARS:
            col = f"{v}__est_{t}"
            df[col] = df[v] * m
            feats_num.append(col)

    feats_cat = CATEGORICS.copy()

    # Cortes anti-fuga: NaN en lags y objetivo
    lag_cols = [c for c in feats_num if c.startswith("lag_")]
    need = ["y_target"] + lag_cols
    dfm = df.dropna(subset=need).copy()

    # Limpiar NaN/Inf restantes en numéricas
    dfm = dfm.dropna(subset=feats_num).copy()
    for c in feats_num:
        vals = dfm[c].to_numpy()
        if not np.isfinite(vals).all():
            dfm = dfm[np.isfinite(dfm[c])].copy()

    keep = ["fecha","estacion","y_target"] + feats_num
    dfm = dfm[keep]

    return dfm, feats_num, feats_cat
