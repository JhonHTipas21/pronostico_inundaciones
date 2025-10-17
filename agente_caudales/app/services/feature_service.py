import pandas as pd
import numpy as np

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # asegura nombres exactos
    rename = {
        "Fecha":"fecha","Lluvia_mm":"lluvia_mm","Temperatura_C":"temperatura_C",
        "Impermeabilidad_pct":"impermeabilidad_pct","Caudal_m3s":"caudal_m3s",
        "Estacion":"estacion"
    }
    return df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

def build_features(df: pd.DataFrame, horizon: int, horizon_units: str = "days"):
    """Devuelve df listo para modelar + lista de columnas de features."""
    df = standardize_columns(df.copy())
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["estacion","fecha"])
    # si tu serie es diaria, horizon interpreta días; si es horaria, horas
    shift = horizon
    target = df.groupby("estacion")["caudal_m3s"].shift(-shift)

    # derivados
    df["month"] = df["fecha"].dt.month
    df["dow"] = df["fecha"].dt.dayofweek

    # APIs y lags
    df["api_3"] = df.groupby("estacion")["lluvia_mm"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["api_7"] = df.groupby("estacion")["lluvia_mm"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    df["lag_q_1"] = df.groupby("estacion")["caudal_m3s"].shift(1)
    df["lag_q_3"] = df.groupby("estacion")["caudal_m3s"].shift(3)

    # ensamblar
    df["y_target"] = target
    feats = [
        "lluvia_mm","temperatura_C","impermeabilidad_pct",
        "api_3","api_7","lag_q_1","lag_q_3","month","dow"
    ]
    dfm = df.dropna(subset=["y_target","lag_q_1","lag_q_3"])[["fecha","estacion","y_target"]+feats]
    return dfm, feats
