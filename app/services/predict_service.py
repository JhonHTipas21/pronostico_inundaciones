# app/services/predict_service.py
"""
Servicio de predicción de caudales con:
  - Clipping dual: estadístico (q99) + físico (capacidad máxima del canal)
  - Cuantificación de incertidumbre vía σ de residuales históricos
  - Pronóstico recursivo a 48h (cada 3h)
"""
import numpy as np
import pandas as pd
from datetime import datetime

from app.services.feature_service import build_features
from app.services.geo_service import get_caudal_max


# ─────────────────────────────────────────────────────────────────────
# Clipping físico + estadístico
# ─────────────────────────────────────────────────────────────────────

def _dual_clip(yhat: np.ndarray, estaciones,
               meta: dict) -> np.ndarray:
    """Clipping dual de predicciones vectorizado."""
    q99 = meta.get("y_q99", None)
    clip_factor = meta.get("clip_factor", 1.5)
    stat_upper = (q99 * clip_factor) if q99 is not None else float("inf")

    phys_uppers = np.array([get_caudal_max(est) for est in estaciones])
    uppers = np.minimum(stat_upper, phys_uppers)

    return np.clip(yhat, 0.0, np.maximum(uppers, 0.1))


# ─────────────────────────────────────────────────────────────────────
# Intervalos de incertidumbre
# ─────────────────────────────────────────────────────────────────────

def _compute_intervals(yhat: np.ndarray, estaciones,
                       meta: dict, z: float = 1.96):
    """Calcula intervalos de predicción ±z×σ por estación (vectorizado)."""
    std_by_station = meta.get("residual_std_by_station", {})
    std_global = meta.get("residual_std_global", 0.5)

    sigmas = np.array([std_by_station.get(est, std_global) for est in estaciones])

    lower = np.maximum(yhat - z * sigmas, 0.0)
    upper = yhat + z * sigmas

    return lower, upper


def _get_sigma(estacion: str, meta: dict) -> float:
    """Obtiene σ de residuales para una estación."""
    std_by_station = meta.get("residual_std_by_station", {})
    return std_by_station.get(estacion, meta.get("residual_std_global", 0.5))


def _safe_float(v) -> float:
    """Convierte a float seguro (sin NaN/Inf)."""
    f = float(v)
    if not np.isfinite(f):
        return 0.0
    return round(f, 4)


# ─────────────────────────────────────────────────────────────────────
# Predicciones principales
# ─────────────────────────────────────────────────────────────────────

def make_predictions(model, meta: dict, payload_df: pd.DataFrame,
                     horizon: int) -> list:
    """Genera predicciones de caudal (m³/s) con clipping dual."""
    dfm, feats_num, feats_cat = build_features(payload_df, horizon=horizon)
    feats = meta["features_numeric"] + meta["features_categorical"]

    missing = [f for f in feats if f not in dfm.columns]
    if missing:
        raise ValueError(f"Faltan features en payload: {missing}")

    yhat = model.predict(dfm[feats])
    yhat = np.where(np.isfinite(yhat), yhat, 0.0)
    yhat = _dual_clip(yhat, dfm["estacion"].values, meta)

    return [max(float(v), 0.0) for v in yhat]


def make_predictions_with_uncertainty(model, meta: dict,
                                      payload_df: pd.DataFrame,
                                      horizon: int) -> list:
    """Genera predicciones con intervalos de incertidumbre al 95%."""
    dfm, feats_num, feats_cat = build_features(payload_df, horizon=horizon)
    feats = meta["features_numeric"] + meta["features_categorical"]

    missing = [f for f in feats if f not in dfm.columns]
    if missing:
        raise ValueError(f"Faltan features en payload: {missing}")

    yhat = model.predict(dfm[feats])
    yhat = np.where(np.isfinite(yhat), yhat, 0.0)
    estaciones = dfm["estacion"].values
    yhat = _dual_clip(yhat, estaciones, meta)
    lower, upper = _compute_intervals(yhat, estaciones, meta)

    results = []
    for i, (_, row) in enumerate(dfm.iterrows()):
        results.append({
            "estacion": row["estacion"],
            "fecha": str(row["fecha"]),
            "caudal_pred_m3s": _safe_float(yhat[i]),
            "lower_95": _safe_float(lower[i]),
            "upper_95": _safe_float(upper[i]),
            "horizonte_h": horizon * 6,
        })

    return results


# ─────────────────────────────────────────────────────────────────────
# Pronóstico Recursivo 48h (cada 3h)
# ─────────────────────────────────────────────────────────────────────

def make_recursive_forecast(model, meta: dict, df_historico: pd.DataFrame,
                            estacion: str, lluvia_mm: float = 0.0,
                            temperatura_C: float = 24.0,
                            impermeabilidad_pct: float = 60.0,
                            caudal_previo: float = 0.5,
                            steps: int = 16) -> dict:
    """
    Genera un pronóstico recursivo a 48h en intervalos de 3h.

    Cada paso:
      1. Predice caudal con horizon=1 (6h) usando datos disponibles.
      2. Agrega la predicción como nuevo registro.
      3. Avanza 3h y repite.

    Parameters
    ----------
    model : modelo entrenado
    meta : metadata del modelo
    df_historico : DataFrame con datos históricos (al menos 30 registros)
    estacion : nombre de la estación
    lluvia_mm : lluvia esperada por paso
    temperatura_C : temperatura esperada
    impermeabilidad_pct : % de impermeabilidad
    caudal_previo : caudal de arranque si no hay datos
    steps : número de pasos de 3h (16 = 48h)

    Returns
    -------
    Dict con {historico, pronostico, q_max_canal_m3s}
    """
    sigma = _get_sigma(estacion, meta)
    q_max = get_caudal_max(estacion)

    # Preparar datos históricos
    if df_historico is not None and len(df_historico) > 0:
        df_hist = df_historico.copy()
        df_hist["fecha"] = pd.to_datetime(df_hist["fecha"])
        df_est = df_hist[df_hist["estacion"] == estacion].sort_values("fecha")

        if len(df_est) == 0:
            df_est = df_hist.sort_values("fecha")

        df_est = df_est.tail(30).copy()
        records = df_est.to_dict("records")

        # Histórico: últimas 24h (8 registros de 3h ≈ 4 de 6h)
        historico = []
        for _, row in df_est.tail(8).iterrows():
            historico.append({
                "fecha": str(row["fecha"]),
                "caudal_m3s": _safe_float(row.get("caudal_m3s", 0)),
            })
    else:
        # Sin datos históricos: crear registros sintéticos
        now = pd.Timestamp.now()
        records = []
        for i in range(30):
            ts = now - pd.Timedelta(hours=i * 6)
            records.append({
                "fecha": ts,
                "lluvia_mm": lluvia_mm * max(0, 1 - i * 0.05),
                "temperatura_C": temperatura_C,
                "impermeabilidad_pct": impermeabilidad_pct,
                "caudal_m3s": caudal_previo,
                "estacion": estacion,
            })
        records.reverse()
        historico = [{
            "fecha": str(records[-1]["fecha"]),
            "caudal_m3s": caudal_previo,
        }]

    # Inferencia recursiva
    pronostico = []
    last_date = pd.to_datetime(records[-1]["fecha"])

    for step in range(steps):
        try:
            # Construir DataFrame para predicción
            req_records = []
            for r in records[-30:]:
                rec = r.copy()
                if isinstance(rec.get("fecha"), pd.Timestamp):
                    rec["fecha"] = rec["fecha"].strftime("%Y-%m-%dT%H:%M:%S")
                elif hasattr(rec.get("fecha"), "strftime"):
                    rec["fecha"] = rec["fecha"].strftime("%Y-%m-%dT%H:%M:%S")
                else:
                    rec["fecha"] = str(rec["fecha"])
                req_records.append(rec)

            df_pred = pd.DataFrame(req_records)
            dfm, fn, fc = build_features(df_pred, horizon=1)
            feats = meta["features_numeric"] + meta["features_categorical"]

            for col in feats:
                if col not in dfm.columns:
                    dfm[col] = 0.0

            if len(dfm) == 0:
                break

            yhat = model.predict(dfm[feats].tail(1))
            yhat = np.where(np.isfinite(yhat), yhat, 0.0)
            pred_val = max(float(yhat[0]), 0.0)

            # Clipping físico estricto (no se excede la capacidad del canal designada)
            pred_val = min(pred_val, q_max)

            # Avanzar 3h
            next_date = last_date + pd.Timedelta(hours=3)
            hora_adelanto = (step + 1) * 3

            # Incertidumbre geométrica exacta basada en σ residual (95% CI)
            lower = max(pred_val - (1.96 * sigma), 0.0)
            upper = pred_val + (1.96 * sigma)
            
            # Clipping físico restrictivo a la cota del upper si rebasa el límite de alerta
            upper = min(upper, q_max)

            # Limpiar NaN/Inf con np.nan_to_num
            pred_val = float(np.nan_to_num(pred_val, nan=0.0, posinf=0.0, neginf=0.0))
            lower = float(np.nan_to_num(lower, nan=0.0, posinf=0.0, neginf=0.0))
            upper = float(np.nan_to_num(upper, nan=0.0, posinf=0.0, neginf=0.0))

            pronostico.append({
                "fecha": next_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "hora_adelanto": hora_adelanto,
                "caudal_pred_m3s": _safe_float(pred_val),
                "lower_95": _safe_float(lower),
                "upper_95": _safe_float(upper),
            })

            # Retroalimentar predicción como nuevo registro
            new_row = {
                "fecha": next_date,
                "lluvia_mm": lluvia_mm,
                "temperatura_C": temperatura_C,
                "impermeabilidad_pct": impermeabilidad_pct,
                "caudal_m3s": pred_val,
                "estacion": estacion,
            }
            records.append(new_row)
            last_date = next_date

        except Exception:
            break

    return {
        "estacion": estacion,
        "q_max_canal_m3s": q_max,
        "n_steps": len(pronostico),
        "historico": historico,
        "pronostico": pronostico,
    }
