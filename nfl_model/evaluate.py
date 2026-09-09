import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

from .model import fit


def metrics(frame):
    nt = frame[frame.margin != 0]
    y = (nt.margin > 0).astype(int)
    p = nt.pred_probability.to_numpy()
    calibration = []
    for low in np.arange(0, 1, .1):
        part = nt[(nt.pred_probability >= low) & (nt.pred_probability < low + .1)]
        if len(part):
            calibration.append({"from": round(float(low), 1), "n": len(part),
                                "mean_probability": float(part.pred_probability.mean()),
                                "observed_home_win_rate": float((part.margin > 0).mean())})
    return {"games": len(frame), "ties_excluded_from_winner_metrics": len(frame) - len(nt),
            "winner_accuracy": float(np.mean((p >= .5) == y)),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "spread_mae_points": float(mean_absolute_error(frame.margin, frame.pred_margin)),
            "total_mae_points": float(mean_absolute_error(frame.total, frame.pred_total)),
            "margin_interval_80_coverage": float((abs(frame.margin - frame.pred_margin) <= frame.radius_margin).mean()),
            "total_interval_80_coverage": float((abs(frame.total - frame.pred_total) <= frame.radius_total).mean()),
            "baselines": {"elo_accuracy": float(np.mean((nt.baseline_probability >= .5) == y)),
                          "elo_brier": float(brier_score_loss(y, nt.baseline_probability)),
                          "always_home_accuracy": float(y.mean()),
                          "constant_total_mae_points": float(mean_absolute_error(frame.total, frame.baseline_total))},
            "calibration_bins": calibration}


def backtest(data, start=2020, end=2025):
    folds, results = [], {}
    for year in range(start, end + 1):
        part = data[(data.season == year) & data.margin.notna()].copy()
        if part.empty:
            continue
        model = fit(data, year)
        part["pred_margin"], part["pred_total"], part["pred_probability"] = model.predict(part)
        part["radius_margin"], part["radius_total"] = model.radius_margin, model.radius_total
        part["baseline_total"] = model.baseline_total
        results[str(year)] = {"train_through": model.train_through, "calibration_year": model.calibration_year,
                              **metrics(part)}
        folds.append(part)
    if not folds:
        raise ValueError("No hay partidos evaluables")
    combined = pd.concat(folds, ignore_index=True)
    report = {"method": "walk-forward por temporada; modelo congelado, features actualizadas semanalmente",
              "odds_used": False, "per_season": results, "aggregate": metrics(combined),
              "limits": ["Backtest retrospectivo con resultados corregidos; no replay de snapshots originales.",
                         "Probabilidad condicionada a que no haya empate.",
                         "No incluye lesiones, cambios de QB ni clima.",
                         "Intervalos empíricos de calibración: cobertura futura no garantizada."]}
    return report, combined
