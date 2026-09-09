"""Regresiones interpretables y calibración temporal de la probabilidad."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, LABELS


@dataclass
class Model:
    margin: object
    total: object
    probability: object
    radius_margin: float
    radius_total: float
    train_through: int
    calibration_year: int
    train_games: int
    baseline_total: float

    def predict(self, data):
        x = data[FEATURES]
        margin = self.margin.predict(x)
        total = np.maximum(self.total.predict(x), np.abs(margin))
        prob = self.probability.predict_proba(margin.reshape(-1, 1))[:, 1]
        return margin, total, prob

    def explain(self, row, target):
        pipe = getattr(self, target)
        x = pd.DataFrame([row])[FEATURES]
        contributions = pipe[0].transform(x)[0] * pipe[1].coef_
        order = np.argsort(np.abs(contributions))[::-1][:4]
        return {"baseline_points": float(pipe[1].intercept_),
                "top_factors": [{"feature": FEATURES[i], "label": LABELS[FEATURES[i]],
                                 "contribution_points": round(float(contributions[i]), 3)} for i in order],
                "other_factors_points": float(sum(contributions) - sum(contributions[order])),
                "meaning": "Contribuciones respecto al promedio de entrenamiento; no causas."}


def fit(data: pd.DataFrame, predict_season: int) -> Model:
    known = data[(data.season >= 2005) & (data.season < predict_season) & data.margin.notna()]
    calibration_year = predict_season - 1
    train = known[known.season < calibration_year]
    calibration = known[known.season == calibration_year]
    if len(train) < 800 or len(calibration) < 200:
        raise ValueError("Se necesitan al menos 800 partidos previos y 200 para calibración temporal")
    # Hiperparámetros fijos antes del backtest; el test no elige modelo ni parámetros.
    def reg(target):
        return make_pipeline(StandardScaler(), Ridge(alpha=100)).fit(train[FEATURES], train[target])
    margin, total = reg("margin"), reg("total")
    calibration_margin = margin.predict(calibration[FEATURES])
    non_ties = calibration.margin.to_numpy() != 0
    probability = LogisticRegression(C=1, max_iter=1000).fit(
        calibration_margin[non_ties].reshape(-1, 1), (calibration.margin.to_numpy()[non_ties] > 0).astype(int))
    rm = float(np.quantile(np.abs(calibration.margin - calibration_margin), .8, method="higher"))
    rt = float(np.quantile(np.abs(calibration.total - total.predict(calibration[FEATURES])), .8, method="higher"))
    # Conservamos estos regresores: así el calibrador ve la misma distribución al servir.
    return Model(margin, total, probability, rm, rt, int(train.season.max()), calibration_year,
                 len(train), float(train.total.mean()))
