"""Regresiones interpretables y calibración temporal de la probabilidad."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, LABELS
from .advanced import ADVANCED_LABELS


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
    margin_features: list = field(default_factory=lambda: list(FEATURES))
    total_features: list = field(default_factory=lambda: list(FEATURES))

    def math_fingerprint(self):
        root = Path(__file__).parent
        state = {'margin_features':self.margin_features,'total_features':self.total_features,
                 'source':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                           for name in ('model.py','features.py','advanced.py')},
                 'calibration_year':self.calibration_year,'train_through':self.train_through}
        for target in ('margin','total'):
            pipe = getattr(self,target)
            state[target] = {'coef':pipe[1].coef_.tolist(),'intercept':float(pipe[1].intercept_),
                             'scale':pipe[0].scale_.tolist(),'mean':pipe[0].mean_.tolist()}
        state['probability'] = {'coef':self.probability.coef_.tolist(),'intercept':self.probability.intercept_.tolist()}
        state['intervals'] = [self.radius_margin,self.radius_total]
        return hashlib.sha256(json.dumps(state,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def predict(self, data):
        margin = self.margin.predict(data[self.margin_features])
        total = np.maximum(self.total.predict(data[self.total_features]), np.abs(margin))
        prob = self.probability.predict_proba(margin.reshape(-1, 1))[:, 1]
        return margin, total, prob

    def explain(self, row, target):
        pipe = getattr(self, target)
        features = getattr(self, target + '_features')
        labels = {**LABELS, **ADVANCED_LABELS}
        x = pd.DataFrame([row])[features]
        contributions = pipe[0].transform(x)[0] * pipe[1].coef_
        order = np.argsort(np.abs(contributions))[::-1][:4]
        return {"baseline_points": float(pipe[1].intercept_),
                "top_factors": [{"feature": features[i], "label": labels[features[i]],
                                 "contribution_points": round(float(contributions[i]), 3)} for i in order],
                "other_factors_points": float(sum(contributions) - sum(contributions[order])),
                "meaning": "Contribuciones respecto al promedio de entrenamiento; no causas."}


def fit(data: pd.DataFrame, predict_season: int, margin_features=None, total_features=None) -> Model:
    margin_features = list(FEATURES if margin_features is None else margin_features)
    total_features = list(FEATURES if total_features is None else total_features)
    known = data[(data.season >= 2005) & (data.season < predict_season) & data.margin.notna()]
    calibration_year = predict_season - 1
    train = known[known.season < calibration_year]
    calibration = known[known.season == calibration_year]
    if len(train) < 800 or len(calibration) < 200:
        raise ValueError("Se necesitan al menos 800 partidos previos y 200 para calibración temporal")
    # Hiperparámetros fijos antes del backtest; el test no elige modelo ni parámetros.
    def reg(target, features):
        return make_pipeline(StandardScaler(), Ridge(alpha=100)).fit(train[features], train[target])
    margin, total = reg("margin", margin_features), reg("total", total_features)
    calibration_margin = margin.predict(calibration[margin_features])
    non_ties = calibration.margin.to_numpy() != 0
    probability = LogisticRegression(C=1, max_iter=1000).fit(
        calibration_margin[non_ties].reshape(-1, 1), (calibration.margin.to_numpy()[non_ties] > 0).astype(int))
    rm = float(np.quantile(np.abs(calibration.margin - calibration_margin), .8, method="higher"))
    rt = float(np.quantile(np.abs(calibration.total - total.predict(calibration[total_features])), .8, method="higher"))
    # Conservamos estos regresores: así el calibrador ve la misma distribución al servir.
    return Model(margin, total, probability, rm, rt, int(train.season.max()), calibration_year,
                 len(train), float(train.total.mean()), margin_features, total_features)
