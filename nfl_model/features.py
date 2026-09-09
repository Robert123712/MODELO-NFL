"""Estado previo a la semana: ningún resultado de esa semana se usa como entrada."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

FEATURES = ["elo_diff", "offense_diff", "defense_diff", "form_diff",
            "scoring_level", "recent_total", "home_field", "rest_diff",
            "short_rest_diff", "season_progress"]
LABELS = {
    "elo_diff": "Fuerza relativa ajustada por rivales",
    "offense_diff": "Diferencia de puntos anotados",
    "defense_diff": "Diferencia de puntos permitidos",
    "form_diff": "Diferencia de margen reciente",
    "scoring_level": "Nivel combinado de anotación",
    "recent_total": "Totales recientes de ambos equipos",
    "home_field": "Localía",
    "rest_diff": "Diferencia de descanso",
    "short_rest_diff": "Diferencia de descanso corto",
    "season_progress": "Avance de la temporada",
}


def build(frame: pd.DataFrame) -> pd.DataFrame:
    elo = defaultdict(lambda: 1500.0)
    history = defaultdict(lambda: deque(maxlen=16))
    last_day = {}
    previous_season = None
    rows = []
    for (season, week), group in frame.groupby(["season", "week"], sort=True):
        if season != previous_season:
            for team in list(elo):
                elo[team] = 1500 + (elo[team] - 1500) * (2 / 3)
            # Evitar que el receso cuente como descanso diferencial de meses.
            last_day = {}
            previous_season = season

        def stats(team):
            games = list(history[team])
            # Peso reducido a temporadas previas y cuatro partidos de prior neutral.
            weights = np.array([1.0 if s == season else 0.5 for _, _, s in games])
            denominator = weights.sum() + 4
            pf = (sum(v[0] * w for v, w in zip(games, weights)) + 4 * 22) / denominator
            pa = (sum(v[1] * w for v, w in zip(games, weights)) + 4 * 22) / denominator
            recent = games[-5:]
            margin = sum(a - b for a, b, _ in recent) / (len(recent) + 2)
            total = (sum(a + b for a, b, _ in recent) + 2 * 44) / (len(recent) + 2)
            return pf, pa, margin, total

        for game in group.to_dict("records"):
            h, a = game["home_team"], game["away_team"]
            hf = float(game["location"] != "Neutral")
            hs, aws = stats(h), stats(a)
            rest = lambda team: min(14, (game["gameday"] - last_day[team]).days) if team in last_day else 7
            hr, ar = rest(h), rest(a)
            game.update(dict(zip(FEATURES, [
                elo[h] - elo[a], hs[0] - aws[0], aws[1] - hs[1], hs[2] - aws[2],
                (hs[0] + hs[1] + aws[0] + aws[1]) / 2, (hs[3] + aws[3]) / 2,
                hf, hr - ar, float(hr <= 5) - float(ar <= 5), min(week, 18) / 18,
            ])))
            game["history_games_min"] = min(len(history[h]), len(history[a]))
            game["baseline_probability"] = 1 / (1 + 10 ** (-(elo[h] - elo[a] + 48 * hf) / 400))
            rows.append(game)
        # Commit del estado solamente después de emitir TODAS las features de la semana.
        for game in group.to_dict("records"):
            h, a = game["home_team"], game["away_team"]
            last_day[h] = last_day[a] = game["gameday"]
            if pd.isna(game["margin"]):
                continue
            hf = float(game["location"] != "Neutral")
            expected = 1 / (1 + 10 ** (-(elo[h] - elo[a] + 48 * hf) / 400))
            actual = 1.0 if game["margin"] > 0 else (0.5 if game["margin"] == 0 else 0.0)
            change = 20 * (actual - expected)
            elo[h] += change
            elo[a] -= change
            history[h].append((game["home_score"], game["away_score"], season))
            history[a].append((game["away_score"], game["home_score"], season))
    return pd.DataFrame(rows)
