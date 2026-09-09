from datetime import datetime, timezone

import pandas as pd

from . import __version__

TEAMS = dict(zip(
    "ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split(),
    ["Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills", "Carolina Panthers",
     "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns", "Dallas Cowboys", "Denver Broncos",
     "Detroit Lions", "Green Bay Packers", "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars",
     "Kansas City Chiefs", "Los Angeles Rams", "Los Angeles Chargers", "Las Vegas Raiders", "Miami Dolphins",
     "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants", "New York Jets",
     "Philadelphia Eagles", "Pittsburgh Steelers", "Seattle Seahawks", "San Francisco 49ers", "Tampa Bay Buccaneers",
     "Tennessee Titans", "Washington Commanders"]))


def kickoff(row):
    if pd.isna(row.gametime):
        return None
    return pd.Timestamp(f"{row.gameday.date()} {row.gametime}", tz="America/New_York").tz_convert("UTC")


def snapshot(data, model, as_of, source, days=8):
    pending = data[(data.season == model.calibration_year + 1) & data.margin.isna()].copy()
    pending["kickoff"] = pending.apply(kickoff, axis=1) if len(pending) else pd.Series(dtype=object)
    pending = pending[pending.kickoff.notna()]
    if len(pending):
        pending = pending[(pending.kickoff > as_of) & (pending.kickoff <= as_of + pd.Timedelta(days=days))]
    games = []
    if len(pending):
        margins, totals, probabilities = model.predict(pending)
        for (_, row), margin, total, probability in zip(pending.iterrows(), margins, totals, probabilities):
            side = "home" if probability >= .5 else "away"
            home_score, away_score = round(float((total + margin) / 2), 1), round(float((total - margin) / 2), 1)
            warnings = ["Sin ajuste de lesiones, QB ni clima"]
            if row.week <= 3:
                warnings.append("Inicio de temporada: peso importante del año anterior")
            if row.history_games_min < 8:
                warnings.append("Historial limitado")
            game = {"league_game_id": row.game_id, "espn_game_id": None if pd.isna(row.espn) else str(row.espn),
                    "starts_at": row.kickoff.isoformat(), "season": int(row.season), "week": int(row.week),
                    "home": {"name": TEAMS[row.home_team], "code": row.home_team},
                    "away": {"name": TEAMS[row.away_team], "code": row.away_team},
                    "projection": {"home_score": home_score, "away_score": away_score},
                    "win_probability": {"home": round(float(probability), 6), "away": round(1-float(probability), 6)},
                    "probability_basis": "conditional_on_no_tie",
                    "pick": {"market": "moneyline", "side": side, "label": f"{row[side + '_team']} ML"},
                    "spread": {"home": round(away_score - home_score, 1), "away": round(home_score - away_score, 1),
                               "meaning": "Línea propia: local -3 significa local favorito por 3 puntos"},
                    "total_points": round(home_score + away_score, 1),
                    "intervals_80": {"home_margin": [round(float(margin-model.radius_margin), 1), round(float(margin+model.radius_margin), 1)],
                                     "total": [round(max(0, float(total-model.radius_total)), 1), round(float(total+model.radius_total), 1)]},
                    "explanations": {target: model.explain(row, target) for target in ("margin", "total")},
                    "data_warning": warnings[0], "warnings": warnings}
            games.append(game)
    return {"schema_version": "1.1", "sport": "NFL", "model_version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(), "as_of": as_of.isoformat(),
            "date_local": str(as_of.tz_convert("America/Chihuahua").date()), "games": games,
            "skipped_without_id": 0, "odds_used": False, "source": source,
            "training": {"regression_through_season": model.train_through, "calibration_season": model.calibration_year,
                         "regression_games": model.train_games},
            "status": "predictions_available" if games else "no_upcoming_games_in_source_window"}
