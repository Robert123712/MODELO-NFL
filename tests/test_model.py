import json

import numpy as np
import pandas as pd
import pytest

from nfl_model.data import COLUMNS, prepare
from nfl_model.features import FEATURES, build
from nfl_model.model import fit
from nfl_model.export import snapshot, TEAMS


def schedule():
    rng = np.random.default_rng(17)
    rows = []
    teams = list(TEAMS)
    for season in range(2005, 2013):
        for week in range(1, 19):
            for n in range(16):
                # Datos sintéticos deliberadamente amplios para probar los cortes, no precisión.
                row = dict.fromkeys(COLUMNS, None)
                row.update(game_id=f"{season}_{week}_{n}", season=season, week=week,
                           game_type="REG", gameday=str((pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7*(week-1))).date()), gametime="13:00",
                           home_team=teams[2*n], away_team=teams[2*n+1], location="Home",
                           home_score=int(rng.integers(3, 40)), away_score=int(rng.integers(3, 40)), espn="123")
                rows.append(row)
    return pd.DataFrame(rows)


AS_OF = pd.Timestamp("2014-01-01T00:00:00Z")


def test_results_cannot_change_own_week_or_past_features():
    raw = schedule()
    original = build(prepare(raw, AS_OF))
    changed = raw.copy()
    future = (changed.season > 2011) | ((changed.season == 2011) & (changed.week >= 10))
    changed.loc[future, "home_score"] = 90
    modified = build(prepare(changed, AS_OF))
    before = (original.season < 2011) | ((original.season == 2011) & (original.week <= 10))
    pd.testing.assert_frame_equal(original.loc[before, FEATURES], modified.loc[before, FEATURES])
    assert not original.loc[~before, FEATURES].equals(modified.loc[~before, FEATURES])


def test_market_columns_are_ignored():
    raw = schedule()
    original = build(prepare(raw, AS_OF))
    for column in ["spread_line", "total_line", "home_moneyline", "away_moneyline", "wind", "home_qb_id"]:
        raw[column] = 999999
    pd.testing.assert_frame_equal(original, build(prepare(raw, AS_OF)))


def test_as_of_masks_future_results():
    data = prepare(schedule(), pd.Timestamp("2011-09-01T12:00:00Z"))
    assert data.loc[data.season >= 2011, "margin"].isna().all()


def test_calibration_and_test_are_separated():
    data = build(prepare(schedule(), AS_OF))
    model = fit(data, 2013)
    assert model.train_through == 2011
    assert model.calibration_year == 2012
    with pytest.raises(ValueError, match="800"):
        fit(data, 2008)


def test_export_sign_scores_probabilities_and_kickoff():
    data = build(prepare(schedule(), AS_OF))
    model = fit(data, 2013)
    future = data.iloc[-1:].copy()
    future["season"], future["week"] = 2013, 1
    future["gameday"] = pd.Timestamp("2013-09-08")
    future["margin"] = np.nan
    as_of = pd.Timestamp("2013-09-07T20:00:00Z")
    result = snapshot(future, model, as_of, {"test": True})
    game = result["games"][0]
    assert game["starts_at"] == "2013-09-08T17:00:00+00:00"
    assert game["spread"]["home"] == -game["spread"]["away"]
    assert game["projection"]["home_score"] + game["projection"]["away_score"] == pytest.approx(game["total_points"])
    assert game["win_probability"]["home"] + game["win_probability"]["away"] == pytest.approx(1)
    assert 0 < game["win_probability"]["home"] < 1
    assert not snapshot(future, model, pd.Timestamp("2013-09-08T18:00:00Z"), {})["games"]
    json.dumps(result, allow_nan=False)


def test_season_aliases_and_neutral_site():
    raw = schedule().iloc[:2].copy()
    raw.loc[0, "home_team"] = "OAK"
    raw.loc[0, "location"] = "Neutral"
    data = build(prepare(raw, AS_OF))
    assert data.iloc[0].home_team == "LV"
    assert data.iloc[0].home_field == 0


def test_explanation_reconstructs_regression():
    data = build(prepare(schedule(), AS_OF))
    model = fit(data, 2013)
    row = data.iloc[-1]
    for target in ("margin", "total"):
        e = model.explain(row, target)
        reconstructed = e["baseline_points"] + sum(x["contribution_points"] for x in e["top_factors"]) + e["other_factors_points"]
        predicted = getattr(model, target).predict(data.iloc[-1:][FEATURES])[0]
        assert reconstructed == pytest.approx(predicted, abs=.003)


def test_math_fingerprint_changes_with_coefficients():
    data = build(prepare(schedule(), AS_OF))
    model = fit(data,2013)
    before = model.math_fingerprint()
    assert before == model.math_fingerprint()
    model.total[1].coef_[0] += .01
    assert before != model.math_fingerprint()
