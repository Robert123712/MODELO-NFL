import numpy as np
import pandas as pd

from nfl_model.advanced import add_advanced, EPA_FEATURES, QB_FEATURES, ADJUSTED_FEATURES, TEAM_FIELDS, PLAYER_FIELDS
from nfl_model.data import COLUMNS, prepare
from nfl_model.features import build


def sample():
    games, teams, qbs = [], [], []
    for week in range(1,4):
        game = dict.fromkeys(COLUMNS, None)
        game.update(game_id=f'2025_{week}_BUF_KC',season=2025,week=week,game_type='REG',
                    gameday=f'2025-09-{week*7:02d}',gametime='13:00',home_team='KC',away_team='BUF',
                    home_score=24,away_score=20,location='Home',espn='1')
        games.append(game)
        for team in ['KC','BUF']:
            row = dict.fromkeys(TEAM_FIELDS,0.)
            row.update(season=2025,week=week,team=team,game_id=game['game_id'],attempts=30,sacks_suffered=2,
                       carries=25,passing_epa=10 if team=='KC' else 2,rushing_epa=1,passing_interceptions=1)
            teams.append(row)
            qb = dict.fromkeys(PLAYER_FIELDS,0.)
            qb.update(season=2025,week=week,team=team,game_id=game['game_id'],player_id=team+'_QB',
                      player_display_name=team+' Quarterback',position='QB',attempts=30,sacks_suffered=2,
                      passing_epa=10,passing_cpoe=2)
            qbs.append(qb)
    data = build(prepare(pd.DataFrame(games),pd.Timestamp('2025-10-01T00:00:00Z')))
    return data,pd.DataFrame(teams),pd.DataFrame(qbs)


def test_advanced_stats_and_qb_identity_are_shifted():
    data,teams,qbs = sample()
    original = add_advanced(data,teams,qbs)
    teams.loc[teams.week>=2,'passing_epa'] = 999
    qbs.loc[qbs.week>=2,'passing_epa'] = 999
    qbs.loc[qbs.week>=2,'player_id'] = 'NEW_QB'
    changed = add_advanced(data,teams,qbs)
    fields = EPA_FEATURES+QB_FEATURES+ADJUSTED_FEATURES+['home_reference_qb_id','away_reference_qb_id']
    pd.testing.assert_frame_equal(original.loc[original.week<=2,fields],changed.loc[changed.week<=2,fields])
    assert changed.iloc[2].home_reference_qb_id == 'NEW_QB'
    assert original.iloc[0].home_reference_qb_id is None
    assert changed.iloc[2].epa_level != original.iloc[2].epa_level


def test_unfinished_results_cannot_update_advanced_history():
    data,teams,qbs = sample()
    data.loc[data.week>=2,'margin'] = np.nan
    original = add_advanced(data,teams,qbs)
    teams.loc[teams.week>=2,'passing_epa'] = 999
    qbs.loc[qbs.week>=2,'passing_epa'] = 999
    changed = add_advanced(data,teams,qbs)
    pd.testing.assert_frame_equal(original,changed)


def test_absent_stats_have_finite_priors_and_explicit_coverage():
    data,teams,qbs = sample()
    result = add_advanced(data,teams.iloc[:0],qbs.iloc[:0])
    assert np.isfinite(result[EPA_FEATURES+QB_FEATURES]).all().all()
    assert (result.advanced_history_min==0).all()
    assert result.home_reference_qb_id.isna().all()
