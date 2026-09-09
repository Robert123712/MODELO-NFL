"""El QB probable de hoy sustituye una entrada equivocada; no altera ganador ni spread."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_model.advanced import QBHistory, QB_FEATURES
from nfl_model.features import FEATURES
from nfl_model.model import Model
from nfl_model.qb_news import PlayerIndex, adjust_game, predict_total, qb_values

SEASON = 2026
TOTAL_FEATURES = FEATURES + QB_FEATURES


def history():
    state = QBHistory()
    for week in range(1, 9):
        state.add({'season': 2025, 'week': week, 'player_id': 'REF', 'player_display_name': 'Suplente Referencia',
                   'attempts': 20, 'sacks_suffered': 2, 'passing_epa': -8., 'passing_cpoe': -4.})
        state.add({'season': 2025, 'week': week, 'player_id': 'NEWS', 'player_display_name': 'Titular Probable',
                   'attempts': 34, 'sacks_suffered': 2, 'passing_epa': 12., 'passing_cpoe': 5.})
        state.add({'season': 2025, 'week': week, 'player_id': 'RIVAL', 'player_display_name': 'Rival Quarterback',
                   'attempts': 30, 'sacks_suffered': 2, 'passing_epa': 3., 'passing_cpoe': 1.})
    return state


def player_index(state, rows=None):
    rows = rows if rows is not None else [{'gsis_id': 'NEWS', 'display_name': 'Titular Probable', 'espn_id': 991.,
                                           'position': 'QB', 'last_season': 2025}]
    return PlayerIndex(pd.DataFrame(rows), state, SEASON)


def trained_model():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(300, len(TOTAL_FEATURES))), columns=TOTAL_FEATURES)
    def regression(features, target):
        return make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(frame[features], target)
    total = 44 + 6 * frame.qb_epa_sum + rng.normal(size=300)
    margin = 2 + 3 * frame.elo_diff + rng.normal(size=300)
    probability = LogisticRegression().fit(margin.to_numpy().reshape(-1, 1), (margin > 0).astype(int))
    return Model(regression(FEATURES, margin), regression(TOTAL_FEATURES, total), probability,
                 13., 17., 2024, 2025, 300, 44., list(FEATURES), list(TOTAL_FEATURES))


def emission(model, state, projected):
    inputs = {feature: 0.2 for feature in FEATURES}
    inputs.update(qb_values(state.metrics('REF', SEASON), state.metrics('RIVAL', SEASON)))
    margin = float(model.margin.predict(pd.DataFrame([inputs])[FEATURES])[0])
    scores = predict_total(model, inputs, margin)
    quarterback = {'espn_player_id': '991', 'name': 'Titular Probable'}
    quarterback.update(projected or {})
    return {'home': {'code': 'KC'}, 'away': {'code': 'DEN'},
            'projection': {'home_score': scores['home_score'], 'away_score': scores['away_score']},
            'total_points': round(scores['home_score'] + scores['away_score'], 1),
            'win_probability': {'home': .6, 'away': .4},
            'spread': {'home': round(scores['away_score'] - scores['home_score'], 1),
                       'away': round(scores['home_score'] - scores['away_score'], 1), 'meaning': 'x'},
            'intervals_80': {'home_margin': [-10., 10.], 'total': [30., 60.]},
            'explanations': {'total': {'top_factors': []}},
            'model_inputs': {'margin': {k: inputs[k] for k in FEATURES}, 'total': dict(inputs)},
            'quarterback_reference': {'home': 'Suplente Referencia', 'away': 'Rival Quarterback'},
            'warnings': ['Aviso general'], 'game_warnings': [], 'data_warning': 'Aviso general',
            'pregame_context': {'status': 'available', 'quantitative_adjustment_applied': False,
                                'teams': {'home': {'quarterback': {'projected': None if projected == {} else quarterback}},
                                          'away': {'quarterback': {'projected': None}}}}}


def run(projected=None, tamper=None):
    state, model = history(), trained_model()
    game = emission(model, state, projected)
    if tamper:
        tamper(game)
    reference = {'KC': ('REF', 'Suplente Referencia'), 'DEN': ('RIVAL', 'Rival Quarterback')}
    report = adjust_game(game, model, state, reference, player_index(state), SEASON)
    return game, report


def test_projected_quarterback_moves_only_the_total():
    before, _ = run(projected={})
    after, report = run()
    assert report['status'] == 'aplicado'
    assert after['total_points'] > before['total_points']
    assert after['win_probability'] == before['win_probability']
    assert abs(after['spread']['home'] - before['spread']['home']) <= .1
    assert after['spread']['home'] == round(after['projection']['away_score'] - after['projection']['home_score'], 1)
    assert after['model_inputs']['total']['qb_epa_sum'] != before['model_inputs']['total']['qb_epa_sum']
    assert after['model_inputs']['margin'] == before['model_inputs']['margin']
    assert after['quarterback_reference'] == {'home': 'Titular Probable', 'away': 'Rival Quarterback'}
    assert after['pregame_context']['quantitative_adjustment_applied'] is True
    assert report['teams']['home']['substituted'] and not report['teams']['away']['substituted']


def test_unknown_projected_quarterback_leaves_the_emission_intact():
    before, _ = run(projected={})
    after, report = run(projected={'espn_player_id': '404', 'name': 'Nadie Registrado'})
    assert report['status'] == 'sin_cambio'
    assert after['projection'] == before['projection'] and after['total_points'] == before['total_points']
    assert 'sin historial' in after['data_warning']
    assert report['teams']['home']['news_projected']['match_basis'] == 'no identificado en nflverse'


def test_inputs_that_do_not_match_the_history_are_not_adjusted():
    before, _ = run(projected={})
    after, report = run(tamper=lambda game: game['model_inputs']['total'].update(qb_epa_sum=9.9))
    assert report['status'] == 'estado_no_coincide'
    assert after['projection'] == before['projection']


def test_published_projection_must_be_reproducible_before_touching_it():
    before, _ = run(projected={})
    after, report = run(tamper=lambda game: game['projection'].update(home_score=99.9))
    assert report['status'] == 'emision_no_reproducible'
    assert after['projection']['home_score'] == 99.9 and after['total_points'] == before['total_points']


def test_specific_notes_stay_out_of_general_warnings():
    game, report = run()
    assert game['data_warning'] not in game['warnings']
    assert any('total ajustado al QB probable' in note for note in game['game_warnings'])
    assert all('total ajustado' not in warning for warning in game['warnings'])


def test_identity_wins_and_repeated_names_stay_unresolved():
    state = history()
    twins = [{'gsis_id': 'NEWS', 'display_name': 'Titular Probable', 'espn_id': 991., 'position': 'QB', 'last_season': 2025}]
    state.add({'season': 2025, 'week': 9, 'player_id': 'CLONE', 'player_display_name': 'Titular Probable',
               'attempts': 10, 'sacks_suffered': 1, 'passing_epa': 1., 'passing_cpoe': 1.})
    index = player_index(state, twins)
    assert index.resolve({'espn_player_id': '991', 'name': 'Titular Probable'}) == ('NEWS', 'espn_player_id')
    assert index.resolve({'espn_player_id': '', 'name': 'Titular Probable'})[0] is None
    assert index.resolve({'espn_player_id': '', 'name': 'Rival Quarterback'}) == (
        'RIVAL', 'nombre único entre quarterbacks recientes')


def test_stale_history_is_not_used_for_name_matching():
    state = QBHistory()
    state.add({'season': 2019, 'week': 1, 'player_id': 'OLD', 'player_display_name': 'Retirado Antiguo',
               'attempts': 30, 'sacks_suffered': 2, 'passing_epa': 5., 'passing_cpoe': 2.})
    index = player_index(state, [])
    assert index.resolve({'espn_player_id': '1', 'name': 'Retirado Antiguo'})[0] is None


def test_metrics_reproduce_the_engine_formula():
    # Temporada anterior con peso .5, prior de 100 dropbacks, CPOE ponderado por intentos.
    state = history()
    epa, cpoe, count = state.metrics('NEWS', SEASON)
    assert count == 8 and epa == pytest.approx(8 * 12. * .5 / (8 * 36 * .5 + 100))
    assert cpoe == pytest.approx(8 * 5. * 34 * .5 / (100 + 8 * 34 * .5))
    assert qb_values((1., 2., 0), (.5, 1., 0))['qb_epa_diff'] == pytest.approx(.5)
