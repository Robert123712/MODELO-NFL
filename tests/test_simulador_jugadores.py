import numpy as np
import pandas as pd

from nfl_model.jugadores.simulador import simular
from nfl_model.jugadores.props import simular_equipo
from nfl_model.jugadores import video


def test_simulador_reproduce_el_marcador_esperado():
    s = simular(24.0, 19.0, n=20000, semilla=1)
    assert abs(s.casa.mean() - 24.0) < 0.6
    assert abs(s.visita.mean() - 19.0) < 0.6
    assert 8.5 < s.casa.std() < 10.5
    # Los números clave salen de contar TD y FG.
    m = np.abs(s.margen)
    assert (m == 3).mean() > (m == 2).mean() and (m == 7).mean() > (m == 5).mean()
    assert (s.casa_td_of <= (s.casa // 6)).all()


def _equipo():
    return pd.DataFrame({
        'player_id': ['qb', 'rb', 'wr', 'te'], 'grupo': ['QB', 'RB', 'WR', 'TE'],
        'proy_target_share': [0, .2, .5, .3], 'proy_carry_share': [.15, .85, 0, 0],
        'proy_offense_pct': [1, .7, .9, .8], 'tasa_recepcion': [.6, .75, .65, .7],
        'tasa_yardas_rec': [10, 8, 13, 10], 'tasa_yardas_carrera': [5, 4.3, 5, 5],
        'tasa_intercepcion': [.02, np.nan, np.nan, np.nan], 'tasa_td_rec': [0, .15, .5, .35],
        'tasa_td_carrera': [.1, .9, 0, 0]})


def test_jugadores_cuadran_con_su_equipo():
    s = simular(24.0, 19.0, n=3000, semilla=2)
    rng = np.random.default_rng(3)
    x = simular_equipo(rng, _equipo(), s.margen, s.casa, 24.0, s.casa_td_of, 32.0, 26.0, 0.6)
    recibidas = sum(x[p]['rec_yds'] for p in ('rb', 'wr', 'te'))
    assert np.allclose(x['qb']['pass_yds'], recibidas, atol=4)
    # Los TD que anotan los jugadores son los TD ofensivos de esa simulación.
    assert (sum(x[p]['td'] for p in x) == s.casa_td_of).all()
    assert (x['qb']['pass_td'] <= s.casa_td_of).all()
    # El que va perdiendo lanza más.
    pierde = s.margen < -10
    gana = s.margen > 10
    assert x['qb']['attempts'][pierde].mean() > x['qb']['attempts'][gana].mean()


def test_video_trae_el_contrato():
    s = simular(24.0, 19.0, n=2000, semilla=4)
    x = simular_equipo(np.random.default_rng(5), _equipo(), s.margen, s.casa, 24.0, s.casa_td_of, 32.0, 26.0, 0.6)
    y = simular_equipo(np.random.default_rng(6), _equipo().assign(player_id=lambda d: d.player_id + '2'),
                       -s.margen, s.visita, 19.0, s.visita_td_of, 32.0, 26.0, 0.6)
    res = {**x, **y}
    def lista(sufijo):
        return [{'player_id': p + sufijo, 'espn_id': '1', 'jugador': p, 'posicion': g, 'grupo': g, 'en_duda': False,
                 'qb_titular': g == 'QB', 'rol': {'target_share': .3, 'carry_share': .3}}
                for p, g in (('qb', 'QB'), ('rb', 'RB'), ('wr', 'WR'), ('te', 'TE'))]
    partido = {'league_game_id': '2026_10_WAS_NYG', 'espn_game_id': '401873052', 'starts_at': 'x',
               'home': {'code': 'NYG', 'name': 'Giants'}, 'away': {'code': 'WAS', 'name': 'Commanders'}}
    v = video.construir(partido, s, res, {'home': lista(''), 'away': lista('2')})
    assert v['equipos']['away']['abbr'] == 'WSH'
    assert sum(v['victorias'].values()) == 2000 and len(v['margen']['cuentas']) == 21
    assert set(v['muestra']) <= {'h', 'a', 'e'} and len(v['muestra']) == 1000
    qb = v['jugadores']['home'][0]
    assert qb['posicion'] == 'QB' and {'250', '300'} <= set(qb['sobre']['pass_yds'])
    assert qb['mejor']['stat'] == 'pass_yds'
    assert all(0.12 <= e['p_evento'] <= 0.60 for j in v['jugadores']['home'] for e in j['eventos'])
