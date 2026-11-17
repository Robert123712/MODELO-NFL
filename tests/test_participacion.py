import numpy as np
import pandas as pd

from nfl_model.jugadores.datos import juegos_jugador
from nfl_model.jugadores.participacion import Parametros, preparar, ausentes, repartir, tasas_por_snap


def _stats(filas):
    base = {'player_display_name': 'x', 'season_type': 'REG', 'opponent_team': 'BBB', 'completions': 0,
            'attempts': 0, 'passing_yards': 0, 'passing_tds': 0, 'passing_interceptions': 0,
            'sacks_suffered': 0, 'carries': 0, 'rushing_yards': 0, 'rushing_tds': 0, 'receptions': 0,
            'targets': 0, 'receiving_yards': 0, 'receiving_tds': 0, 'receiving_air_yards': 0}
    return pd.DataFrame([{**base, **f} for f in filas])


def _liga(semanas=4, falta_en=None):
    """Equipo AAA: WR1 lleva 6 targets, WR2 3, TE 1; RB 20 acarreos. WR1 falta en `falta_en`."""
    stats, snaps, indice = [], [], []
    jugadores = {'w1': ('WR', 6, 0, .9), 'w2': ('WR', 3, 0, .7), 't1': ('TE', 1, 0, .6),
                 'r1': ('RB', 0, 20, .7), 'w3': ('WR', 0, 0, .2)}
    for pid, (pos, *_rest) in jugadores.items():
        indice.append({'gsis_id': pid, 'pfr_id': 'pfr_' + pid, 'espn_id': None, 'display_name': pid, 'position': pos})
    for w in range(1, semanas + 1):
        gid = f'2025_{w:02d}_AAA_BBB'
        for pid, (pos, tg, ca, sp) in jugadores.items():
            if w == falta_en and pid == 'w1':
                continue
            stats.append({'player_id': pid, 'position': pos, 'season': 2025, 'week': w, 'game_id': gid,
                          'team': 'AAA', 'targets': tg, 'carries': ca})
            snaps.append({'game_id': gid, 'season': 2025, 'game_type': 'REG', 'week': w, 'player': pid,
                          'pfr_player_id': 'pfr_' + pid, 'position': pos, 'team': 'AAA', 'opponent': 'BBB',
                          'offense_snaps': int(sp * 60), 'offense_pct': sp})
    return juegos_jugador(_stats(stats), pd.DataFrame(snaps), pd.DataFrame(indice))


def test_union_calcula_shares_por_equipo():
    juegos = _liga()
    partido = juegos[juegos.game_id == '2025_01_AAA_BBB'].set_index('player_id')
    assert partido.target_share['w1'] == 0.6
    assert partido.carry_share['r1'] == 1.0
    assert np.isclose(partido.target_share.sum(), 1)


def test_estado_previo_no_mira_el_partido_que_predice():
    juegos = _liga()
    prep = preparar(juegos, Parametros(previo_partidos=0)).set_index(['game_id', 'player_id'])
    assert np.isnan(prep.loc[('2025_01_AAA_BBB', 'w1'), 'previo_target_share'])
    assert np.isclose(prep.loc[('2025_02_AAA_BBB', 'w1'), 'base_target_share'], 0.6)


def test_ausente_reparte_su_share_y_el_equipo_suma_uno():
    juegos = _liga(semanas=4, falta_en=4)
    p = Parametros(previo_partidos=0, misma_posicion=1.0)
    prep = preparar(juegos, p)
    faltan = ausentes(prep, p)
    assert set(faltan[faltan.game_id == '2025_04_AAA_BBB'].player_id) == {'w1'}
    proy = repartir(prep, faltan, p)
    sem4 = proy[proy.game_id == '2025_04_AAA_BBB'].set_index('player_id')
    assert np.isclose(sem4.proy_target_share.sum(), 1)
    # Todo lo de WR1 se queda con los WR: el TE conserva su 10%.
    assert np.isclose(sem4.proy_target_share['t1'], 0.1)
    assert sem4.proy_target_share['w2'] > 0.8
    # Snaps: nunca más de 100% y no cruzan de posición.
    assert sem4.proy_offense_pct.max() <= 1
    assert np.isclose(sem4.proy_offense_pct['r1'], 0.7)


def test_prior_por_snaps_para_jugador_sin_historial():
    juegos = _liga()
    tasas = tasas_por_snap(juegos)
    assert tasas['WR']['target_share'] > 0
    prep = preparar(juegos, Parametros(), tasas)
    primero = prep[prep.week == 1]
    assert primero.base_target_share.notna().all()
