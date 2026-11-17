"""Walk-forward de los props: marcador del Ridge (predicción fuera de muestra
de `reports/refined-predictions.csv`), rol de la fase 1 y tasas previas.

Por partido se simula el marcador y, dentro, a los jugadores. Se califica la
media de cada estadística contra el promedio de temporada y los últimos 4, y la
P(over) en la línea "promedio de temporada redondeado a .5".
"""
import numpy as np
import pandas as pd

from .backtest import proyectar_historico
from .participacion import Parametros
from .props import simular_equipo, mezclar, COLUMNA_REAL
from .simulador import simular
from . import tasas as T

# Estadística -> (columna real, posiciones calificadas)
CALIFICA = {'pass_yds': ('passing_yards', ['QB']), 'pass_td': ('passing_tds', ['QB']),
            'rush_yds': ('rushing_yards', ['RB', 'QB']), 'rec': ('receptions', ['WR', 'TE', 'RB']),
            'rec_yds': ('receiving_yards', ['WR', 'TE', 'RB']), 'td': ('td_anotados', ['RB', 'WR', 'TE'])}


def preparar_todo(juegos, equipos, tasas_snap=None, p=Parametros()):
    proy = proyectar_historico(juegos, p, tasas_snap)
    proy = T.preparar(proy, equipos)
    fr = T.fraccion_pase(equipos, float(equipos.td_pase.sum() / max((equipos.td_pase + equipos.td_carrera).sum(), 1)))
    proy = proy.merge(fr, on=['game_id', 'team'], how='left')
    proy['td_anotados'] = proy.rushing_tds + proy.receiving_tds
    return promedios_previos(proy)


def promedios_previos(proy):
    """Promedio de la temporada y de los últimos 4 antes de cada partido (para la mezcla)."""
    g = proy.sort_values(['player_id', 'season', 'week'])
    for k, col in COLUMNA_REAL.items():
        x = g.groupby([g.player_id, g.season])[col]
        proy.loc[g.index, f'prom_temp_{k}'] = x.transform(lambda s: s.shift().expanding().mean())
        proy.loc[g.index, f'prom_ult4_{k}'] = x.transform(lambda s: s.shift().rolling(4, min_periods=1).mean())
    return proy


def simular_partido(filas, pred_casa, pred_visita, casa, visita, n, semilla):
    sim = simular(pred_casa, pred_visita, n=n, semilla=semilla)
    rng = np.random.default_rng(semilla + 1)
    resultado = {}
    for lado, equipo, puntos, esperado, td, signo in (
            ('home', casa, sim.casa, pred_casa, sim.casa_td_of, 1),
            ('away', visita, sim.visita, pred_visita, sim.visita_td_of, -1)):
        f = filas[filas.team == equipo]
        if f.empty:
            continue
        resultado.update(simular_equipo(rng, f, signo * sim.margen, puntos, esperado, td,
                                        float(f.esperado_equipo_targets.iloc[0]), float(f.esperado_equipo_carries.iloc[0]),
                                        float(f.fraccion_pase.iloc[0])))
    return sim, resultado


def correr(proy, predicciones, temporadas, n=1000, mezcla=True):
    filas = []
    pred = predicciones.set_index('game_id')
    for gid, f in proy[proy.season.isin(temporadas)].groupby('game_id'):
        if gid not in pred.index:
            continue
        r = pred.loc[gid]
        casa, visita = gid.split('_')[3], gid.split('_')[2]
        pc, pv = (r.pred_total + r.pred_margin) / 2, (r.pred_total - r.pred_margin) / 2
        _, res = simular_partido(f, pc, pv, casa, visita, n, abs(hash(gid)) % 2**31)
        for fila in f.itertuples():
            x = res.get(fila.player_id)
            if x is None:
                continue
            if mezcla:
                x = mezclar(x, {k: getattr(fila, f'prom_temp_{k}') for k in COLUMNA_REAL},
                            {k: getattr(fila, f'prom_ult4_{k}') for k in COLUMNA_REAL})
            d = {'game_id': gid, 'player_id': fila.player_id, 'grupo': fila.grupo, 'season': fila.season,
                 'juegos_temporada': fila.juegos_temporada}
            for k, (col, _) in CALIFICA.items():
                d[f'sim_{k}'] = float(np.mean(x[k]))
                d[f'real_{k}'] = float(getattr(fila, col))
                d[f'_sims_{k}'] = x[k]
            filas.append(d)
    return pd.DataFrame(filas)


def evaluar(res, proy, minimo_previos=3):
    g = proy.sort_values(['player_id', 'season', 'week']).copy()
    base = {}
    for k, (col, _) in CALIFICA.items():
        x = g.groupby([g.player_id, g.season])[col]
        g[f'temp_{k}'] = x.transform(lambda s: s.shift().expanding().mean())
        g[f'ult4_{k}'] = x.transform(lambda s: s.shift().rolling(4, min_periods=1).mean())
    d = res.merge(g[['game_id', 'player_id'] + [c for c in g if c.startswith(('temp_', 'ult4_'))]], on=['game_id', 'player_id'])
    d = d[d.juegos_temporada >= minimo_previos]
    salida = {}
    for k, (_, grupos) in CALIFICA.items():
        e = d[d.grupo.isin(grupos)].dropna(subset=[f'temp_{k}', f'ult4_{k}'])
        if k == 'pass_yds':
            # Solo el QB que lanzó: el titular.
            e = e[e.real_pass_yds > 0]
        real = e[f'real_{k}']
        fila = {'n': int(len(e))}
        for m in ('sim', 'temp', 'ult4'):
            fila[m] = round(float(np.mean(np.abs(real - e[f'{m}_{k}']))), 3)
        # Calibración en la línea = promedio de temporada redondeado a .5
        linea = np.floor(e[f'temp_{k}']) + 0.5
        p_over = np.array([np.mean(s > l) for s, l in zip(e[f'_sims_{k}'], linea)])
        over = (real > linea).to_numpy()
        bandas = []
        for lo, hi in ((0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)):
            m = (p_over >= lo) & (p_over < hi)
            if m.sum() >= 30:
                bandas.append({'p_modelo': round(float(p_over[m].mean()), 3), 'p_real': round(float(over[m].mean()), 3), 'n': int(m.sum())})
        fila['calibracion'] = bandas
        fila['brier'] = round(float(np.mean((p_over - over) ** 2)), 4)
        salida[k] = fila
    return salida
