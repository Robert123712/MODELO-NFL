"""Walk-forward del rol por jugador contra promedio de temporada y últimos 4.

Cada partido usa solo lo jugado antes. La disponibilidad sale de quién tuvo snaps
(equivale a conocer los inactivos ~90 minutos antes). Se califican RB, WR y TE con
al menos 3 partidos previos en la temporada, para que los promedios existan.
"""
import numpy as np
import pandas as pd

from .participacion import Parametros, preparar, ausentes, repartir, volumen_equipo, tasas_por_snap


def proyectar_historico(juegos: pd.DataFrame, p: Parametros, tasas: dict | None = None) -> pd.DataFrame:
    prep = preparar(juegos, p, tasas)
    faltan = ausentes(prep, p)
    proy = repartir(prep, faltan, p)
    return proy.merge(volumen_equipo(juegos, p), on=['game_id', 'team'], how='left')


def _baselines(juegos: pd.DataFrame) -> pd.DataFrame:
    g = juegos.sort_values(['player_id', 'season', 'week']).copy()
    llave = [g.player_id, g.season]
    for k in ('targets', 'carries', 'offense_pct', 'target_share', 'carry_share'):
        x = g.groupby(llave)[k]
        g[f'temporada_{k}'] = x.transform(lambda s: s.shift().expanding().mean())
        g[f'ult4_{k}'] = x.transform(lambda s: s.shift().rolling(4, min_periods=1).mean())
    return g[[c for c in g.columns if c.startswith(('temporada_', 'ult4_'))]]


def evaluar(proy: pd.DataFrame, temporadas, minimo_previos: int = 3) -> dict:
    datos = proy.join(_baselines(proy))
    datos = datos[datos.season.isin(temporadas) & (datos.juegos_temporada >= minimo_previos)]
    datos['modelo_targets'] = datos.proy_target_share * datos.esperado_equipo_targets
    datos['modelo_carries'] = datos.proy_carry_share * datos.esperado_equipo_carries
    datos['modelo_offense_pct'] = datos.proy_offense_pct
    datos['modelo_target_share'] = datos.proy_target_share
    datos['modelo_carry_share'] = datos.proy_carry_share
    salida = {}
    casos = {'targets': ['RB', 'WR', 'TE'], 'carries': ['RB', 'QB'], 'offense_pct': ['RB', 'WR', 'TE'],
             'target_share': ['RB', 'WR', 'TE'], 'carry_share': ['RB', 'QB']}
    for k, grupos in casos.items():
        d = datos[datos.grupo.isin(grupos)].dropna(subset=[k, f'modelo_{k}', f'temporada_{k}', f'ult4_{k}'])
        salida[k] = {'n': int(len(d))}
        for metodo in ('modelo', 'temporada', 'ult4'):
            salida[k][metodo] = round(float(np.mean(np.abs(d[k] - d[f'{metodo}_{k}']))), 4)
    return salida


def puntaje(resultado: dict) -> float:
    """Lo que se minimiza al afinar: error de targets + acarreos del modelo."""
    return resultado['targets']['modelo'] + resultado['carries']['modelo']


__all__ = ['proyectar_historico', 'evaluar', 'puntaje', 'tasas_por_snap', 'Parametros']
