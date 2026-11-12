"""Backtest del simulador por jugador (fases 2 y 3).

    python scripts/backtest_props.py

Marcador: predicciones fuera de muestra del Ridge (reports/refined-predictions.csv).
Pesos de la mezcla y ruido afinados en 2023-24; 2025 es prueba ciega.
Escribe reports/props.json.
"""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from nfl_model.jugadores.datos import descargar, juegos_jugador, equipo_partido
from nfl_model.jugadores.participacion import tasas_por_snap
from nfl_model.jugadores.backtest_props import preparar_todo, correr, evaluar
from nfl_model.jugadores.simulador import simular, parametros

warnings.filterwarnings('ignore', category=RuntimeWarning)


def marcador(pred):
    filas, muestra = [], []
    for r in pred.itertuples():
        h, a = (r.pred_total + r.pred_margin) / 2, (r.pred_total - r.pred_margin) / 2
        s = simular(h, a, n=2000, semilla=abs(hash(r.game_id)) % 2**31)
        m = s.margen
        filas.append({'p': float((m > 0).mean() + 0.5 * (m == 0).mean()), 'q10': np.quantile(m, .1),
                      'q90': np.quantile(m, .9), 'real': r.margin, 'ridge': r.pred_probability})
        muestra.append(np.abs(m[:200]))
    d = pd.DataFrame(filas)
    a = np.concatenate(muestra)
    real = pred.margin.abs()
    sin_empate = d.real != 0
    y = d.real > 0
    return {'partidos': int(len(d)),
            'brier': {'simulador': round(float(((d.p - y)[sin_empate] ** 2).mean()), 4),
                      'ridge_logistica': round(float(((d.ridge - y)[sin_empate] ** 2).mean()), 4)},
            'cobertura_80_margen': round(float(((d.real >= d.q10) & (d.real <= d.q90)).mean()), 3),
            'margen_exacto': {k: {'simulado': round(float((a == v).mean()), 3), 'real': round(float((real == v).mean()), 3)}
                              for k, v in (('3', 3), ('7', 7), ('empate', 0))},
            'una_posesion': {'simulado': round(float((a <= 8).mean()), 3), 'real': round(float((real <= 8).mean()), 3)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--simulaciones', type=int, default=1000)
    args = parser.parse_args()
    pred = pd.read_csv(args.root / 'reports/refined-predictions.csv')
    stats, snaps, indice, fuentes = descargar(args.root, 2021, 2025)
    juegos = juegos_jugador(stats, snaps, indice)
    juegos = juegos[juegos.season <= 2025]
    proy = preparar_todo(juegos, equipo_partido(stats), tasas_por_snap(juegos[juegos.season < 2023]))
    res = correr(proy, pred, [2025], n=args.simulaciones)
    reporte = {'simulador': parametros(), 'marcador_2020_2025': marcador(pred),
               'props_prueba_ciega_2025': evaluar(res, proy),
               'metodo': ('Walk-forward. Disponibilidad = jugadores con snaps. Error medio absoluto por '
                          'jugador-partido con 3+ partidos previos en la temporada; calibración con la línea '
                          '= promedio de temporada redondeado a .5.')}
    (args.root / 'reports/props.json').write_text(json.dumps(reporte, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: reporte[k] for k in ('marcador_2020_2025',)}, indent=2))
    print(json.dumps({k: {m: v[m] for m in ('n', 'sim', 'temp', 'ult4', 'brier')} for k, v in reporte['props_prueba_ciega_2025'].items()}, indent=2))


if __name__ == '__main__':
    main()
