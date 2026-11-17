"""Backtest del rol por jugador (fase 1): afina en 2023-24 y prueba a ciegas en 2025.

    python scripts/backtest_participacion.py            # solo evalúa los parámetros fijados
    python scripts/backtest_participacion.py --afinar   # repite la búsqueda en 2023-24

Escribe reports/participacion.json. La prueba de 2025 nunca participa en la búsqueda.
"""
import argparse
import itertools
import json
from functools import partial
from multiprocessing import Pool
import warnings
from pathlib import Path

from nfl_model.jugadores.datos import descargar, juegos_jugador
from nfl_model.jugadores.backtest import Parametros, evaluar, proyectar_historico, puntaje, tasas_por_snap

AFINAR, PRUEBA = [2023, 2024], [2025]
REJILLA = {'vida_media': [2, 3, 4, 6, 8], 'peso_temporada_pasada': [0.25, 0.5, 1.0],
           'previo_partidos': [0, 0.5, 1, 2], 'misma_posicion': [0, 0.5, 1]}


def _probar(juegos, tasas, valores):
    warnings.filterwarnings('ignore')
    q = Parametros(**dict(zip(REJILLA, valores)))
    return puntaje(evaluar(proyectar_historico(juegos, q, tasas), AFINAR)), q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--afinar', action='store_true')
    args = parser.parse_args()
    stats, snaps, indice, fuentes = descargar(args.root, 2021, max(PRUEBA))
    juegos = juegos_jugador(stats, snaps, indice)
    juegos = juegos[juegos.season <= max(PRUEBA)]
    # El prior por snaps se mide antes de la ventana de afinación.
    tasas = tasas_por_snap(juegos[juegos.season < min(AFINAR)])
    busqueda = []
    p = Parametros()
    if args.afinar:
        with Pool() as pool:
            busqueda = pool.map(partial(_probar, juegos, tasas), list(itertools.product(*REJILLA.values())))
        busqueda.sort(key=lambda x: x[0])
        p = busqueda[0][1]
    proy = proyectar_historico(juegos, p, tasas)
    reporte = {
        'parametros': p.como_dict(),
        'tasas_por_snap': tasas,
        'afinacion': {'temporadas': AFINAR, 'resultado': evaluar(proy, AFINAR)},
        'prueba_ciega': {'temporadas': PRUEBA, 'resultado': evaluar(proy, PRUEBA)},
        'metodo': ('Walk-forward; disponibilidad = jugadores con snaps (inactivos conocidos). '
                   'Error medio absoluto por jugador-partido; RB/WR/TE (acarreos: RB/QB) con 3+ '
                   'partidos previos en la temporada. Modelo de conteo = share proyectado por '
                   'volumen esperado del equipo.'),
        'fuentes': fuentes,
    }
    if busqueda:
        reporte['afinacion']['mejores'] = [{'puntaje': round(s, 4), **q.como_dict()} for s, q in busqueda[:5]]
    salida = args.root / 'reports/participacion.json'
    salida.write_text(json.dumps(reporte, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: reporte[k] for k in ('parametros', 'afinacion', 'prueba_ciega')}, indent=2))


if __name__ == '__main__':
    main()
