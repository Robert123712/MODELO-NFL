"""Eficiencia por oportunidad de cada jugador, antes de cada partido.

Sumas exponenciales (numerador y denominador por separado) en los partidos que
jugó, con vida media de 8 partidos y la temporada pasada a la mitad, encogidas
hacia el promedio de su posición con oportunidades ficticias. En NFL hay 17
partidos: sin encogimiento, un WR con 3 recepciones largas parecería élite.

- % de recepción = recepciones / targets (30 targets ficticios)
- yardas por recepción (20 recepciones ficticias)
- yardas por acarreo (40 acarreos ficticios)
- intercepciones por intento del QB (150 intentos ficticios)
- parte de los TD de pase del equipo que atrapa, encogida hacia su target share
  (6 TD ficticios); igual con los de carrera y el carry share
- por equipo: fracción de sus TD ofensivos que son de pase (10 TD ficticios)
"""
import numpy as np
import pandas as pd

VIDA_MEDIA = 8.0
PESO_PASADA = 0.5
FICTICIOS = {'recepcion': 30, 'yardas_rec': 20, 'yardas_carrera': 40, 'intercepcion': 150,
             'td_rec': 6, 'td_carrera': 6, 'fraccion_pase': 10}
# (numerador, denominador)
COCIENTES = {'recepcion': ('receptions', 'targets'), 'yardas_rec': ('receiving_yards', 'receptions'),
             'yardas_carrera': ('rushing_yards', 'carries'), 'intercepcion': ('passing_interceptions', 'attempts'),
             'td_rec': ('receiving_tds', 'equipo_td_pase'), 'td_carrera': ('rushing_tds', 'equipo_td_carrera')}


def _sumas(frame, llave, columnas, orden):
    """Suma exponencial previa a cada fila, por `llave`, sin mirar la fila misma."""
    decaimiento = 0.5 ** (1 / VIDA_MEDIA)
    f = frame.sort_values(orden)
    idx = f.index.to_numpy()
    grupo = f[llave].to_numpy()
    temporada = f.season.to_numpy()
    x = np.nan_to_num(f[columnas].to_numpy(dtype=float))
    salida = np.zeros_like(x)
    acum = np.zeros(len(columnas))
    actual, temp = None, None
    for i in range(len(f)):
        if grupo[i] != actual:
            actual, temp = grupo[i], temporada[i]
            acum[:] = 0
        elif temporada[i] != temp:
            acum *= PESO_PASADA
            temp = temporada[i]
        salida[i] = acum
        acum = acum * decaimiento + x[i]
    return pd.DataFrame(salida, index=idx, columns=[f'previo_{c}' for c in columnas]).sort_index()


def liga(juegos: pd.DataFrame) -> dict:
    """Promedios por posición (el prior del encogimiento)."""
    salida = {}
    for grupo, g in juegos.groupby('grupo'):
        salida[grupo] = {nombre: float(g[a].sum() / max(g[b].sum(), 1)) for nombre, (a, b) in COCIENTES.items()
                         if b in g}
    return salida


def preparar(juegos: pd.DataFrame, equipos: pd.DataFrame, prior: dict | None = None) -> pd.DataFrame:
    """Agrega a cada jugador-partido sus tasas previas encogidas (`tasa_*`).

    `equipos` es `datos.equipo_partido`: TD de pase y de carrera del equipo.
    Los TD del equipo se usan como denominador de la parte de TD de cada jugador.
    """
    j = juegos.merge(equipos[['game_id', 'team', 'td_pase', 'td_carrera']].rename(
        columns={'td_pase': 'equipo_td_pase', 'td_carrera': 'equipo_td_carrera'}), on=['game_id', 'team'], how='left')
    j[['equipo_td_pase', 'equipo_td_carrera']] = j[['equipo_td_pase', 'equipo_td_carrera']].fillna(0)
    j.index = juegos.index
    prior = prior or liga(j)
    columnas = sorted({c for par in COCIENTES.values() for c in par})
    previas = _sumas(j, 'player_id', columnas, ['player_id', 'season', 'week', 'game_id'])
    j = j.join(previas)
    for nombre, (a, b) in COCIENTES.items():
        if nombre in ('td_rec', 'td_carrera'):
            continue
        base = j.grupo.map(lambda g: prior.get(g, {}).get(nombre, np.nan))
        k = FICTICIOS[nombre]
        j[f'tasa_{nombre}'] = (j[f'previo_{a}'] + k * base) / (j[f'previo_{b}'] + k)
    # Parte de los TD del equipo: hacia su share de oportunidades proyectado.
    for nombre, share in (('td_rec', 'proy_target_share'), ('td_carrera', 'proy_carry_share')):
        a, b = COCIENTES[nombre]
        k = FICTICIOS[nombre]
        hacia = j[share] if share in j else 0.0
        j[f'tasa_{nombre}'] = (j[f'previo_{a}'] + k * hacia) / (j[f'previo_{b}'] + k)
    return j


def fraccion_pase(equipos: pd.DataFrame, liga_fraccion: float) -> pd.DataFrame:
    """Fracción de TD ofensivos que son de pase, por equipo antes de cada partido."""
    e = equipos.copy()
    previas = _sumas(e, 'team', ['td_pase', 'td_carrera'], ['team', 'season', 'week', 'game_id'])
    e = e.join(previas)
    k = FICTICIOS['fraccion_pase']
    e['fraccion_pase'] = (e.previo_td_pase + k * liga_fraccion) / (e.previo_td_pase + e.previo_td_carrera + k)
    return e[['game_id', 'team', 'fraccion_pase']]
