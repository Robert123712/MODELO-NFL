"""Estadísticas por jugador dentro de cada simulación del partido.

Como en NBA, los jugadores salen de las mismas simulaciones que el marcador:

1. **Volumen según el guion.** Targets y acarreos del equipo = su promedio
   reciente corregido por el margen de esa simulación (el que pierde pasa más y
   corre menos), con el ruido que dejan esas regresiones en 2021-2024.
2. **Reparto.** Targets y acarreos se reparten entre jugadores con los shares
   proyectados (fase 1), multinomial.
3. **Eficiencia.** Recepciones binomiales con su % de recepción; yardas por
   recepción gamma (colas largas) y por acarreo normal, con sus tasas encogidas.
   La eficiencia del equipo sube o baja con los puntos de esa simulación respecto
   a lo esperado: un partido de 35 puntos trae más yardas que uno de 13.
4. **TD amarrados al marcador.** Los TD ofensivos de la simulación se dividen en
   pase y carrera y se reparten por la parte de TD de cada jugador.
5. **QB.** Yardas de pase = yardas por recepción de su equipo, pases completos =
   recepciones, intentos = targets / 0.956, intercepciones binomiales.
"""
import numpy as np
import pandas as pd

# Regresiones 2021-2024: volumen = a + b·esperado + c·margen del equipo (+ ruido).
TARGETS = (12.544, 0.604, -0.135, 7.36)
ACARREOS = (14.86, 0.45, 0.276, 6.35)
TARGETS_POR_INTENTO = 0.956
ELASTICIDAD_PUNTOS = 0.32   # yardas por oportunidad contra puntos sobre lo esperado
FORMA_GAMMA = 1.2           # yardas por recepción: CV ~0.9
SD_CARRERA = 5.5            # yardas por acarreo, desviación por acarreo
# El rol y la eficiencia de hoy no se conocen exactos: cada simulación mueve el
# share y la eficiencia de cada jugador con un factor lognormal (afinado 2023-24).
RUIDO_SHARE = 0.35
RUIDO_EFICIENCIA = 0.15
ESTADISTICAS = ('pass_yds', 'pass_td', 'completions', 'attempts', 'interceptions',
                'rush_yds', 'carries', 'rec', 'rec_yds', 'targets', 'td')


def simular_equipo(rng, jugadores: pd.DataFrame, margen, puntos, esperado_puntos, td_of,
                   esperado_targets, esperado_acarreos, fraccion_pase, qb_id=None) -> dict:
    """`jugadores`: una fila por disponible con proy_target_share, proy_carry_share y tasa_*.

    Devuelve {player_id: {estadística: arreglo de n simulaciones}}.
    """
    n = len(margen)
    a, b, c, sd = TARGETS
    targets_eq = np.clip(np.rint(a + b * esperado_targets + c * margen + rng.normal(0, sd, n)), 8, None).astype(int)
    a, b, c, sd = ACARREOS
    acarreos_eq = np.clip(np.rint(a + b * esperado_acarreos + c * margen + rng.normal(0, sd, n)), 6, None).astype(int)
    eficiencia = np.clip(1 + ELASTICIDAD_PUNTOS * (puntos - esperado_puntos) / max(esperado_puntos, 7), 0.7, 1.4)

    ts = jugadores.proy_target_share.fillna(0).clip(lower=0).to_numpy()
    cs = jugadores.proy_carry_share.fillna(0).clip(lower=0).to_numpy()
    ts = ts / ts.sum() if ts.sum() > 0 else np.full(len(ts), 1 / len(ts))
    cs = cs / cs.sum() if cs.sum() > 0 else np.full(len(cs), 1 / len(cs))
    k = len(ts)
    # Tope al ruido del rol: nadie triplica su share de un partido a otro.
    ts_sim = ts[None, :] * np.clip(rng.lognormal(0, RUIDO_SHARE, (n, k)), 0.5, 2.0)
    cs_sim = cs[None, :] * np.clip(rng.lognormal(0, RUIDO_SHARE, (n, k)), 0.5, 2.0)
    ts_sim /= ts_sim.sum(1, keepdims=True)
    cs_sim /= cs_sim.sum(1, keepdims=True)
    tg = rng.multinomial(targets_eq, ts_sim)          # (n, jugadores)
    ca = rng.multinomial(acarreos_eq, cs_sim)
    ef_jug = np.clip(rng.lognormal(-RUIDO_EFICIENCIA ** 2 / 2, RUIDO_EFICIENCIA, (n, k)), 0.7, 1.4)
    rec = rng.binomial(tg, jugadores.tasa_recepcion.clip(0.2, 0.95).to_numpy())
    ypr = jugadores.tasa_yardas_rec.clip(3, 25).to_numpy()
    escala = (ypr / FORMA_GAMMA)[None, :] * eficiencia[:, None] * ef_jug
    yardas_rec = np.where(rec > 0, rng.gamma(np.maximum(rec, 1e-9) * FORMA_GAMMA, escala), 0.0)
    ypc = jugadores.tasa_yardas_carrera.clip(1, 8).to_numpy()
    yardas_car = rng.normal(ca * ypc[None, :] * eficiencia[:, None] * ef_jug, np.sqrt(ca) * SD_CARRERA)
    yardas_car = np.maximum(yardas_car, -3 * ca)

    td_pase = rng.binomial(td_of, fraccion_pase)
    td_car = td_of - td_pase
    wr = jugadores.tasa_td_rec.fillna(0).clip(lower=0).to_numpy() + 1e-6
    wc = jugadores.tasa_td_carrera.fillna(0).clip(lower=0).to_numpy() + 1e-6
    # QB: no recibe sus propios pases.
    es_qb = (jugadores.grupo == 'QB').to_numpy()
    wr = np.where(es_qb, 0, wr)
    td_rec = rng.multinomial(td_pase, wr / wr.sum())
    td_carr = rng.multinomial(td_car, wc / wc.sum())

    salida = {}
    ids = jugadores.player_id.tolist()
    if qb_id is None and es_qb.any():
        qb_id = jugadores[es_qb].sort_values('proy_offense_pct', ascending=False).player_id.iloc[0]
    intentos = np.rint(targets_eq / TARGETS_POR_INTENTO).astype(int)
    for i, pid in enumerate(ids):
        x = {'rush_yds': np.rint(yardas_car[:, i]), 'carries': ca[:, i], 'rec': rec[:, i],
             'rec_yds': np.rint(yardas_rec[:, i]), 'targets': tg[:, i], 'td': td_rec[:, i] + td_carr[:, i],
             'pass_yds': np.zeros(n), 'pass_td': np.zeros(n, int), 'completions': np.zeros(n, int),
             'attempts': np.zeros(n, int), 'interceptions': np.zeros(n, int)}
        if pid == qb_id:
            x['pass_yds'] = np.rint(yardas_rec.sum(1))
            x['pass_td'] = td_pase
            x['completions'] = rec.sum(1)
            x['attempts'] = intentos
            tasa_int = jugadores.tasa_intercepcion.iloc[i]
            x['interceptions'] = rng.binomial(intentos, float(np.clip(tasa_int if np.isfinite(tasa_int) else 0.023, 0.005, 0.06)))
        salida[pid] = x
    return salida


# Mezcla de la media final (mínimos cuadrados en 2023-24, pesos negativos a 0):
# simulación, promedio de la temporada y últimos 4 partidos.
MEZCLA = {'pass_yds': (0.69, 0.235, 0.075), 'rush_yds': (0.78, 0.065, 0.155),
          'rec': (0.73, 0.225, 0.045), 'rec_yds': (0.75, 0.25, 0.0)}
COLUMNA_REAL = {'pass_yds': 'passing_yards', 'rush_yds': 'rushing_yards', 'rec': 'receptions',
                'rec_yds': 'receiving_yards'}


def mezclar(x: dict, temporada: dict, ultimos: dict) -> dict:
    """Escala cada estadística para que su media sea la mezcla; conserva la forma y
    la correlación con el marcador. Sin promedios previos se queda la simulación."""
    for k, (ws, wt, wu) in MEZCLA.items():
        media = float(np.mean(x[k]))
        t, u = temporada.get(k), ultimos.get(k)
        if media <= 0 or t is None or u is None or not (np.isfinite(t) and np.isfinite(u)):
            continue
        objetivo = ws * media + wt * t + wu * u
        x[k] = x[k] * (objetivo / media)
    return x
