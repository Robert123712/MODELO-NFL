"""Datos por partido para el video corto: lo que un resumen no guarda.

Mismo espíritu que `modelo_nba/video.py`: todo sale de las mismas simulaciones
que los props, jugador por jugador dentro de cada partido simulado, así que
"cuando X pasa de 100 yardas su equipo gana Y%" es una cuenta directa.
"""
import numpy as np

UMBRALES = {'pass_yds': (200, 250, 300, 350), 'pass_td': (1, 2, 3, 4), 'rush_yds': (50, 75, 100, 125),
            'rec_yds': (50, 75, 100, 125), 'rec': (4, 6, 8, 10), 'td': (1, 2)}
ETIQUETA = {'pass_yds': 'yardas por pase', 'pass_td': 'pases de touchdown', 'rush_yds': 'yardas por tierra',
            'rec_yds': 'yardas por recepción', 'rec': 'recepciones', 'td': 'touchdowns'}
PRINCIPAL = {'QB': 'pass_yds', 'RB': 'rush_yds', 'WR': 'rec_yds', 'TE': 'rec_yds'}
EVENTOS = {'QB': ('pass_yds', 'pass_td', 'rush_yds'), 'RB': ('rush_yds', 'rec_yds', 'td'),
           'WR': ('rec_yds', 'rec', 'td'), 'TE': ('rec_yds', 'rec', 'td')}
P_EVENTO = (0.12, 0.60)
MEJOR_PERCENTIL = 0.99
CINTA = 120
MUESTRA = 1000
ESPN_ABBR = {'WAS': 'WSH', 'LA': 'LAR'}
POR_LADO = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1}


def _jugador(j, x, gana):
    base = float(gana.mean())
    grupo = j['grupo']
    principal = PRINCIPAL[grupo]
    v = np.asarray(x[principal])
    # "Su mejor partido": el de 1 en 100 simulaciones. El máximo absoluto de
    # 10,000 apila todas las colas a la vez y da récords que la NFL nunca vio.
    objetivo = np.quantile(v, MEJOR_PERCENTIL)
    mejor = int(np.argmin(np.abs(v - objetivo) + (v < objetivo) * 1e6))
    eventos = []
    for s in EVENTOS[grupo]:
        for u in UMBRALES[s]:
            pasa = np.asarray(x[s]) >= u
            p = float(pasa.mean())
            if not P_EVENTO[0] <= p <= P_EVENTO[1]:
                continue
            eventos.append({'stat': s, 'umbral': u, 'texto': f'{u}+ {ETIQUETA[s]}', 'p_evento': round(p, 4),
                            'p_gana': round(float(gana[pasa].mean()), 4),
                            'p_gana_si_no': round(float(gana[~pasa].mean()), 4),
                            'empuje': round(float(gana[pasa].mean()) - base, 4)})
    return {
        'athlete_id': int(j['espn_id']) if j.get('espn_id') else None,
        'espn_id': int(j['espn_id']) if j.get('espn_id') else None,
        'jugador': j['jugador'], 'posicion': j['posicion'], 'en_duda': j['en_duda'],
        'media': {s: round(float(np.mean(x[s])), 1) for s in ('pass_yds', 'pass_td', 'rush_yds', 'rec_yds', 'rec', 'td')},
        'sobre': {s: {str(u): round(float((np.asarray(x[s]) >= u).mean()), 4) for u in us} for s, us in UMBRALES.items()},
        'anytime_td': round(float((np.asarray(x['td']) >= 1).mean()), 4),
        'mejor': {'stat': principal, 'valor': int(v[mejor]), 'td': int(np.asarray(x['td'])[mejor] + np.asarray(x['pass_td'])[mejor]),
                  'gano': bool(gana[mejor]), 'una_de_cada': int(round(1 / (1 - MEJOR_PERCENTIL)))},
        'eventos': sorted(eventos, key=lambda e: -abs(e['empuje'])),
    }


def _elegir(lista, res):
    """QB titular, 2 corredores y 4 receptores/alas con más volumen."""
    escogidos, cuenta = [], {}
    def volumen(j):
        r = j['rol']
        return r['target_share'] + r['carry_share'] + (10 if j['qb_titular'] else 0)
    for j in sorted(lista, key=volumen, reverse=True):
        g = j['grupo']
        if g == 'QB' and not j['qb_titular']:
            continue
        limite = POR_LADO[g] + (1 if g in ('WR', 'TE') and cuenta.get('WR', 0) + cuenta.get('TE', 0) < 4 else 0)
        if cuenta.get(g, 0) < limite and j['player_id'] in res:
            escogidos.append(j)
            cuenta[g] = cuenta.get(g, 0) + 1
    return escogidos


def construir(partido, sim, res, jugadores):
    margen = np.asarray(sim.margen)
    n = len(margen)
    gana = {'home': margen > 0, 'away': margen < 0}
    orillas = np.arange(-31.5, 32.5, 3.0)
    cuentas, _ = np.histogram(np.clip(margen, -31, 31), bins=orillas)
    abbr = {l: ESPN_ABBR.get(partido[l]['code'], partido[l]['code']) for l in ('home', 'away')}
    letra = np.where(margen > 0, 'h', np.where(margen < 0, 'a', 'e'))
    return {
        'liga': 'nfl',
        'game_id': int(partido['espn_game_id']) if partido.get('espn_game_id') else partido['league_game_id'],
        'league_game_id': partido['league_game_id'],
        'starts_at': partido['starts_at'],
        'simulaciones': n,
        'equipos': {l: {'abbr': abbr[l], 'nombre': partido[l].get('name')} for l in ('home', 'away')},
        'victorias': {'home': int(gana['home'].sum()), 'away': int(gana['away'].sum()), 'empates': int((margen == 0).sum())},
        'marcador_medio': {'home': round(float(sim.casa.mean()), 1), 'away': round(float(sim.visita.mean()), 1)},
        'margen': {'desde': float(orillas[0]), 'ancho': 3.0, 'cuentas': cuentas.tolist(),
                   'una_posesion': round(float((np.abs(margen) <= 8).mean()), 4),
                   'cerrados_7': round(float((np.abs(margen) <= 7).mean()), 4),
                   'palizas_20': round(float((np.abs(margen) >= 20).mean()), 4),
                   'tiempo_extra': round(float(sim.tiempo_extra.mean()), 4)},
        'cinta': [{'gana': abbr['home'] if m > 0 else abbr['away'] if m < 0 else 'EMPATE', 'por': int(abs(m))}
                  for m in margen[:CINTA]],
        'muestra': ''.join(letra[:MUESTRA]),
        'jugadores': {l: [_jugador(j, res[j['player_id']], gana[l]) for j in _elegir(jugadores[l], res)]
                      for l in ('home', 'away')},
    }
