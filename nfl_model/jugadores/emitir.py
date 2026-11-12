"""Props por jugador y datos de video para los partidos que vienen.

Toma la emisión publicada (`data/edgebook-latest.json`: marcador esperado del
Ridge, que no se toca) y el contexto de ESPN (lesiones y QB probable), arma a
los disponibles de cada equipo, corre 10,000 simulaciones por partido y escribe:

- `data/latest.json`: por partido, marcador simulado y props por jugador.
- `data/video/<espn_game_id>.json`: lo que necesita el video corto (contrato en
  el hilo del video: victorias, márgenes, cinta, eventos por jugador).

El contrato 1.1 con Edgebook (`edgebook-latest.json`) no cambia.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .datos import descargar, juegos_jugador, equipo_partido, GRUPO
from .participacion import tasas_por_snap
from .backtest_props import preparar_todo, simular_partido
from .props import mezclar, COLUMNA_REAL
from .simulador import parametros as parametros_simulador
from . import video

SIMULACIONES = 10_000
FUERA = {'out', 'doubtful', 'injured reserve', 'suspension', 'suspended', 'pup', 'reserve',
         'physically unable to perform', 'non-football injury'}
EN_DUDA = {'questionable'}
VENTANA = 3
ESTADISTICAS = ('pass_yds', 'pass_td', 'completions', 'attempts', 'interceptions',
                'rush_yds', 'carries', 'rec', 'rec_yds', 'targets', 'td')
PRINCIPALES = {'QB': ('pass_yds', 'pass_td', 'completions', 'interceptions', 'rush_yds'),
               'RB': ('rush_yds', 'carries', 'rec', 'rec_yds', 'td'),
               'WR': ('rec', 'rec_yds', 'targets', 'td'),
               'TE': ('rec', 'rec_yds', 'targets', 'td')}
ABANICO = {'pass_yds': 120, 'rush_yds': 60, 'rec_yds': 60, 'rec': 5, 'carries': 10, 'targets': 6,
           'completions': 10, 'pass_td': 3, 'interceptions': 2, 'td': 2}


def _espn_a_gsis(indice):
    i = indice.dropna(subset=['espn_id', 'gsis_id'])
    return {str(int(float(e))): g for e, g in zip(i.espn_id, i.gsis_id)}


def _estado_lesiones(contexto_juego, lado, espn_gsis):
    equipo = (contexto_juego or {}).get('teams', {}).get(lado, {})
    estado = {}
    for p in equipo.get('injuries', {}).get('players', []):
        g = espn_gsis.get(str(p.get('espn_player_id')))
        if g:
            estado[g] = (p.get('status') or '').lower()
    for p in equipo.get('roster', {}).get('players', []):
        g = espn_gsis.get(str(p.get('espn_player_id')))
        s = (p.get('status') or '').lower()
        if g and s in FUERA and g not in estado:
            estado[g] = s
    qb = (equipo.get('quarterback') or {}).get('projected') or {}
    return estado, espn_gsis.get(str(qb.get('espn_player_id'))) if qb else None


def filas_futuras(juegos, partido, contexto_juego, indice, espn_gsis):
    """Una fila sin estadísticas por cada disponible del partido que viene."""
    gid = partido['league_game_id']
    season, week = int(gid.split('_')[0]), int(gid.split('_')[1])
    filas, avisos = [], []
    ultimo = juegos.sort_values(['season', 'week']).groupby('player_id').tail(1).set_index('player_id')
    for lado in ('home', 'away'):
        equipo = partido[lado]['code']
        previos = (juegos[juegos.team == equipo][['game_id', 'season', 'week']].drop_duplicates()
                   .sort_values(['season', 'week']).tail(VENTANA).game_id)
        candidatos = set(juegos[juegos.game_id.isin(previos) & (juegos.team == equipo)].player_id)
        candidatos = {p for p in candidatos if ultimo.loc[p, 'team'] == equipo}
        estado, qb = _estado_lesiones(contexto_juego, lado, espn_gsis)
        fuera = {p for p, s in estado.items() if s in FUERA}
        disponibles = candidatos - fuera
        qbs = [p for p in disponibles if ultimo.loc[p, 'grupo'] == 'QB']
        if qb and qb not in fuera:
            if qb not in disponibles:
                disponibles.add(qb)
                avisos.append(f'{equipo}: QB probable sin partidos recientes con el equipo')
            disponibles -= set(qbs) - {qb}
        elif qbs:
            recientes = juegos[juegos.player_id.isin(qbs)].sort_values(['season', 'week']).groupby('player_id').tail(1)
            titular = recientes.sort_values(['season', 'week', 'attempts']).player_id.iloc[-1]
            disponibles -= set(qbs) - {titular}
            qb = titular
        for p in sorted(disponibles):
            if p in ultimo.index:
                info = ultimo.loc[p]
                nombre, posicion = info.player_display_name, info.position
            else:
                fila = indice[indice.gsis_id == p].iloc[0]
                nombre, posicion = fila.display_name, fila.position
            if GRUPO.get(posicion) is None:
                continue
            filas.append({'game_id': gid, 'player_id': p, 'player_display_name': nombre, 'position': posicion,
                          'grupo': GRUPO[posicion], 'season': season, 'week': week, 'season_type': 'REG',
                          'team': equipo, 'opponent_team': partido['away' if lado == 'home' else 'home']['code'],
                          'en_duda': estado.get(p) in EN_DUDA, 'qb_titular': p == qb, 'futuro': True})
    return pd.DataFrame(filas), avisos


def _resumen_jugador(x, grupo):
    salida = {}
    for k in PRINCIPALES.get(grupo, ()):
        v = np.asarray(x[k], dtype=float)
        media = float(v.mean())
        desde = max(0.5, np.floor(media - ABANICO[k] / 2) + 0.5)
        lineas = np.arange(desde, desde + ABANICO[k] + 1, 1.0)
        salida[k] = {'media': round(media, 1), 'mediana': float(np.median(v)),
                     'p10': float(np.quantile(v, 0.1)), 'p90': float(np.quantile(v, 0.9)),
                     'over': {'desde': float(lineas[0]), 'p': [round(float((v > l).mean()), 4) for l in lineas]}}
    salida['anytime_td'] = round(float((np.asarray(x['td']) >= 1).mean()), 4)
    return salida


def emitir(root: Path, snapshot: dict, contexto: dict, n: int = SIMULACIONES, solo=None):
    juegos_snap = [g for g in snapshot['games'] if solo is None or g['league_game_id'] in solo
                   or g.get('espn_game_id') in solo]
    if not juegos_snap:
        return None
    season = int(juegos_snap[0]['league_game_id'].split('_')[0])
    stats, snaps, indice, fuentes = descargar(root, 2021, season)
    juegos = juegos_jugador(stats, snaps, indice)
    equipos = equipo_partido(stats)
    espn_gsis = _espn_a_gsis(indice)
    gsis_espn = {g: e for e, g in espn_gsis.items()}
    futuras, avisos, eq_futuros = [], [], []
    for p in juegos_snap:
        filas, av = filas_futuras(juegos, p, contexto.get('games', {}).get(p['league_game_id']), indice, espn_gsis)
        futuras.append(filas)
        avisos += av
        for lado in ('home', 'away'):
            eq_futuros.append({'game_id': p['league_game_id'], 'team': p[lado]['code'],
                               'season': season, 'week': int(p['league_game_id'].split('_')[1]),
                               'td_pase': 0, 'td_carrera': 0, 'fg': 0})
    futuras = pd.concat(futuras, ignore_index=True)
    todos = pd.concat([juegos.assign(futuro=False), futuras], ignore_index=True)
    for c in ('targets', 'carries', 'receptions', 'receiving_yards', 'rushing_yards', 'attempts',
              'passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_tds', 'receiving_tds',
              'equipo_targets', 'equipo_carries'):
        todos[c] = todos[c].fillna(0)
    todos = todos.sort_values(['season', 'week', 'game_id', 'team', 'player_id']).reset_index(drop=True)
    eqs = pd.concat([equipos, pd.DataFrame(eq_futuros)], ignore_index=True)
    proy = preparar_todo(todos, eqs, tasas_por_snap(juegos[juegos.season < 2023]))
    proy = proy[proy.futuro == True]  # noqa: E712

    partidos, videos = [], {}
    for p in juegos_snap:
        gid = p['league_game_id']
        filas = proy[proy.game_id == gid]
        casa, visita = p['home']['code'], p['away']['code']
        pc, pv = p['projection']['home_score'], p['projection']['away_score']
        semilla = int(p.get('espn_game_id') or 0) % 2**31
        sim, res = simular_partido(filas, pc, pv, casa, visita, n, semilla)
        jugadores = {'home': [], 'away': []}
        for f in filas.sort_values('proy_offense_pct', ascending=False).itertuples():
            x = res.get(f.player_id)
            if x is None:
                continue
            x = mezclar(x, {k: getattr(f, f'prom_temp_{k}') for k in COLUMNA_REAL},
                        {k: getattr(f, f'prom_ult4_{k}') for k in COLUMNA_REAL})
            res[f.player_id] = x
            lado = 'home' if f.team == casa else 'away'
            jugadores[lado].append({
                'player_id': f.player_id, 'espn_id': gsis_espn.get(f.player_id), 'jugador': f.player_display_name,
                'posicion': f.position, 'grupo': f.grupo, 'en_duda': bool(f.en_duda),
                'qb_titular': bool(f.qb_titular),
                'rol': {'snaps': round(float(f.proy_offense_pct), 3), 'target_share': round(float(f.proy_target_share), 3),
                        'carry_share': round(float(f.proy_carry_share), 3)},
                'props': _resumen_jugador(x, f.grupo)})
        m = sim.margen
        partidos.append({'league_game_id': gid, 'espn_game_id': p.get('espn_game_id'), 'starts_at': p['starts_at'],
                         'home': p['home'], 'away': p['away'],
                         'marcador_modelo': {'home': pc, 'away': pv},
                         'simulacion': {'simulaciones': n, 'p_home': round(float((m > 0).mean()), 4),
                                        'p_away': round(float((m < 0).mean()), 4),
                                        'p_empate': round(float((m == 0).mean()), 4),
                                        'home': round(float(sim.casa.mean()), 1), 'away': round(float(sim.visita.mean()), 1),
                                        'tiempo_extra': round(float(sim.tiempo_extra.mean()), 4)},
                         'jugadores': jugadores})
        videos[p.get('espn_game_id') or gid] = video.construir(p, sim, res, jugadores)
    salida = {'generated_at': datetime.now(timezone.utc).isoformat(), 'sport': 'NFL',
              'base': {'emision': snapshot.get('generated_at'), 'model_version': snapshot.get('model_version')},
              'contexto_observado': contexto.get('observed_at'),
              'simulador': parametros_simulador(), 'avisos': avisos, 'games': partidos,
              'nota': 'Experimental: props en prueba (docs/JUGADORES.md). No entra al contrato 1.1.'}
    return salida, videos
