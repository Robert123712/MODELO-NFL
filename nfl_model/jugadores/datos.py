"""Juegos por jugador desde nflverse: estadísticas semanales + snaps ofensivos.

Una fila por jugador ofensivo de habilidad (QB, RB, FB, WR, TE) y partido en que
tuvo al menos un snap ofensivo o una jugada registrada. Los snaps vienen con ID de
Pro Football Reference y las estadísticas con ID gsis; el puente es `players.csv`.
Todo se guarda en `data/raw/jugadores` con su hash, como el resto de las descargas.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..data import ALIASES

BASE = 'https://github.com/nflverse/nflverse-data/releases/download'
POSICIONES = ('QB', 'RB', 'FB', 'WR', 'TE')
# FB se cuenta como RB: comparte acarreos y targets con ellos.
GRUPO = {'QB': 'QB', 'RB': 'RB', 'FB': 'RB', 'WR': 'WR', 'TE': 'TE'}
STATS = ['player_id', 'player_display_name', 'position', 'season', 'week', 'season_type',
         'game_id', 'team', 'opponent_team', 'completions', 'attempts', 'passing_yards',
         'passing_tds', 'passing_interceptions', 'sacks_suffered', 'carries', 'rushing_yards',
         'rushing_tds', 'receptions', 'targets', 'receiving_yards', 'receiving_tds',
         'receiving_air_yards']
SNAPS = ['game_id', 'season', 'game_type', 'week', 'player', 'pfr_player_id', 'position',
         'team', 'opponent', 'offense_snaps', 'offense_pct']
CONTEO = ['completions', 'attempts', 'passing_yards', 'passing_tds', 'passing_interceptions',
          'sacks_suffered', 'carries', 'rushing_yards', 'rushing_tds', 'receptions', 'targets',
          'receiving_yards', 'receiving_tds', 'receiving_air_yards']


def _bajar(url, path, columnas, actual):
    meta_path = path.with_suffix('.json')
    if path.exists() and meta_path.exists() and not actual:
        contenido = path.read_bytes()
        meta = json.loads(meta_path.read_text())
        if hashlib.sha256(contenido).hexdigest() != meta['cache_sha256']:
            raise ValueError(f'Cache alterado: {path.name}')
        return pd.read_csv(io.BytesIO(contenido), low_memory=False), meta
    respuesta = requests.get(url, timeout=120)
    if respuesta.status_code == 404 and actual:
        return pd.DataFrame(columns=columnas), {'url': url, 'status': 'not_available_yet'}
    respuesta.raise_for_status()
    crudo = respuesta.content
    frame = pd.read_csv(io.BytesIO(crudo), usecols=columnas, low_memory=False)
    contenido = frame.to_csv(index=False).encode()
    meta = {'url': url, 'source_sha256': hashlib.sha256(crudo).hexdigest(),
            'cache_sha256': hashlib.sha256(contenido).hexdigest(),
            'fetched_at': datetime.now(timezone.utc).isoformat(), 'rows': len(frame)}
    path.write_bytes(contenido)
    meta_path.write_text(json.dumps(meta, indent=2))
    return frame, meta


def descargar(root: Path, desde: int, hasta: int):
    """Baja estadísticas, snaps y el índice de jugadores. La temporada `hasta` siempre se refresca."""
    cache = root / 'data/raw/jugadores'
    cache.mkdir(parents=True, exist_ok=True)
    trabajos = [('stats', y) for y in range(desde, hasta + 1)] + [('snaps', y) for y in range(desde, hasta + 1)]

    def uno(item):
        tipo, year = item
        if tipo == 'stats':
            url, cols = f'{BASE}/stats_player/stats_player_week_{year}.csv', STATS
        else:
            url, cols = f'{BASE}/snap_counts/snap_counts_{year}.csv', SNAPS
        frame, meta = _bajar(url, cache / f'{tipo}-{year}.csv', cols, year == hasta)
        return tipo, frame, meta

    with ThreadPoolExecutor(max_workers=4) as pool:
        resultados = list(pool.map(uno, trabajos))
    indice, meta_indice = _bajar(f'{BASE}/players/players.csv', cache / 'players.csv',
                                 ['gsis_id', 'pfr_id', 'espn_id', 'display_name', 'position'], True)
    stats = pd.concat([f for t, f, _ in resultados if t == 'stats' and len(f)], ignore_index=True)
    snaps = pd.concat([f for t, f, _ in resultados if t == 'snaps' and len(f)], ignore_index=True)
    fuentes = [m for _, _, m in resultados] + [meta_indice]
    return stats, snaps, indice, fuentes


def juegos_jugador(stats: pd.DataFrame, snaps: pd.DataFrame, indice: pd.DataFrame) -> pd.DataFrame:
    """Une estadísticas y snaps en una fila por jugador-partido, con los totales del equipo."""
    stats = stats[stats.position.isin(POSICIONES)].copy()
    snaps = snaps[snaps.position.isin(POSICIONES) & (snaps.offense_snaps > 0)].copy()
    puente = indice.dropna(subset=['pfr_id', 'gsis_id']).drop_duplicates('pfr_id')
    snaps = snaps.merge(puente[['pfr_id', 'gsis_id']], left_on='pfr_player_id', right_on='pfr_id', how='left')
    snaps = snaps.dropna(subset=['gsis_id'])
    snaps = snaps.rename(columns={'gsis_id': 'player_id', 'player': 'nombre_snaps', 'position': 'pos_snaps'})
    for frame in (stats, snaps):
        frame['team'] = frame.team.replace(ALIASES)
    clave = ['game_id', 'player_id']
    snaps = snaps.drop_duplicates(clave)
    juntos = stats.merge(snaps[clave + ['team', 'season', 'week', 'game_type', 'nombre_snaps', 'pos_snaps', 'offense_snaps', 'offense_pct']],
                         on=clave, how='outer', suffixes=('', '_snaps'))
    for col in ('team', 'season', 'week'):
        juntos[col] = juntos[col].fillna(juntos[f'{col}_snaps'])
    juntos['season_type'] = juntos.season_type.fillna(juntos.game_type.map(lambda t: 'REG' if t == 'REG' else 'POST'))
    juntos['player_display_name'] = juntos.player_display_name.fillna(juntos.nombre_snaps)
    juntos['position'] = juntos.position.fillna(juntos.pos_snaps)
    juntos['grupo'] = juntos.position.map(GRUPO)
    juntos[CONTEO] = juntos[CONTEO].fillna(0)
    juntos['offense_snaps'] = juntos.offense_snaps.fillna(0)
    # Sin snaps registrados pero con jugadas: el snap % se desconoce, no es cero.
    juntos['offense_pct'] = juntos.offense_pct.where(juntos.offense_snaps > 0)
    juntos['season'] = juntos.season.astype(int)
    juntos['week'] = juntos.week.astype(int)
    juntos = juntos.drop(columns=[c for c in juntos.columns if c.endswith('_snaps') and c != 'offense_snaps'] + ['nombre_snaps', 'pos_snaps', 'game_type'])
    equipo = juntos.groupby(['game_id', 'team']).agg(
        equipo_targets=('targets', 'sum'), equipo_carries=('carries', 'sum'),
        equipo_intentos=('attempts', 'sum'), equipo_snaps=('offense_snaps', 'max')).reset_index()
    juntos = juntos.merge(equipo, on=['game_id', 'team'], how='left')
    juntos['target_share'] = np.where(juntos.equipo_targets > 0, juntos.targets / juntos.equipo_targets.clip(lower=1), np.nan)
    juntos['carry_share'] = np.where(juntos.equipo_carries > 0, juntos.carries / juntos.equipo_carries.clip(lower=1), np.nan)
    # Orden cronológico: el ID nflverse empieza con temporada_semana.
    return juntos.sort_values(['season', 'week', 'game_id', 'team', 'player_id']).reset_index(drop=True)
