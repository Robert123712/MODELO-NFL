"""Estadísticas gratuitas nflverse, con actualización del estado DESPUÉS de cada semana."""
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data import ALIASES

TEAM_FIELDS = ['season', 'week', 'team', 'game_id', 'attempts', 'sacks_suffered',
               'carries', 'passing_epa', 'rushing_epa', 'passing_interceptions',
               'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost']
PLAYER_FIELDS = ['season', 'week', 'team', 'game_id', 'player_id', 'player_display_name',
                 'position', 'attempts', 'sacks_suffered', 'passing_epa', 'passing_cpoe']
EPA_FEATURES = ['epa_off_diff', 'epa_def_diff', 'epa_level', 'turnover_diff', 'pace_level']
QB_FEATURES = ['qb_epa_diff', 'qb_epa_sum', 'qb_cpoe_diff', 'qb_cpoe_sum']
ADJUSTED_FEATURES = ['adjusted_off_diff', 'adjusted_def_diff', 'adjusted_epa_level']
ADVANCED_LABELS = {'epa_off_diff': 'Eficiencia ofensiva por jugada',
                   'epa_def_diff': 'Eficiencia defensiva por jugada',
                   'epa_level': 'Eficiencia combinada por jugada',
                   'turnover_diff': 'Diferencia histórica de pérdidas de balón',
                   'pace_level': 'Volumen histórico de jugadas',
                   'qb_epa_diff': 'Diferencia de eficiencia del QB de referencia',
                   'qb_epa_sum': 'Eficiencia conjunta de quarterbacks de referencia',
                   'qb_cpoe_diff': 'Diferencia de precisión de pase sobre lo esperado',
                   'qb_cpoe_sum': 'Precisión conjunta de pase sobre lo esperado',
                   'adjusted_off_diff':'Eficiencia ofensiva ajustada por rivales previos',
                   'adjusted_def_diff':'Eficiencia defensiva ajustada por rivales previos',
                   'adjusted_epa_level':'Eficiencia combinada ajustada por rivales previos'}


def download_advanced(root: Path, season: int):
    cache = root / 'data/raw/advanced'
    cache.mkdir(parents=True, exist_ok=True)

    def one(item):
        kind, year = item
        url = f'https://github.com/nflverse/nflverse-data/releases/download/stats_{kind}/stats_{kind}_week_{year}.csv'
        path = cache / f'{kind}-{year}.csv'
        meta_path = path.with_suffix('.json')
        if path.exists() and meta_path.exists() and year < season:
            content = path.read_bytes()
            meta = json.loads(meta_path.read_text())
            if hashlib.sha256(content).hexdigest() != meta['cache_sha256']:
                raise ValueError(f'Cache alterado: {path.name}')
            return kind, pd.read_csv(io.BytesIO(content)), meta
        response = requests.get(url, timeout=90)
        if response.status_code == 404 and year == season:
            return kind, pd.DataFrame(), {'url': url, 'status': 'not_available_yet'}
        response.raise_for_status()
        raw = response.content
        fields = TEAM_FIELDS if kind == 'team' else PLAYER_FIELDS
        frame = pd.read_csv(io.BytesIO(raw), usecols=fields)
        if kind == 'player':
            frame = frame[(frame.position == 'QB') & (frame.attempts > 0)].copy()
        content = frame.to_csv(index=False).encode()
        meta = {'url': url, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'cache_sha256': hashlib.sha256(content).hexdigest(),
                'fetched_at': datetime.now(timezone.utc).isoformat(), 'rows': len(frame)}
        path.write_bytes(content)
        meta_path.write_text(json.dumps(meta, indent=2))
        return kind, frame, meta

    jobs = [(kind, year) for year in range(2004, season+1) for kind in ('team', 'player')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, jobs))
    teams = pd.concat([f for k, f, _ in results if k == 'team' and len(f)], ignore_index=True)
    players = pd.concat([f for k, f, _ in results if k == 'player' and len(f)], ignore_index=True)
    for frame in (teams, players):
        frame['team'] = frame.team.replace(ALIASES)
    if teams.duplicated(['game_id', 'team']).any() or players.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Estadísticas duplicadas: se requiere revisar la fuente')
    return teams, players, [m for _, _, m in results]


def add_advanced(data, teams, players):
    """Únicamente se incorporan stats de juegos con resultado conocido en `data`."""
    tlookup = {(r['game_id'], r['team']): r for r in teams.to_dict('records')}
    plookup = defaultdict(list)
    for r in players.to_dict('records'):
        plookup[(r['game_id'], r['team'])].append(r)
    team_history = defaultdict(lambda: deque(maxlen=12))
    qb_history = defaultdict(lambda: deque(maxlen=12))
    reference = {}
    rows = []

    def summarize(team, season):
        games = list(team_history[team])
        weights = np.array([1.0 if g['season'] == season else .5 for g in games])
        n = weights.sum()
        def avg(key, prior=0):
            return (sum(g[key] * w for g, w in zip(games, weights)) + 4*prior)/(n+4)
        return avg('off'), avg('def'), avg('turnovers', .025), avg('plays', 64), avg('adjusted_off'), avg('adjusted_def')

    def qb(team, season):
        ref = reference.get(team)
        if ref is None:
            return 0., 0., None, None, 0
        pid, name = ref
        hist = list(qb_history[pid])
        weights = np.array([1.0 if g['season'] == season else .5 for g in hist])
        # EPA por dropback con prior de 100 dropbacks; CPOE ponderado por intentos.
        denom = sum(g['dropbacks']*w for g, w in zip(hist, weights)) + 100
        epa = sum(g['epa']*w for g, w in zip(hist, weights))/denom
        valid = [(g,w) for g,w in zip(hist,weights) if np.isfinite(g['cpoe'])]
        cpoe = sum(g['cpoe']*g['attempts']*w for g,w in valid)/(100+sum(g['attempts']*w for g,w in valid))
        return epa, cpoe, pid, name, len(hist)

    for (season, week), group in data.groupby(['season', 'week'], sort=True):
        preweek = {team:summarize(team,season) for team in set(group.home_team)|set(group.away_team)}
        for row in group.to_dict('records'):
            h, a = summarize(row['home_team'], season), summarize(row['away_team'], season)
            qh, qa = qb(row['home_team'], season), qb(row['away_team'], season)
            row.update(dict(zip(EPA_FEATURES + QB_FEATURES, [h[0]-a[0], a[1]-h[1],
                (h[0]+a[0]+h[1]+a[1])/2, a[2]-h[2], (h[3]+a[3])/2,
                qh[0]-qa[0], qh[0]+qa[0], qh[1]-qa[1], qh[1]+qa[1]])))
            row.update(dict(zip(ADJUSTED_FEATURES, [h[4]-a[4],a[5]-h[5],(h[4]+a[4]+h[5]+a[5])/2])))
            row['advanced_history_min'] = min(len(team_history[row['home_team']]), len(team_history[row['away_team']]))
            for side, value in [('home', qh), ('away', qa)]:
                row[f'{side}_reference_qb_id'] = value[2]
                row[f'{side}_reference_qb'] = value[3]
            rows.append(row)
        for row in group.to_dict('records'):
            if pd.isna(row['margin']):
                continue
            for side, other in [('home','away'), ('away','home')]:
                team, opponent = row[side+'_team'], row[other+'_team']
                own = tlookup.get((row['game_id'], team))
                opp = tlookup.get((row['game_id'], opponent))
                if own and opp:
                    def volume(r):
                        return r['attempts']+r['sacks_suffered']+r['carries']
                    required = ['passing_epa','rushing_epa','attempts','sacks_suffered','carries',
                                'passing_interceptions','sack_fumbles_lost','rushing_fumbles_lost','receiving_fumbles_lost']
                    if all(np.isfinite(r[k]) for r in (own,opp) for k in required) and min(volume(own),volume(opp))>0:
                        team_history[team].append({'season': season,
                            'off': (own['passing_epa']+own['rushing_epa'])/volume(own),
                            'def': (opp['passing_epa']+opp['rushing_epa'])/volume(opp),
                            'adjusted_off': (own['passing_epa']+own['rushing_epa'])/volume(own)-preweek[opponent][1],
                            'adjusted_def': (opp['passing_epa']+opp['rushing_epa'])/volume(opp)-preweek[opponent][0],
                            'turnovers': sum(own[k] for k in ['passing_interceptions','sack_fumbles_lost','rushing_fumbles_lost','receiving_fumbles_lost'])/volume(own),
                            'plays': volume(own)})
                qbs = plookup.get((row['game_id'],team), [])
                if qbs:
                    leader = max(qbs, key=lambda q:q['attempts'])
                    reference[team] = leader['player_id'], leader['player_display_name']
                    for q in qbs:
                        if np.isfinite(q['passing_epa']) and q['attempts']+q['sacks_suffered']>0:
                            qb_history[q['player_id']].append({'season':season,'epa':q['passing_epa'],
                                'dropbacks':q['attempts']+q['sacks_suffered'],'attempts':q['attempts'],'cpoe':q['passing_cpoe']})
    return pd.DataFrame(rows)
