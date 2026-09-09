"""Sustituye el historial del QB de referencia por el del QB probable reportado hoy.

Corrige una entrada equivocada; no agrega un peso nuevo. La regresión del total
conserva sus coeficientes y `reports/qb-news.json` mide cuánto cambia el error.
El QB probable proviene del depth chart y del reporte de bajas: no es un titular
confirmado, así que la sustitución se declara partido por partido.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path

import pandas as pd
import requests

from .advanced import QB_FEATURES
from .live_context import QB_MISMATCH_NOTE, normalized

INDEX_URL = 'https://github.com/nflverse/nflverse-data/releases/download/players/players.csv'
INDEX_FIELDS = ['gsis_id', 'display_name', 'espn_id', 'position', 'last_season']
RECENT_SEASONS = 3
BASIS = 'QB probable de hoy (depth chart y bajas), no titular confirmado'
GENERAL_WARNING = ('Cuando el QB probable de hoy difiere del último titular observado, el total usa el historial '
                   'del QB probable; el ganador y el margen esperado no cambian.')
ADJUSTED_NOTE = '{code}: total ajustado al QB probable ({name})'
MEASURED_EFFECT = ('Sustituir la referencia por el titular real habría movido el error medio del total '
                   '0.0014 puntos en 2020-2025, con intervalo 95% [-0.0288, 0.0305]: no se distingue del azar. '
                   'Ver reports/qb-news.json.')


def download_player_index(root: Path):
    """Puente ESPN→nflverse por identidad, descargado hoy y guardado con su hash."""
    response = requests.get(INDEX_URL, timeout=90)
    response.raise_for_status()
    raw = response.content
    frame = pd.read_csv(io.BytesIO(raw), usecols=INDEX_FIELDS, low_memory=False)
    frame = frame[(frame.position == 'QB') & frame.gsis_id.notna()].copy()
    cache = root / 'data/raw'
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'players-index.csv').write_bytes(frame.to_csv(index=False).encode())
    return frame, {'url': INDEX_URL, 'sha256': hashlib.sha256(raw).hexdigest(),
                   'fetched_at': datetime.now(timezone.utc).isoformat(), 'quarterbacks': int(len(frame))}


class PlayerIndex:
    """Resuelve el QB probable de ESPN contra el historial de nflverse."""

    def __init__(self, frame, history, season, recent_seasons=RECENT_SEASONS):
        self.by_espn = {str(int(row.espn_id)): str(row.gsis_id)
                        for row in frame.itertuples() if pd.notna(row.espn_id)}
        self.by_name = defaultdict(set)
        for player_id, name in history.names.items():
            last = history.last_season(player_id)
            if last is not None and last >= season - recent_seasons:
                self.by_name[normalized(name)].add(player_id)
        self.history = history
        self.season = season

    def has_history(self, player_id):
        return bool(player_id) and self.history.metrics(player_id, self.season)[2] > 0

    def resolve(self, projected):
        """Primero la identidad; el nombre solo decide si es único entre QB recientes."""
        by_id = self.by_espn.get(str(projected.get('espn_player_id') or ''))
        if self.has_history(by_id):
            return by_id, 'espn_player_id'
        matches = self.by_name.get(normalized(projected.get('name') or ''), set())
        if len(matches) == 1:
            return next(iter(matches)), 'nombre único entre quarterbacks recientes'
        if matches:
            return None, 'nombre repetido entre quarterbacks recientes'
        if by_id:
            return None, 'identificado sin dropbacks en el historial de nflverse'
        return None, 'no identificado en nflverse'


def qb_values(home, away):
    return dict(zip(QB_FEATURES, [home[0] - away[0], home[0] + away[0],
                                  home[1] - away[1], home[1] + away[1]]))


def side_detail(game, side, context, history, reference, index, season):
    team = game[side]['code']
    reference_id, reference_name = reference.get(team) or (None, None)
    projected = (context['teams'].get(side, {}).get('quarterback') or {}).get('projected')
    detail = {'team': team, 'model_reference': {'player_id': reference_id, 'name': reference_name},
              'news_projected': None, 'substituted': False, 'reason': None}
    chosen = reference_id
    if not projected:
        detail['reason'] = 'El contexto no reporta QB probable'
    else:
        player_id, basis = index.resolve(projected)
        detail['news_projected'] = {'espn_player_id': projected.get('espn_player_id'),
                                    'name': projected.get('name'), 'player_id': player_id,
                                    'match_basis': basis}
        if player_id is None:
            detail['reason'] = f'QB probable sin historial utilizable: {basis}'
        elif player_id == reference_id:
            detail['reason'] = 'El QB probable ya era la referencia del modelo'
        else:
            chosen, detail['substituted'] = player_id, True
    before = history.metrics(reference_id, season) if reference_id else (0., 0., 0)
    after = history.metrics(chosen, season) if chosen else (0., 0., 0)
    detail['qb_epa_per_dropback'] = {'before': round(before[0], 6), 'after': round(after[0], 6)}
    detail['qb_cpoe'] = {'before': round(before[1], 4), 'after': round(after[1], 4)}
    detail['games_in_history'] = int(after[2])
    detail['last_season_with_data'] = history.last_season(chosen) if chosen else None
    return detail, before, after


def note_warnings(game, notes, drop=()):
    """Un aviso específico vive en data_warning y en game_warnings, nunca en warnings."""
    if not notes and not drop:
        return
    kept = [note for note in game.get('game_warnings', []) if note not in drop]
    game['game_warnings'] = list(dict.fromkeys(kept + notes))
    # Sin avisos específicos, `data_warning` vuelve al aviso general, como al exportar.
    general = game.get('warnings') or [None]
    game['data_warning'] = '; '.join(game['game_warnings'])[:200] if game['game_warnings'] else general[0]


def adjust_game(game, model, history, reference, index, season):
    # Reprocesar una emisión ya ajustada no vuelve a sustituir: sus entradas ya son las nuevas.
    if (game.get('qb_adjustment') or {}).get('status') == 'aplicado':
        return {'status': 'ya_ajustada', 'reason': 'La emisión ya trae el ajuste por QB probable'}
    context = game.get('pregame_context') or {}
    if context.get('status') != 'available':
        return {'status': 'sin_contexto', 'reason': context.get('reason') or context.get('status')}
    inputs = dict((game.get('model_inputs') or {}).get('total') or {})
    if not set(QB_FEATURES) <= inputs.keys():
        return {'status': 'sin_variables_de_qb'}
    details, before, after = {}, {}, {}
    for side in ('home', 'away'):
        details[side], before[side], after[side] = side_detail(
            game, side, context, history, reference, index, season)
    published = qb_values(before['home'], before['away'])
    if any(abs(published[key] - inputs[key]) > 1e-6 for key in QB_FEATURES):
        # El historial reconstruido no explica lo publicado: no se toca la emisión.
        return {'status': 'estado_no_coincide', 'teams': details,
                'reason': 'El historial actual no reproduce las entradas de QB de la emisión'}
    margin = float(model.margin.predict(pd.DataFrame([game['model_inputs']['margin']])[model.margin_features])[0])
    previous = predict_total(model, inputs, margin)
    if previous['home_score'] != game['projection']['home_score'] or previous['away_score'] != game['projection']['away_score']:
        return {'status': 'emision_no_reproducible', 'teams': details,
                'reason': 'El modelo actual no reproduce el marcador publicado'}
    report = {'status': 'aplicado', 'basis': BASIS, 'measured_effect': MEASURED_EFFECT, 'teams': details}
    notes = []
    for side, detail in details.items():
        if detail['news_projected'] and detail['news_projected']['player_id'] is None:
            notes.append(f'{detail["team"]}: QB probable sin historial en el modelo')
        elif detail['substituted'] and detail['games_in_history'] < 4:
            notes.append(f'{detail["team"]}: historial corto del QB probable ({detail["games_in_history"]} juegos)')
        elif detail['substituted'] and (detail['last_season_with_data'] or season) < season - 1:
            notes.append(f'{detail["team"]}: historial del QB probable hasta {detail["last_season_with_data"]}')
    if not any(detail['substituted'] for detail in details.values()):
        report['status'] = 'sin_cambio'
        note_warnings(game, notes)
        return report
    # El aviso de discrepancia deja de ser cierto en cuanto el total usa al QB probable.
    drop = [QB_MISMATCH_NOTE.format(code=detail['team']) for detail in details.values() if detail['substituted']]
    notes += [ADJUSTED_NOTE.format(code=detail['team'], name=detail['news_projected']['name'])
              for detail in details.values() if detail['substituted']]
    updated = {**inputs, **qb_values(after['home'], after['away'])}
    current = predict_total(model, updated, margin)
    game['projection'] = {'home_score': current['home_score'], 'away_score': current['away_score']}
    game['total_points'] = round(current['home_score'] + current['away_score'], 1)
    game['spread'] = {**game['spread'], 'home': round(current['away_score'] - current['home_score'], 1),
                      'away': round(current['home_score'] - current['away_score'], 1)}
    game['intervals_80']['total'] = [round(max(0., current['total'] - model.radius_total), 1),
                                     round(current['total'] + model.radius_total, 1)]
    game['explanations']['total'] = model.explain(updated, 'total')
    game['model_inputs']['total'] = updated
    # Contrato: `quarterback_reference` nombra el historial que entra en la fórmula.
    game['quarterback_reference'] = dict(game.get('quarterback_reference') or {})
    for side, detail in details.items():
        used = detail['news_projected']['name'] if detail['substituted'] else detail['model_reference']['name']
        game['quarterback_reference'][side] = used or 'Sin datos'
    context['quantitative_adjustment_applied'] = True
    report.update({'total_points_before': round(previous['home_score'] + previous['away_score'], 1),
                   'total_points_after': game['total_points'],
                   'change_points': round(game['total_points'] - round(previous['home_score'] + previous['away_score'], 1), 1),
                   'unchanged': ['win_probability', 'pick', 'margen esperado'],
                   'unchanged_reason': 'Las variables de QB solo entran en la regresión del total; '
                                       'el spread publicado solo puede moverse 0.1 por redondeo de los marcadores.'})
    game['warnings'] = list(dict.fromkeys(game['warnings'] + [GENERAL_WARNING]))[:10]
    note_warnings(game, notes, drop)
    return report


def predict_total(model, inputs, margin):
    total = float(model.total.predict(pd.DataFrame([inputs])[model.total_features])[0])
    total = max(total, abs(margin))
    return {'total': total, 'home_score': round((total + margin) / 2, 1),
            'away_score': round((total - margin) / 2, 1)}


def apply(snapshot, model, capture, root: Path, season: int):
    """Ajusta cada partido con contexto disponible; un fallo de fuente no cancela la emisión."""
    history, reference = capture.get('qb_history'), capture.get('reference') or {}
    if history is None:
        return {'status': 'sin_historial_de_qb'}
    try:
        frame, source = download_player_index(root)
    except (requests.RequestException, ValueError, KeyError) as error:
        for game in snapshot['games']:
            game['qb_adjustment'] = {'status': 'indice_no_disponible'}
            note_warnings(game, ['Índice de jugadores no disponible: sin ajuste por QB probable'])
        return {'status': 'indice_no_disponible', 'reason': str(error)[:180]}
    index = PlayerIndex(frame, history, season)
    counts = defaultdict(int)
    for game in snapshot['games']:
        report = adjust_game(game, model, history, reference, index, season)
        game['qb_adjustment'] = report
        counts[report['status']] += 1
        counts['sides_substituted'] += sum(1 for side in report.get('teams', {}).values() if side['substituted'])
    return {'status': 'collected', 'basis': BASIS, 'measured_effect': MEASURED_EFFECT,
            'games_adjusted': counts['aplicado'], 'games_without_change': counts['sin_cambio'],
            'sides_substituted': counts['sides_substituted'],
            'games_without_context': counts['sin_contexto'],
            'games_not_reproducible': counts['estado_no_coincide'] + counts['emision_no_reproducible'],
            'source': source}
