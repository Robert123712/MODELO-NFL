"""Qué haría el ajuste por QB probable sobre la emisión ya publicada, sin emitir.

Reprocesa una copia de `data/edgebook-latest.json` con el contexto que esa emisión
ya archivó. No escribe emisiones ni historial: sirve para revisar el efecto antes
de publicar y para comprobar que el motor actual reproduce lo publicado.
"""
import copy
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_model.advanced import add_advanced, download_advanced
from nfl_model.data import download, prepare
from nfl_model.features import build
from nfl_model.model import fit
from nfl_model.qb_news import apply as apply_qb_news

root = Path(__file__).resolve().parents[1]
published = json.loads((root / 'data/edgebook-latest.json').read_text())
season = published['games'][0]['season']
as_of = pd.Timestamp(published['as_of'])
raw, source = download(root)
teams, players, _ = download_advanced(root, season)
capture = {}
data = add_advanced(build(prepare(raw, as_of)), teams, players, capture=capture)
config = json.loads((root / 'config/model.json').read_text())
model = fit(data, season, config['margin_features'], config['total_features'])
preview = copy.deepcopy(published)
summary = apply_qb_news(preview, model, capture, root, season)
games = []
for before, after in zip(published['games'], preview['games']):
    report = after['qb_adjustment']
    games.append({'game': f'{before["away"]["code"]} @ {before["home"]["code"]}',
                  'status': report['status'],
                  'total_points': {'before': before['total_points'], 'after': after['total_points']},
                  'change_points': report.get('change_points', 0.0),
                  'data_warning': {'before': before['data_warning'], 'after': after['data_warning']},
                  'quarterbacks': {side: {'model_reference': detail['model_reference']['name'],
                                          'news_projected': (detail['news_projected'] or {}).get('name'),
                                          'substituted': detail['substituted'],
                                          'reason': detail['reason'],
                                          'qb_epa_per_dropback': detail['qb_epa_per_dropback']}
                                   for side, detail in report.get('teams', {}).items()}})
changes = [abs(g['change_points']) for g in games]
report = {'kind': 'simulacion_sobre_emision_archivada',
          'not_an_emission': 'No sustituye la emisión publicada ni genera historial.',
          'published_emission': {'generated_at': published['generated_at'],
                                 'model_version': published['model_version'],
                                 'context_observed_at': published.get('context_observed_at')},
          'engine_now': {'model_version': f'{model.math_fingerprint()[:12]}',
                         'reproduces_published_projection': all(g['status'] != 'emision_no_reproducible' for g in games)},
          'summary': summary, 'games': games,
          'total_points_change': {'games': len(games), 'max': max(changes, default=0.0),
                                  'mean_absolute': round(sum(changes) / len(changes), 3) if changes else 0.0}}
(root / 'reports/qb-news-preview.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'summary': summary, 'total_points_change': report['total_points_change'],
                  'engine_now': report['engine_now']}, indent=2, ensure_ascii=False))
for game in games:
    moved = [f'{side}: {qb["model_reference"]} -> {qb["news_projected"]}'
             for side, qb in game['quarterbacks'].items() if qb['substituted']]
    print(f'{game["game"]:12} {game["status"]:12} {game["total_points"]["before"]:5} -> '
          f'{game["total_points"]["after"]:5} ({game["change_points"]:+.1f})  {"; ".join(moved)}')
