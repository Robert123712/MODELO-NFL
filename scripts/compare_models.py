"""Selección SOLO 2017–2019; comparación posterior 2020–2025, sin retocar candidatos."""
import json
from pathlib import Path

import pandas as pd
import numpy as np

from nfl_model.advanced import download_advanced, add_advanced, EPA_FEATURES, QB_FEATURES
from nfl_model.cli import write_json
from nfl_model.data import download, prepare
from nfl_model.features import build, FEATURES
from nfl_model.evaluate import backtest

root = Path(__file__).resolve().parents[1]
raw, source = download(root)
teams, players, sources = download_advanced(root, 2026)
data = add_advanced(build(prepare(raw, pd.Timestamp.now(tz='UTC'))), teams, players)
candidate_features = {'baseline': FEATURES, 'epa': FEATURES+EPA_FEATURES,
                      'epa_qb': FEATURES+EPA_FEATURES+QB_FEATURES}
development = {}
for name, features in candidate_features.items():
    report, _ = backtest(data, 2017, 2019, features, features)
    development[name] = report['aggregate']
    print('Desarrollo', name, report['aggregate']['spread_mae_points'], report['aggregate']['total_mae_points'], flush=True)
margin_choice = min(development, key=lambda c:development[c]['spread_mae_points'])
total_choice = min(development, key=lambda c:development[c]['total_mae_points'])
config = {'selection_period': [2017,2019], 'criterion': 'MAE por cabeza; candidatos fijados antes de evaluar 2020–2025',
          'margin_candidate':margin_choice,'total_candidate':total_choice,
          'margin_features':candidate_features[margin_choice], 'total_features':candidate_features[total_choice]}
baseline, bp = backtest(data)
selected, sp = backtest(data, margin_features=config['margin_features'], total_features=config['total_features'])
# Incertidumbre pareada por semana: no tratar todos los partidos como independientes.
assert bp.game_id.tolist() == sp.game_id.tolist()
paired = pd.DataFrame({'season':bp.season,'week':bp.week,
    'gain':abs(bp.total-bp.pred_total).to_numpy()-abs(sp.total-sp.pred_total).to_numpy()})
blocks = paired.groupby(['season','week']).gain.agg(['sum','count']).to_numpy()
rng = np.random.default_rng(42)
indices = rng.integers(0,len(blocks),size=(2000,len(blocks)))
resampled = blocks[indices]
gains = resampled[:,:,0].sum(axis=1)/resampled[:,:,1].sum(axis=1)
uncertainty = {'method':'bootstrap pareado por semana, 2000 remuestreos, semilla 42',
               'total_mae_improvement_points':float(paired.gain.mean()),
               'interval_95':np.quantile(gains,[.025,.975]).tolist(),
               'note':'Positivo favorece al modelo nuevo; no garantiza rendimiento futuro.'}
report = {'development': development,'selection':config, 'confirmation_baseline':baseline,'confirmation_selected':selected,
          'uncertainty': uncertainty,
          'source': source, 'advanced_sources': sources,
          'coverage': {'team_rows':len(teams),'qb_rows':len(players),
                       'evaluation_games_without_prior_advanced_data': int(((data.season.between(2020,2025)) & (data.advanced_history_min==0)).sum())},
          'interpretation': 'El historial 2020–2025 del baseline ya se había visto. Confirmación retrospectiva, no prueba prospectiva ciega.'}
write_json(root/'reports/refinement.json',report)
write_json(root/'config/model.json',config)
columns = ['game_id','season','week','margin','total','pred_margin','pred_total','pred_probability']
sp[columns].to_csv(root/'reports/refined-predictions.csv',index=False)
print(json.dumps({'selection':config,'baseline':baseline['aggregate'],'selected':selected['aggregate']},indent=2))
