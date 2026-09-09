"""Evaluación acotada del ajuste de EPA por rival; conserva los informes v0.2."""
import json
from pathlib import Path
import pandas as pd

from nfl_model.advanced import download_advanced, add_advanced, EPA_FEATURES, QB_FEATURES, ADJUSTED_FEATURES
from nfl_model.cli import write_json
from nfl_model.data import download, prepare
from nfl_model.features import build, FEATURES
from nfl_model.evaluate import backtest

root = Path(__file__).resolve().parents[1]
raw, source = download(root)
teams, players, sources = download_advanced(root, 2026)
data = add_advanced(build(prepare(raw,pd.Timestamp.now(tz='UTC'))),teams,players)
sets = {'v02':(FEATURES,FEATURES+EPA_FEATURES+QB_FEATURES),
        'opponent_adjusted':(FEATURES+ADJUSTED_FEATURES,FEATURES+EPA_FEATURES+QB_FEATURES+ADJUSTED_FEATURES)}
dev = {}
for name,(margin,total) in sets.items():
    report,_ = backtest(data,2017,2019,margin,total)
    dev[name] = report['aggregate']
choices = {head:min(dev,key=lambda k:dev[k][metric]) for head,metric in [('margin','spread_mae_points'),('total','total_mae_points')]}
config = {'selection_period':[2017,2019],'criterion':'MAE; dos conjuntos fijos, evaluación posterior sin reajuste',
          'margin_candidate':choices['margin'],'total_candidate':choices['total'],
          'margin_features':sets[choices['margin']][0],'total_features':sets[choices['total']][1]}
old,_ = backtest(data,margin_features=sets['v02'][0],total_features=sets['v02'][1])
new,predictions = backtest(data,margin_features=config['margin_features'],total_features=config['total_features'])
report = {'development':dev,'selection':config,'v02':old,'selected':new,'source':source,
          'limits':['Comparación retrospectiva sobre temporadas ya examinadas; no es prueba ciega.',
                    'No se validan ajustes numéricos de lesiones, QB actual, plantilla o clima: faltan archivos previos al partido.',
                    'EPA ajustado por fuerza previa del rival, sin filtrar garbage time ni kneel-downs.']}
write_json(root/'reports/opponent-adjustment.json',report)
write_json(root/'config/model.json',config)
predictions[['game_id','season','week','margin','total','pred_margin','pred_total','pred_probability']].to_csv(root/'reports/v03-predictions.csv',index=False)
print(json.dumps({'selection':choices,'development':{k:{m:v[m] for m in ['spread_mae_points','total_mae_points']} for k,v in dev.items()},
                  'v02':{m:old['aggregate'][m] for m in ['winner_accuracy','spread_mae_points','total_mae_points']},
                  'new':{m:new['aggregate'][m] for m in ['winner_accuracy','spread_mae_points','total_mae_points']}},indent=2))
