"""Resumen legible del ajuste por QB probable de la emisión recién generada."""
import json
from pathlib import Path

data = json.loads((Path(__file__).resolve().parents[1] / 'data/edgebook-latest.json').read_text())
print(json.dumps(data.get('qb_news_adjustment', {'status': 'ausente'}), indent=2, ensure_ascii=False))
for game in data['games']:
    report = game.get('qb_adjustment', {})
    moved = '; '.join(f"{side['team']}: {side['model_reference']['name']} -> {side['news_projected']['name']}"
                      for side in report.get('teams', {}).values() if side['substituted'])
    print(f"{game['away']['code']:>3} @ {game['home']['code']:<3} total {game['total_points']:5} "
          f"{report.get('status', 'sin_datos'):12} {moved}")
