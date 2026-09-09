"""Empaqueta únicamente la pantalla pública y la última emisión del modelo."""
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'data/edgebook-latest.json').read_text())
assert data['sport'] == 'NFL' and isinstance(data['games'], list)
for game in data['games']:
    for key in ('starts_at', 'home', 'away', 'projection', 'win_probability', 'spread', 'total_points', 'pick'):
        assert key in game, f'Falta {key}'
dist = root / 'dist'
dist.mkdir(exist_ok=True)
for name in ('index.html', 'styles.css', 'app.js'):
    shutil.copyfile(root / 'web' / name, dist / name)
(dist / 'data').mkdir(exist_ok=True)
shutil.copyfile(root / 'data/edgebook-latest.json', dist / 'data/edgebook-latest.json')
print(f'Pantalla preparada: {len(data["games"])} partidos. http://127.0.0.1:8765')
