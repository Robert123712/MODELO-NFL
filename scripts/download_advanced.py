from pathlib import Path
from nfl_model.advanced import download_advanced

root = Path(__file__).resolve().parents[1]
teams, players, sources = download_advanced(root, 2026)
print(f'{len(teams)} filas de equipo; {len(players)} filas QB; {len(sources)} fuentes')
