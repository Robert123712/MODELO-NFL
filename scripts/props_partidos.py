"""Props por jugador y datos de video de los partidos que vienen.

    python scripts/props_partidos.py                    # todos los partidos por jugar
    python scripts/props_partidos.py --solo 2026_10_WAS_NYG

Lee data/edgebook-latest.json (el marcador esperado no cambia), consulta lesiones
y QB probable en ESPN y escribe data/latest.json y data/video/<espn_game_id>.json.
"""
import argparse
import json
from pathlib import Path
import warnings

import pandas as pd

from nfl_model.cli import write_json
from nfl_model.live_context import collect_context
from nfl_model.jugadores.emitir import emitir

warnings.filterwarnings('ignore', category=RuntimeWarning)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--solo', nargs='*', help='league_game_id o espn_game_id')
    parser.add_argument('--simulaciones', type=int, default=10_000)
    args = parser.parse_args()
    snapshot = json.loads((args.root / 'data/edgebook-latest.json').read_text())
    ahora = pd.Timestamp.now(tz='UTC')
    snapshot['games'] = [g for g in snapshot['games'] if pd.Timestamp(g['starts_at']) > ahora
                         and (not args.solo or g['league_game_id'] in args.solo or g.get('espn_game_id') in args.solo)]
    if not snapshot['games']:
        print('Sin partidos por jugar en la emisión publicada')
        return
    contexto = collect_context({**snapshot, 'as_of': ahora.isoformat()}, args.root)
    sello = pd.Timestamp(contexto['observed_at']).strftime('%Y%m%dT%H%M%S%fZ')
    write_json(args.root / f'data/context/props-{sello}.json', contexto)
    salida, videos = emitir(args.root, snapshot, contexto, n=args.simulaciones)
    write_json(args.root / 'data/latest.json', salida)
    for gid, v in videos.items():
        write_json(args.root / f'data/video/{gid}.json', v)
    for g in salida['games']:
        s = g['simulacion']
        print(f"{g['away']['code']} @ {g['home']['code']}: casa {s['p_home']:.1%}, {s['away']}-{s['home']}, "
              f"{sum(len(v) for v in g['jugadores'].values())} jugadores")
    for a in salida['avisos']:
        print('Aviso:', a)


if __name__ == '__main__':
    main()
