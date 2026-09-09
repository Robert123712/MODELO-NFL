"""¿Sirve saber quién es el titular? Compara el QB de referencia con el titular real.

El titular real del partido predicho es un oráculo: aquí mide el techo de lo que
puede aportar una noticia de titularidad, nunca alimenta una emisión.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_model.advanced import add_advanced, download_advanced
from nfl_model.data import download, prepare
from nfl_model.features import build
from nfl_model.model import fit

SEASON = 2026
START, END = 2020, 2025


def starters_by_game(players: pd.DataFrame) -> dict:
    leaders = players.sort_values('attempts').groupby(['game_id', 'team']).tail(1)
    return {(r.game_id, r.team): (r.player_id, r.player_display_name) for r in leaders.itertuples()}


def evaluate(root: Path):
    raw, _ = download(root)
    as_of = pd.Timestamp.now(tz='UTC')
    base = build(prepare(raw, as_of))
    teams, players, _ = download_advanced(root, SEASON)
    config = json.loads((root / 'config/model.json').read_text())
    features = config['total_features']
    reference = add_advanced(base, teams, players)
    actual = add_advanced(base, teams, players, starters=starters_by_game(players))
    key = ['game_id']
    merged = reference[key + ['season', 'total', 'margin', 'home_reference_qb_id', 'away_reference_qb_id']].merge(
        actual[key + ['home_reference_qb_id', 'away_reference_qb_id']], on=key, suffixes=('_ref', '_real'))
    changed = ((merged.home_reference_qb_id_ref != merged.home_reference_qb_id_real) |
               (merged.away_reference_qb_id_ref != merged.away_reference_qb_id_real))
    seasons = {}
    frames = []
    for year in range(START, END + 1):
        trained_on_reference = fit(reference, year, total_features=features)
        trained_on_actual = fit(actual, year, total_features=features)
        test_reference = reference[(reference.season == year) & reference.margin.notna()]
        test_actual = actual[(actual.season == year) & actual.margin.notna()]
        assert list(test_reference.game_id) == list(test_actual.game_id)
        part = pd.DataFrame({'game_id': test_reference.game_id.to_numpy(), 'season': year,
                             'total': test_reference.total.to_numpy(),
                             'reference': trained_on_reference.total.predict(test_reference[features]),
                             'news_at_serving': trained_on_reference.total.predict(test_actual[features]),
                             'news_at_training': trained_on_actual.total.predict(test_actual[features])})
        frames.append(part)
        seasons[str(year)] = {name: float(mean_absolute_error(part.total, part[name]))
                              for name in ('reference', 'news_at_serving', 'news_at_training')}
    weeks = reference[['game_id', 'week']]
    combined = pd.concat(frames, ignore_index=True).merge(
        pd.DataFrame({'game_id': merged.game_id, 'starter_differs': changed}), on='game_id').merge(weeks, on='game_id')

    def summary(part):
        return {'games': int(len(part)),
                **{name: round(float(mean_absolute_error(part.total, part[name])), 4)
                   for name in ('reference', 'news_at_serving', 'news_at_training')}}

    def paired(part, variant):
        # Incertidumbre pareada por semana: los partidos de una jornada no son independientes.
        gain = (part.total - part.reference).abs().to_numpy() - (part.total - part[variant]).abs().to_numpy()
        blocks = pd.DataFrame({'season': part.season, 'week': part.week, 'gain': gain}).groupby(
            ['season', 'week']).gain.agg(['sum', 'count']).to_numpy()
        rng = np.random.default_rng(42)
        resampled = blocks[rng.integers(0, len(blocks), size=(2000, len(blocks)))]
        draws = resampled[:, :, 0].sum(axis=1) / resampled[:, :, 1].sum(axis=1)
        return {'total_mae_improvement_points': round(float(gain.mean()), 4),
                'interval_95': [round(float(v), 4) for v in np.quantile(draws, [.025, .975])],
                'note': 'Positivo favorece al titular real; un intervalo que cruza cero no distingue del azar.'}

    early = combined.week <= 2
    return {'question': 'Sustituir el QB de referencia por el titular real, ¿baja el error del total?',
            'oracle': 'El titular real del partido predicho no está disponible al emitir; acota el efecto máximo.',
            'variants': {'reference': 'QB con más intentos en el último juego observado del equipo (motor actual)',
                         'news_at_serving': 'Regresión entrenada con la referencia; al predecir se usa el historial del titular real',
                         'news_at_training': 'Regresión entrenada y evaluada con el historial del titular real'},
            'seasons': {k: {n: round(v, 4) for n, v in s.items()} for k, s in seasons.items()},
            'total_mae_points': summary(combined),
            'starter_differs_from_reference': {
                'rate': round(float(combined.starter_differs.mean()), 4),
                'rate_weeks_1_2': round(float(combined[early].starter_differs.mean()), 4),
                'rate_week_3_onward': round(float(combined[~early].starter_differs.mean()), 4),
                'games_with_change': summary(combined[combined.starter_differs]),
                'games_without_change': summary(combined[~combined.starter_differs])},
            'season_start': {'weeks_1_2': summary(combined[early]),
                             'weeks_1_2_with_change': summary(combined[early & combined.starter_differs]),
                             'week_3_onward': summary(combined[~early])},
            'uncertainty': {variant: paired(combined, variant) for variant in ('news_at_serving', 'news_at_training')},
            'uncertainty_weeks_1_2': {variant: paired(combined[early], variant)
                                      for variant in ('news_at_serving', 'news_at_training')},
            'limits': ['Oráculo retrospectivo: una noticia acierta menos que el resultado observado.',
                       'Solo evalúa el total; margen y probabilidad no usan variables de QB.',
                       'Mismos hiperparámetros fijos; no se optimiza nada con este experimento.']}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    report = evaluate(root)
    (root / 'reports/qb-news.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: report[key] for key in
                      ('total_mae_points', 'starter_differs_from_reference', 'season_start',
                       'uncertainty', 'uncertainty_weeks_1_2')}, indent=2, ensure_ascii=False))
