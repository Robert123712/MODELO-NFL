import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import download, prepare
from .evaluate import backtest
from .export import snapshot
from .features import build
from .model import fit
from .advanced import download_advanced, add_advanced
from .live_context import collect_context, attach_context
from .qb_news import apply as apply_qb_news


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Modelo NFL sin momios")
    parser.add_argument("command", choices=["run", "backtest"])
    parser.add_argument("--as-of", help="Corte ISO con zona horaria; por defecto ahora UTC")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--days", type=int, default=8)
    parser.add_argument("--test-start", type=int, default=2020)
    parser.add_argument("--baseline", action="store_true", help="Reproducir las variables originales v0.1")
    parser.add_argument("--no-live-context", action="store_true", help="Solo motor estadístico, sin consultas de contexto actual")
    args = parser.parse_args()
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    if as_of.tzinfo is None:
        parser.error("--as-of requiere zona horaria, por ejemplo 2026-09-09T00:00:00Z")
    as_of = as_of.tz_convert('UTC')
    raw, source = download(args.root)
    data = build(prepare(raw, as_of))
    # Año NFL: enero/febrero pertenecen a la temporada anterior.
    local = as_of.tz_convert("America/New_York")
    season = local.year - (1 if local.month < 3 else 0)
    config_path = args.root / 'config/model.json'
    selected = json.loads(config_path.read_text()) if config_path.exists() and not args.baseline else {}
    capture = {}
    if selected:
        teams, players, sources = download_advanced(args.root, season)
        data = add_advanced(data, teams, players, capture=capture)
        write_json(args.root / 'data/advanced-sources.json', sources)
        source['advanced'] = {'team_rows':len(teams), 'qb_rows':len(players),
                              'manifest':'data/advanced-sources.json'}
    options = {key: selected[key] for key in ('margin_features','total_features') if key in selected}
    if args.command == "backtest":
        report, predictions = backtest(data, start=args.test_start, end=season-1, **options)
        report["source"] = source
        write_json(args.root / "reports/backtest.json", report)
        predictions[["game_id", "season", "week", "margin", "total", "pred_margin", "pred_total", "pred_probability"]].to_csv(
            args.root / "reports/backtest-predictions.csv", index=False)
        print(json.dumps(report["aggregate"], indent=2))
    else:
        model = fit(data, season, **options)
        output = snapshot(data, model, as_of, source, args.days)
        if not args.no_live_context:
            context = collect_context(output, args.root)
            stamp_context = pd.Timestamp(context['observed_at']).strftime('%Y%m%dT%H%M%S%fZ')
            write_json(args.root / f'data/context/{stamp_context}.json', context)
            output = attach_context(output, context)
            # El ajuste por noticias corre después del contexto: sustituye entradas, no pesos.
            output['qb_news_adjustment'] = apply_qb_news(output, model, capture, args.root, season)
        # Sello de emisión real, después de preparar todos los datos. No simular publicación pasada.
        issued_at = pd.Timestamp.now(tz='UTC')
        output['generated_at'] = issued_at.isoformat()
        output['emission_kind'] = 'prospective' if abs((issued_at-as_of).total_seconds()) < 3600 else 'retrospective'
        if output['emission_kind']=='prospective':
            valid_games = [g for g in output['games'] if pd.Timestamp(g['starts_at']) > issued_at]
            output['skipped_started_during_collection'] = len(output['games'])-len(valid_games)
            output['games'] = valid_games
        output['status'] = 'predictions_available' if output['games'] else 'no_upcoming_games_in_source_window'
        stamp = issued_at.strftime("%Y%m%dT%H%M%S%fZ")
        write_json(args.root / f"data/history/{stamp}.json", output)
        write_json(args.root / "data/edgebook-latest.json", output)
        (args.root / "artifacts").mkdir(exist_ok=True)
        joblib.dump(model, args.root / "artifacts/model.joblib")
        print(json.dumps({"status": output["status"], "games": len(output["games"]), "training": output["training"]}, indent=2))


if __name__ == "__main__":
    main()
