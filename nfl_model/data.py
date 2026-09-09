"""Descarga auditable y lista explícita de campos permitidos."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import requests

URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
# Ni momios, líneas, clima observado al final ni QB retrospectivo entran al modelo.
COLUMNS = ["game_id", "season", "game_type", "week", "gameday", "gametime",
           "home_team", "away_team", "home_score", "away_score", "location", "espn"]
ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def download(root: Path) -> tuple[pd.DataFrame, dict]:
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    raw = response.content
    frame = pd.read_csv(io.BytesIO(raw), usecols=COLUMNS, dtype={"espn": "string"})
    if frame.empty or frame.game_id.duplicated().any():
        raise ValueError("Calendario vacío o IDs duplicados")
    metadata = {"url": URL, "sha256": hashlib.sha256(raw).hexdigest(),
                "fetched_at": datetime.now(timezone.utc).isoformat(), "rows": len(frame)}
    cache = root / "data/raw"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{metadata['sha256']}.csv").write_bytes(raw)
    (cache / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return frame, metadata


def prepare(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    frame = frame[COLUMNS].copy()
    frame = frame[frame.game_type.isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    frame["gameday"] = pd.to_datetime(frame.gameday)
    for side in ("home", "away"):
        frame[f"{side}_team"] = frame[f"{side}_team"].replace(ALIASES)
    # Los resultados del mismo día se excluyen conservadoramente (posibles parciales).
    today = as_of.tz_convert("America/New_York").date()
    future = frame.gameday.dt.date >= today
    frame.loc[future, ["home_score", "away_score"]] = float("nan")
    frame["margin"] = frame.home_score - frame.away_score
    frame["total"] = frame.home_score + frame.away_score
    return frame.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)
