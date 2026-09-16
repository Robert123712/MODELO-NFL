"""Rendimiento de las emisiones ya publicadas, calificado contra el resultado real.

El backtest mide un modelo sobre partidos que ya estaban en el archivo. Esto mide
otra cosa: lo que el modelo dijo ANTES de cada partido, en una emisión sellada con
su huella, contra lo que pasó. Es la única prueba que no se puede ajustar hacia
atrás, y por eso es la que vale ante alguien que dude.

Los momios de cierre entran solo para comparar, nunca como entrada del motor.
`data/COLUMNS` sigue sin incluirlos y la matemática no los ve.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Columnas que se leen SOLO para evaluar. Viven aparte de `data.COLUMNS` a
# proposito: esa lista es lo que el motor puede ver, y los momios no entran ahi
# ni por descuido. Aqui se leen del mismo archivo ya descargado y verificado.
COLUMNAS_EVALUACION = ["game_id", "season", "week", "gameday", "home_score", "away_score",
                       "spread_line", "total_line", "home_moneyline", "away_moneyline"]

SCHEMA_VERSION = "1.0"
SPORT = "NFL"
# Debajo de esto los números son ruido y la pantalla debe decirlo.
MINIMO_SIGNIFICATIVO = 100
# Mismos tramos que publica MLB: así las dos pantallas se leen igual.
TRAMOS = [(0.0, 0.35), (0.35, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.65), (0.65, 1.01)]


def emisiones(root: Path):
    """Cada partido con la PRIMERA emisión que lo publicó bajo cada versión.

    Una versión emite varias veces el mismo partido mientras se acerca el juego.
    Quedarse con la primera es una regla fija: elegir después cuál emisión contar
    sería escoger el resultado que más conviene.
    """
    elegidas = defaultdict(dict)
    for archivo in sorted((root / "data/history").glob("*.json")):
        emision = json.loads(archivo.read_text())
        if emision.get("sport") != SPORT:
            continue
        for juego in emision.get("games", []):
            versiones = elegidas[emision["model_version"]]
            versiones.setdefault(juego["league_game_id"], {**juego, "emitted_at": emision["generated_at"]})
    return elegidas


def resultados(frame: pd.DataFrame):
    """Marcador final y línea de cierre por partido, de la fuente que ya se baja."""
    jugados = frame[frame.home_score.notna() & frame.away_score.notna()]
    return {
        fila.game_id: {
            "home_score": float(fila.home_score),
            "away_score": float(fila.away_score),
            "margin": float(fila.home_score - fila.away_score),
            "total": float(fila.home_score + fila.away_score),
            "gameday": str(fila.gameday),
            # Cierre del mercado. Puede faltar; entonces no hay con qué comparar.
            "close_spread": None if pd.isna(fila.spread_line) else float(fila.spread_line),
            "close_total": None if pd.isna(fila.total_line) else float(fila.total_line),
            "close_home_ml": None if pd.isna(fila.home_moneyline) else float(fila.home_moneyline),
            "close_away_ml": None if pd.isna(fila.away_moneyline) else float(fila.away_moneyline),
        }
        for fila in jugados.itertuples()
    }


def probabilidad_implicita(momio):
    """De momio americano a probabilidad, sin quitar el margen todavía."""
    momio = float(momio)
    return -momio / (-momio + 100) if momio < 0 else 100 / (momio + 100)


def mercado_sin_vig(casa, visita):
    """Probabilidad de la casa con el margen repartido entre los dos lados."""
    if casa is None or visita is None:
        return None
    p_casa, p_visita = probabilidad_implicita(casa), probabilidad_implicita(visita)
    suma = p_casa + p_visita
    return p_casa / suma if suma > 0 else None


def _binario(pares, referencia=0.25):
    """Métricas de un mercado binario, en el mismo formato que publica MLB."""
    if not pares:
        return None
    probabilidades = np.array([p for p, _ in pares])
    reales = np.array([y for _, y in pares])
    aciertos = float(np.mean((probabilidades >= 0.5) == (reales == 1)))
    recortadas = np.clip(probabilidades, 1e-9, 1 - 1e-9)
    calibracion = []
    for bajo, alto in TRAMOS:
        dentro = (probabilidades >= bajo) & (probabilidades < alto)
        if dentro.sum():
            calibracion.append({"from": bajo, "to": alto, "n": int(dentro.sum()),
                                "predicted": round(float(probabilidades[dentro].mean()), 4),
                                "observed": round(float(reales[dentro].mean()), 4)})
    return {
        "n": len(pares),
        "base_rate": round(float(reales.mean()), 4),
        "brier": round(float(np.mean((probabilidades - reales) ** 2)), 4),
        "log_loss": round(float(-np.mean(reales * np.log(recortadas) + (1 - reales) * np.log(1 - recortadas))), 4),
        "hit_rate": round(aciertos, 4),
        "baseline_brier": referencia,
        "calibration": calibracion,
    }


def _error(fechados):
    """Error de un pronóstico numérico, con su desglose por mes."""
    if not fechados:
        return None

    def resumen(valores):
        n = len(valores)
        media = sum(valores) / n
        return {"n": n, "mae": round(sum(abs(v) for v in valores) / n, 3),
                "bias": round(media, 3),
                "std_error": round((sum((v - media) ** 2 for v in valores) / n / n) ** 0.5, 4)}

    por_mes = defaultdict(list)
    for fecha, error in fechados:
        por_mes[fecha[:7]].append(error)
    return {**resumen([error for _, error in fechados]),
            "by_month": [{"month": mes, **resumen(errores)} for mes, errores in sorted(por_mes.items())]}


def calificar(juegos, reales):
    """Un cubo de mercados y errores para una versión del modelo."""
    mercados = defaultdict(list)
    margen, total, cierre_margen, cierre_total = [], [], [], []
    dentro = {"margin": [], "total": []}
    fechas, sin_resultado = [], 0
    for identificador, juego in sorted(juegos.items()):
        real = reales.get(identificador)
        if real is None:
            sin_resultado += 1
            continue
        fechas.append(real["gameday"])
        # El empate no lo estima el modelo: su probabilidad es condicional a que
        # no lo haya, así que calificarlo mediría algo que nunca prometió.
        if real["margin"] != 0:
            mercados["moneyline"].append((juego["win_probability"]["home"], 1 if real["margin"] > 0 else 0))
            del_mercado = mercado_sin_vig(real["close_home_ml"], real["close_away_ml"])
            if del_mercado is not None:
                mercados["closing_moneyline"].append((del_mercado, 1 if real["margin"] > 0 else 0))
        # El spread propio se publica con el signo de la casa: -4.1 es la casa
        # favorita por 4.1, o sea un margen esperado de +4.1.
        esperado = -juego["spread"]["home"]
        margen.append((real["gameday"], esperado - real["margin"]))
        total.append((real["gameday"], juego["total_points"] - real["total"]))
        if real["close_spread"] is not None:
            cierre_margen.append((real["gameday"], real["close_spread"] - real["margin"]))
        if real["close_total"] is not None:
            cierre_total.append((real["gameday"], real["close_total"] - real["total"]))
        bajo, alto = juego["intervals_80"]["home_margin"]
        dentro["margin"].append(bajo <= real["margin"] <= alto)
        bajo, alto = juego["intervals_80"]["total"]
        dentro["total"].append(bajo <= real["total"] <= alto)
    return {"mercados": mercados, "margen": margen, "total": total,
            "cierre_margen": cierre_margen, "cierre_total": cierre_total,
            "dentro": dentro, "fechas": fechas, "sin_resultado": sin_resultado}


def cobertura(marcas, nominal=0.8):
    if not marcas:
        return None
    return {"n": len(marcas), "covered": round(float(np.mean(marcas)), 4), "nominal": nominal}


def cargar(root: Path, sha: str) -> pd.DataFrame:
    """El mismo CSV que uso el motor, releido con las columnas de evaluacion."""
    return pd.read_csv(root / f"data/raw/{sha}.csv", usecols=COLUMNAS_EVALUACION, low_memory=False)


def build(root: Path, frame: pd.DataFrame):
    reales = resultados(frame)
    versiones = []
    for version, juegos in emisiones(root).items():
        cubo = calificar(juegos, reales)
        calificadas = len(cubo["fechas"])
        versiones.append({
            "model_version": version,
            # Solo para ordenar: cuando dos versiones llevan los mismos partidos
            # calificados, la pantalla debe abrir con la que corre hoy.
            "last_emitted_at": max(juego["emitted_at"] for juego in juegos.values()),
            # Toda emisión de NFL nació sellada: no hay mezcla que advertir.
            "unlabeled": False,
            "from": min(cubo["fechas"]) if cubo["fechas"] else None,
            "to": max(cubo["fechas"]) if cubo["fechas"] else None,
            "graded": calificadas,
            "pending": cubo["sin_resultado"],
            "significant": calificadas >= MINIMO_SIGNIFICATIVO,
            "markets": {nombre: _binario(pares) for nombre, pares in cubo["mercados"].items() if pares},
            "projected_total": _error(cubo["total"]),
            "projected_margin": _error(cubo["margen"]),
            # La línea de cierre es el mejor pronóstico público que existe. Si el
            # modelo no le gana, no aporta nada que no estuviera ya en el precio.
            "closing_line": {"total": _error(cubo["cierre_total"]), "margin": _error(cubo["cierre_margen"])},
            "interval_coverage": {"margin": cobertura(cubo["dentro"]["margin"]),
                                  "total": cobertura(cubo["dentro"]["total"])},
        })
    # Mas partidos primero, y entre empatadas la de emision mas reciente. Sin el
    # segundo criterio el orden lo decide el nombre del archivo y la pantalla
    # abre con una version que ya se retiro.
    versiones.sort(key=lambda v: (v["graded"], v["last_emitted_at"]), reverse=True)
    for version in versiones:
        del version["last_emitted_at"]
    return {"schema_version": SCHEMA_VERSION, "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "minimum_meaningful": MINIMO_SIGNIFICATIVO, "versions": versiones}
