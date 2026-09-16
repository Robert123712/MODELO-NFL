"""El record califica lo que se emitio antes del juego, no lo que se sabe despues."""
import json

import pandas as pd
import pytest

from nfl_model import record


def emision(root, sello, generado, spread_home, total, p_casa, identificador="2026_01_AA_BB"):
    juego = {"league_game_id": identificador, "season": 2026, "week": 1,
             "home": {"code": "BB"}, "away": {"code": "AA"},
             "win_probability": {"home": p_casa, "away": 1 - p_casa},
             "spread": {"home": spread_home}, "total_points": total,
             "intervals_80": {"home_margin": [-10, 10], "total": [30, 60]}}
    (root / "data/history").mkdir(parents=True, exist_ok=True)
    (root / f"data/history/{generado}.json").write_text(json.dumps(
        {"sport": "NFL", "model_version": sello, "generated_at": generado, "games": [juego]}))


def resultados(marcador_casa=27, marcador_visita=20, cierre_spread=3.0, cierre_total=44.5,
               momio_casa=-150, momio_visita=130):
    return pd.DataFrame([{
        "game_id": "2026_01_AA_BB", "season": 2026, "week": 1, "gameday": "2026-09-10",
        "home_score": marcador_casa, "away_score": marcador_visita,
        "spread_line": cierre_spread, "total_line": cierre_total,
        "home_moneyline": momio_casa, "away_moneyline": momio_visita,
    }])


def test_the_own_spread_is_read_with_the_sign_the_emission_publishes(tmp_path):
    # -6.5 es la casa favorita por 6.5, o sea un margen esperado de +6.5.
    # La casa gano por 7, asi que el modelo se quedo corto por medio punto.
    emision(tmp_path, "v1", "20260909T000000Z", -6.5, 44.0, 0.6)
    datos = record.build(tmp_path, resultados())
    assert datos["versions"][0]["projected_margin"]["bias"] == pytest.approx(-0.5)


def test_the_closing_line_is_read_from_the_home_side_too(tmp_path):
    emision(tmp_path, "v1", "20260909T000000Z", -6.5, 44.0, 0.6)
    # Cierre 3.0 contra un margen real de 7: el mercado tambien se quedo corto.
    datos = record.build(tmp_path, resultados())
    assert datos["versions"][0]["closing_line"]["margin"]["bias"] == pytest.approx(-4.0)


def test_the_first_emission_of_a_version_is_the_one_graded(tmp_path):
    emision(tmp_path, "v1", "20260909T010000Z", -6.5, 44.0, 0.6)
    emision(tmp_path, "v1", "20260909T020000Z", -20.0, 99.0, 0.99)
    datos = record.build(tmp_path, resultados())
    version = datos["versions"][0]
    assert version["graded"] == 1
    assert version["projected_total"]["bias"] == pytest.approx(-3.0)


def test_each_version_is_graded_on_its_own(tmp_path):
    emision(tmp_path, "v1", "20260909T010000Z", -6.5, 44.0, 0.6)
    emision(tmp_path, "v2", "20260909T020000Z", -1.0, 50.0, 0.51)
    datos = record.build(tmp_path, resultados())
    assert {v["model_version"] for v in datos["versions"]} == {"v1", "v2"}
    assert all(v["graded"] == 1 for v in datos["versions"])


def test_a_tie_is_not_graded_as_a_winner(tmp_path):
    emision(tmp_path, "v1", "20260909T000000Z", -6.5, 44.0, 0.6)
    datos = record.build(tmp_path, resultados(marcador_casa=20, marcador_visita=20))
    version = datos["versions"][0]
    # El empate entra al error del marcador pero no al mercado de ganador: la
    # probabilidad se publica condicionada a que no lo haya.
    assert version["graded"] == 1
    assert "moneyline" not in version["markets"]
    assert version["projected_margin"]["n"] == 1


def test_the_market_without_price_is_left_out_instead_of_guessed(tmp_path):
    emision(tmp_path, "v1", "20260909T000000Z", -6.5, 44.0, 0.6)
    datos = record.build(tmp_path, resultados(cierre_spread=None, momio_casa=None))
    version = datos["versions"][0]
    assert version["closing_line"]["margin"] is None
    assert "closing_moneyline" not in version["markets"]
    assert version["projected_margin"]["n"] == 1


def test_the_devigged_market_probability_adds_up_to_one():
    casa = record.mercado_sin_vig(-150, 130)
    visita = record.mercado_sin_vig(130, -150)
    assert casa + visita == pytest.approx(1.0)
    assert casa > 0.5


def test_a_game_without_result_is_pending_not_wrong(tmp_path):
    emision(tmp_path, "v1", "20260909T000000Z", -6.5, 44.0, 0.6)
    vacio = resultados().assign(home_score=None, away_score=None)
    version = record.build(tmp_path, vacio)["versions"][0]
    assert version["graded"] == 0 and version["pending"] == 1


def test_the_newest_version_leads_when_two_are_tied(tmp_path):
    # La pantalla abre con versions[0]. Con las dos parejas en partidos
    # calificados, el desempate lo tiene que ganar la que corre hoy: si lo
    # decidiera el nombre del archivo, se mostraria la version ya retirada.
    emision(tmp_path, "v1", "20260909T010000Z", -6.5, 44.0, 0.6)
    emision(tmp_path, "v2", "20260909T020000Z", -3.0, 41.0, 0.55)
    datos = record.build(tmp_path, resultados())
    assert [v["model_version"] for v in datos["versions"]] == ["v2", "v1"]
    # El sello de emision es criterio de orden, no algo que la pantalla lea.
    assert "last_emitted_at" not in datos["versions"][0]


def test_more_graded_games_still_beat_a_newer_version(tmp_path):
    emision(tmp_path, "v1", "20260909T010000Z", -6.5, 44.0, 0.6)
    emision(tmp_path, "v1", "20260909T011000Z", -6.5, 44.0, 0.6, identificador="2026_01_CC_DD")
    emision(tmp_path, "v2", "20260909T020000Z", -3.0, 41.0, 0.55)
    otro = resultados().assign(game_id="2026_01_CC_DD")
    datos = record.build(tmp_path, pd.concat([resultados(), otro], ignore_index=True))
    assert [v["model_version"] for v in datos["versions"]] == ["v1", "v2"]
    assert datos["versions"][0]["graded"] == 2
