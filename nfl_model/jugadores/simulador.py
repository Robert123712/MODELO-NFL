"""Monte Carlo del marcador por series ofensivas: TD y goles de campo.

El marcador esperado de cada equipo lo da el Ridge que ya se publica (no se toca
su matemática). Aquí solo se le da forma: cada equipo tiene N series (las mismas
para los dos, 10-12) y cada serie termina en TD, gol de campo o nada. Las
probabilidades por serie se fijan para reproducir los puntos esperados, con la
parte de TD y de FG que se ve en 2021-2025 (78% de los puntos llegan por TD).

Contar TD y FG en vez de sacar una normal da solo los números clave (3 y 7) y la
dispersión correcta: con N=11 y un binomial, la desviación por equipo sale ~9.3
puntos, la misma que tienen los residuos del Ridge en 2020-2025.

Empate al final del tiempo regular: tiempo extra. Gana quien anota primero (FG
55%, TD 45% de las veces) y 7% de los tiempos extra terminan empatados, como en
temporada regular. En playoffs no hay empate.
"""
from dataclasses import dataclass, asdict

import numpy as np

PUNTOS_TD = 6.0
PAT = 0.95                # extra punto convertido (2021-2025)
PARTE_TD = 0.78           # de los puntos, los que llegan por TD
NO_OFENSIVO = 0.052       # TD de defensa o equipos especiales
SERIES = (10, 11, 12)
P_SERIES = (0.3, 0.4, 0.3)
OT_EMPATE = 0.07
OT_FG = 0.55


@dataclass
class Simulacion:
    """Una fila por simulación. `*_td_of` son TD de la ofensiva (pase o carrera)."""
    casa: np.ndarray
    visita: np.ndarray
    casa_td_of: np.ndarray
    visita_td_of: np.ndarray
    casa_fg: np.ndarray
    visita_fg: np.ndarray
    tiempo_extra: np.ndarray
    series: np.ndarray

    @property
    def margen(self):
        return self.casa - self.visita

    def resumen(self):
        m = self.margen
        return {'simulaciones': int(len(m)), 'p_casa': float((m > 0).mean()),
                'p_visita': float((m < 0).mean()), 'p_empate': float((m == 0).mean()),
                'casa': float(self.casa.mean()), 'visita': float(self.visita.mean()),
                'tiempo_extra': float(self.tiempo_extra.mean())}


def _probabilidades(esperado, series):
    por_serie = np.maximum(esperado, 1.0) / series
    p_td = PARTE_TD * por_serie / (PUNTOS_TD + PAT)
    p_fg = (1 - PARTE_TD) * por_serie / 3.0
    total = p_td + p_fg
    escala = np.where(total > 0.95, 0.95 / total, 1.0)
    return p_td * escala, p_fg * escala


def _equipo(rng, esperado, series):
    p_td, p_fg = _probabilidades(esperado, series)
    resultado = rng.random((len(series), SERIES[-1]))
    activa = np.arange(SERIES[-1])[None, :] < series[:, None]
    td = ((resultado < p_td[:, None]) & activa).sum(1)
    fg = ((resultado >= p_td[:, None]) & (resultado < (p_td + p_fg)[:, None]) & activa).sum(1)
    pat = rng.binomial(td, PAT)
    no_of = rng.binomial(td, NO_OFENSIVO)
    return PUNTOS_TD * td + pat + 3 * fg, td - no_of, fg


def simular(casa: float, visita: float, n: int = 10_000, semilla: int = 0, playoffs: bool = False) -> Simulacion:
    rng = np.random.default_rng(semilla)
    series = rng.choice(SERIES, size=n, p=P_SERIES)
    pc, tdc, fgc = _equipo(rng, np.full(n, casa), series)
    pv, tdv, fgv = _equipo(rng, np.full(n, visita), series)
    pc, pv = pc.astype(float), pv.astype(float)
    empate = pc == pv
    # Tiempo extra: el más fuerte anota primero un poco más seguido.
    p_casa_ot = np.clip(0.5 + (casa - visita) / 50, 0.2, 0.8)
    sigue = empate & (rng.random(n) < (0 if playoffs else OT_EMPATE))
    decide = empate & ~sigue
    gana_casa = rng.random(n) < p_casa_ot
    es_fg = rng.random(n) < OT_FG
    puntos = np.where(es_fg, 3.0, 6.0)
    pc = pc + np.where(decide & gana_casa, puntos, 0)
    pv = pv + np.where(decide & ~gana_casa, puntos, 0)
    tdc = tdc + (decide & gana_casa & ~es_fg)
    tdv = tdv + (decide & ~gana_casa & ~es_fg)
    fgc = fgc + (decide & gana_casa & es_fg)
    fgv = fgv + (decide & ~gana_casa & es_fg)
    return Simulacion(pc, pv, tdc, tdv, fgc, fgv, empate, series)


def parametros():
    return {'puntos_td': PUNTOS_TD, 'pat': PAT, 'parte_td': PARTE_TD, 'no_ofensivo': NO_OFENSIVO,
            'series': list(SERIES), 'p_series': list(P_SERIES), 'ot_empate': OT_EMPATE, 'ot_fg': OT_FG}


__all__ = ['Simulacion', 'simular', 'parametros', 'asdict']
