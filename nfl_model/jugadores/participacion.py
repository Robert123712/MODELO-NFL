"""Rol de cada jugador: snaps, share de targets y share de acarreos proyectados.

Es el equivalente NFL de `minutos.py` del modelo NBA:

1. **Base de cada jugador.** Promedio exponencial de su snap %, target share y carry
   share en los partidos que jugó, en cualquier equipo (un traspaso conserva su rol).
   La vida media se mide en partidos jugados; la temporada anterior pesa menos.
2. **Encogimiento.** Con pocos partidos, el share se encoge hacia lo que su snap %
   sugiere para su posición (un WR al 80% de los snaps suele llevar ~20% de targets).
3. **Reparto.** Entre los disponibles, el share de los que faltan (jugaron en los
   últimos partidos del equipo y hoy no están) se reparte primero dentro de su
   posición y el resto entre todos; al final los shares del equipo suman 1.

Los parámetros se afinan en 2023-24 y se prueban a ciegas en 2025
(`scripts/backtest_participacion.py`).
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

SHARES = ('offense_pct', 'target_share', 'carry_share')


@dataclass(frozen=True)
class Parametros:
    # Afinados en 2023-24 (scripts/backtest_participacion.py --afinar).
    vida_media: float = 3.0        # partidos jugados
    peso_temporada_pasada: float = 0.5
    previo_partidos: float = 0.0   # partidos ficticios al prior por snaps
    misma_posicion: float = 0.5    # fracción del share ausente que se queda en su posición
    ventana_ausentes: int = 3      # partidos del equipo para considerar a alguien "del rol"
    vida_media_equipo: float = 8.0  # volumen del equipo (targets y acarreos por partido)

    def como_dict(self):
        return asdict(self)


def _promedios(juegos: pd.DataFrame, p: Parametros):
    """Estado previo y posterior a cada fila: sumas ponderadas de cada share por jugador."""
    decaimiento = 0.5 ** (1 / p.vida_media)
    orden = juegos.sort_values(['player_id', 'season', 'week', 'game_id']).index.to_numpy()
    jugador = juegos.player_id.to_numpy()[orden]
    temporada = juegos.season.to_numpy()[orden]
    valores = {k: juegos[k].to_numpy(dtype=float)[orden] for k in SHARES}
    n = len(orden)
    previo = {k: np.full(n, np.nan) for k in SHARES}
    post = {k: np.full(n, np.nan) for k in SHARES}
    peso_previo, peso_post = np.zeros((n, len(SHARES))), np.zeros((n, len(SHARES)))
    juegos_temporada = np.zeros(n)
    sumas = np.zeros(len(SHARES))
    pesos = np.zeros(len(SHARES))
    actual, temp_actual, cuenta = None, None, 0
    for i in range(n):
        if jugador[i] != actual:
            actual, temp_actual, cuenta = jugador[i], temporada[i], 0
            sumas[:] = 0
            pesos[:] = 0
        elif temporada[i] != temp_actual:
            sumas *= p.peso_temporada_pasada
            pesos *= p.peso_temporada_pasada
            temp_actual, cuenta = temporada[i], 0
        juegos_temporada[i] = cuenta
        for j, k in enumerate(SHARES):
            previo[k][i] = sumas[j] / pesos[j] if pesos[j] > 0 else np.nan
            peso_previo[i, j] = pesos[j]
            x = valores[k][i]
            sumas[j] *= decaimiento
            pesos[j] *= decaimiento
            if np.isfinite(x):
                sumas[j] += x
                pesos[j] += 1
            post[k][i] = sumas[j] / pesos[j] if pesos[j] > 0 else np.nan
            peso_post[i, j] = pesos[j]
        cuenta += 1
    salida = pd.DataFrame(index=orden)
    for j, k in enumerate(SHARES):
        salida[f'previo_{k}'] = previo[k]
        salida[f'peso_{k}'] = peso_previo[:, j]
        salida[f'post_{k}'] = post[k]
        salida[f'peso_post_{k}'] = peso_post[:, j]
    salida['juegos_temporada'] = juegos_temporada
    return salida.sort_index()


def tasas_por_snap(juegos: pd.DataFrame) -> dict:
    """Share de targets y acarreos por cada 100% de snaps, por posición (el prior)."""
    con_snaps = juegos[juegos.offense_pct.notna()]
    tasas = {}
    for grupo, g in con_snaps.groupby('grupo'):
        snaps = g.offense_pct.sum()
        tasas[grupo] = {'target_share': float(g.target_share.sum() / snaps),
                        'carry_share': float(g.carry_share.sum() / snaps)}
    return tasas


def _encoger(share, peso, snap, tasa, p: Parametros):
    prior = snap * tasa
    share = np.where(np.isfinite(share), share, prior)
    peso = np.where(np.isfinite(share), peso, 0)
    total = peso + p.previo_partidos
    return np.where(total > 0, (share * peso + prior * p.previo_partidos) / np.where(total > 0, total, 1), prior)


def preparar(juegos: pd.DataFrame, p: Parametros, tasas: dict | None = None) -> pd.DataFrame:
    """Agrega a cada fila el estado del jugador ANTES de ese partido, ya encogido."""
    tasas = tasas or tasas_por_snap(juegos)
    estado = _promedios(juegos, p)
    salida = juegos.join(estado)
    prior_snap = salida.groupby('grupo').offense_pct.transform('median')
    salida['base_offense_pct'] = salida.previo_offense_pct.fillna(prior_snap * 0.5)
    for k in ('target_share', 'carry_share'):
        tasa = salida.grupo.map(lambda g: tasas.get(g, {}).get(k, 0.0))
        salida[f'base_{k}'] = _encoger(salida[f'previo_{k}'].to_numpy(), salida[f'peso_{k}'].to_numpy(),
                                       salida.base_offense_pct.to_numpy(), tasa.to_numpy(), p)
        snap_post = salida.post_offense_pct.fillna(salida.base_offense_pct)
        salida[f'post_base_{k}'] = _encoger(salida[f'post_{k}'].to_numpy(), salida[f'peso_post_{k}'].to_numpy(),
                                            snap_post.to_numpy(), tasa.to_numpy(), p)
    salida['post_base_offense_pct'] = salida.post_offense_pct.fillna(salida.base_offense_pct)
    return salida


def ausentes(preparado: pd.DataFrame, p: Parametros) -> pd.DataFrame:
    """Jugadores del rol que no aparecen en un partido de su equipo.

    Del rol = jugó alguno de los `ventana_ausentes` partidos previos del equipo, y en
    su último partido seguía en ese equipo. Lleva su estado posterior a ese partido.
    """
    partidos = (preparado[['game_id', 'team', 'season', 'week']].drop_duplicates()
                .sort_values(['team', 'season', 'week']).reset_index(drop=True))
    partidos['n'] = partidos.groupby('team').cumcount()
    filas = preparado.merge(partidos[['game_id', 'team', 'n']], on=['game_id', 'team'])
    cols = ['player_id', 'grupo', 'post_base_offense_pct', 'post_base_target_share', 'post_base_carry_share']
    registros = []
    por_equipo = {t: g for t, g in filas.groupby('team')}
    for team, g in por_equipo.items():
        por_n = {n: set(x.player_id) for n, x in g.groupby('n')}
        for n in sorted(por_n):
            presentes = por_n[n]
            vistos = {}
            for k in range(n - 1, max(-1, n - 1 - p.ventana_ausentes), -1):
                for pid in por_n.get(k, ()):
                    vistos.setdefault(pid, k)
            for pid, k in vistos.items():
                if pid in presentes:
                    continue
                registros.append((team, n, pid, k))
    if not registros:
        return pd.DataFrame(columns=['game_id', 'team', *cols])
    aus = pd.DataFrame(registros, columns=['team', 'n', 'player_id', 'n_visto'])
    # Si después jugó en otro equipo antes de este partido, ya no es del rol.
    ultimo_equipo = filas[['player_id', 'team', 'season', 'week']].sort_values(['season', 'week'])
    aus = aus.merge(partidos[['team', 'n', 'game_id', 'season', 'week']], on=['team', 'n'])
    estados = filas.set_index(['team', 'n', 'player_id'])[cols[1:]]
    aus = aus.join(estados, on=['team', 'n_visto', 'player_id'])
    otro = ultimo_equipo.merge(aus[['player_id', 'team', 'season', 'week']].rename(
        columns={'team': 't', 'season': 's', 'week': 'w'}), on='player_id')
    otro = otro[(otro.team != otro.t) & ((otro.season < otro.s) | ((otro.season == otro.s) & (otro.week < otro.w)))]
    if len(otro):
        traspasados = set(zip(otro.player_id, otro.t, otro.s, otro.w))
        aus = aus[[(r.player_id, r.team, r.season, r.week) not in traspasados for r in aus.itertuples()]]
    return aus[['game_id', 'team', *cols]]


def repartir(disponibles: pd.DataFrame, faltan: pd.DataFrame, p: Parametros) -> pd.DataFrame:
    """Shares proyectados de los disponibles de cada partido-equipo.

    `disponibles` trae `base_*` por jugador; `faltan` trae `post_base_*` de los del
    rol que no están. De lo que deja cada ausente, `misma_posicion` se queda en su
    posición (si hay a quién) y el resto se reparte entre todos (los snaps se
    quedan completos en su posición), siempre
    proporcional a la base. Devuelve `proy_*`; targets y acarreos suman 1 por
    equipo y el snap % tiene tope de 1.
    """
    salida = disponibles.copy()
    llave = ['game_id', 'team']
    for k in ('target_share', 'carry_share', 'offense_pct'):
        tmp = salida[llave + ['grupo']].copy()
        tmp['base'] = salida[f'base_{k}'].clip(lower=0).to_numpy()
        tmp['suma_grupo'] = tmp.groupby(llave + ['grupo']).base.transform('sum')
        tmp['suma_total'] = tmp.groupby(llave).base.transform('sum')
        m = faltan.groupby(llave + ['grupo'])[f'post_base_{k}'].sum().rename('m').reset_index()
        receptores = tmp.drop_duplicates(llave + ['grupo'])[llave + ['grupo', 'suma_grupo']]
        m = m.merge(receptores, on=llave + ['grupo'], how='left').fillna({'suma_grupo': 0})
        # Los snaps no cruzan de posición: un WR ausente no le da snaps a un RB.
        alfa = 1.0 if k == 'offense_pct' else p.misma_posicion
        m['queda'] = np.where(m.suma_grupo > 0, alfa * m.m, 0.0)
        m['general'] = 0.0 if k == 'offense_pct' else m.m - m.queda
        general = m.groupby(llave).general.sum().rename('general').reset_index()
        tmp = (tmp.merge(m[llave + ['grupo', 'queda']], on=llave + ['grupo'], how='left')
                  .merge(general, on=llave, how='left').fillna({'queda': 0, 'general': 0}))
        valor = (tmp.base
                 + tmp.queda * tmp.base / tmp.suma_grupo.where(tmp.suma_grupo > 0, 1)
                 + tmp.general * tmp.base / tmp.suma_total.where(tmp.suma_total > 0, 1)).to_numpy()
        if k == 'offense_pct':
            salida[f'proy_{k}'] = np.clip(valor, 0, 1)
        else:
            serie = pd.Series(valor, index=salida.index)
            total = serie.groupby([salida.game_id, salida.team]).transform('sum')
            salida[f'proy_{k}'] = (serie / total.where(total > 0, 1)).to_numpy()
    return salida


def volumen_equipo(juegos: pd.DataFrame, p: Parametros) -> pd.DataFrame:
    """Targets y acarreos esperados por equipo antes de cada partido (promedio exponencial)."""
    equipo = (juegos.groupby(['game_id', 'team', 'season', 'week'])
              [['equipo_targets', 'equipo_carries']].first().reset_index()
              .sort_values(['team', 'season', 'week']))
    alfa = 1 - 0.5 ** (1 / p.vida_media_equipo)
    for k in ('equipo_targets', 'equipo_carries'):
        equipo[f'esperado_{k}'] = equipo.groupby('team')[k].transform(
            lambda s: s.shift().ewm(alpha=alfa, adjust=True).mean())
        equipo[f'esperado_{k}'] = equipo[f'esperado_{k}'].fillna(equipo[k].mean())
    return equipo[['game_id', 'team', 'esperado_equipo_targets', 'esperado_equipo_carries']]
