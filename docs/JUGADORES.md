# Rol por jugador (fase 1 del simulador con estadísticas)

Primera pieza del simulador NFL por jugador, armado como el modelo NBA. Proyecta,
para cada partido, el **snap %**, el **target share** y el **carry share** de cada
QB, RB, WR y TE disponible. Es el equivalente de `minutos.py` en NBA: las fases
siguientes (simulador Monte Carlo de 10,000 partidos y props) reparten jugadas
con estos shares.

Todavía **no entra a la emisión**: `data/edgebook-latest.json`, la huella
matemática y el contrato 1.1 no cambian.

Código en `nfl_model/jugadores/`; backtest en `scripts/backtest_participacion.py`,
resultado en `reports/participacion.json`.

## Datos

nflverse, por temporada: `stats_player_week` (targets, acarreos, yardas, TD) y
`snap_counts` (snaps ofensivos). Los snaps traen ID de Pro Football Reference y
las estadísticas ID gsis; el puente es `players.csv`. En 2021–2026 cruza 99.8% de
las filas con snaps; lo que no cruza (sobre todo linieros) se descarta. Se guarda
con hash en `data/raw/jugadores/`.

## Cómo se calcula

1. **Base del jugador.** Promedio exponencial de cada share en los partidos que
   jugó, en cualquier equipo. Vida media de **3 partidos jugados**; la temporada
   anterior entra con **peso 0.5**.
2. **Prior por snaps.** Un jugador sin historial toma el share que su posición
   produce por cada snap (medido en 2021–2022), por un snap % inicial de la mitad
   de la mediana de su posición. Con historial no hay encogimiento: la búsqueda
   eligió 0 partidos ficticios.
3. **Ausentes.** Del rol = jugó alguno de los últimos 3 partidos del equipo y no
   se fue a otro equipo después. Lo que deja cada ausente se reparte **50% dentro de
   su posición y 50% entre todos**, proporcional a la base. Los snaps se quedan
   completos en su posición y tienen tope de 100%.
4. **Normalización.** Targets y acarreos del equipo suman 1 entre los disponibles.
5. **Volumen.** Para medir en conteos, targets y acarreos esperados del equipo con
   un promedio exponencial de vida media 8 partidos. Es provisional: la fase 2 lo
   reemplaza con volumen y guion de partido dentro de la simulación.

## Backtest

Walk-forward: cada partido usa solo lo jugado antes. La disponibilidad sale de
quién tuvo snaps, que equivale a conocer los inactivos ~90 minutos antes. Se
califican RB, WR y TE (acarreos: RB y QB) con 3+ partidos previos en la temporada.
Se afinó en 2023–2024 con una rejilla de 180 combinaciones; **2025 es prueba ciega**.

Error medio absoluto por jugador-partido, 2025:

| | Modelo | Promedio de temporada | Últimos 4 |
| --- | ---: | ---: | ---: |
| Targets (4,923) | **1.454** | 1.505 | 1.504 |
| Acarreos (1,869) | **2.602** | 2.733 | 2.744 |
| Target share | **0.0447** | 0.0461 | 0.0464 |
| Carry share | **0.0797** | 0.0866 | 0.0847 |
| Snap % | 0.1247 | 0.1278 | **0.1234** |

Targets bajan 3.4% contra el promedio de temporada y acarreos 4.8%. El snap %
queda prácticamente empatado con los últimos 4 (0.0013 peor): hoy solo se usa
como prior de jugadores nuevos y para redistribuir snaps, no para contar targets.

## Pendiente para fases siguientes

- Probabilidad de jugar por estatus del reporte (Questionable, Doubtful) calibrada
  con `injuries` de nflverse, para servir antes de conocer los inactivos.
- Share de zona roja (play-by-play) para repartir TD.
- Volumen y tasa de pase dentro de la simulación, según el guion del partido.
