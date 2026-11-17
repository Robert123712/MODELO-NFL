# Simulador NFL con estadísticas por jugador

Primera pieza del simulador NFL por jugador, armado como el modelo NBA. Proyecta,
para cada partido, el **snap %**, el **target share** y el **carry share** de cada
QB, RB, WR y TE disponible. Es el equivalente de `minutos.py` en NBA: las fases
siguientes (simulador Monte Carlo de 10,000 partidos y props) reparten jugadas
con estos shares.

El simulador **no toca la emisión**: `data/edgebook-latest.json`, la huella
matemática y el contrato 1.1 no cambian. El marcador esperado sigue saliendo
del Ridge; el simulador le da forma (TD, FG, empates) y le agrega jugadores.

```sh
python scripts/props_partidos.py                 # partidos por jugar de la emisión publicada
python scripts/props_partidos.py --solo 2026_10_WAS_NYG
python scripts/backtest_participacion.py         # fase 1
python scripts/backtest_props.py                 # fases 2 y 3 -> reports/props.json
```

Salidas: `data/latest.json` (props por jugador y partido) y
`data/video/<espn_game_id>.json` (datos del video corto).

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

## Fase 2: simulador del marcador (`simulador.py`)

10,000 partidos por juego. Cada equipo tiene las mismas N series (10, 11 o 12) y
cada serie termina en TD, gol de campo o nada; las probabilidades reproducen el
marcador del Ridge con 78% de los puntos por TD (2021-2025). Extra punto 95%;
5.2% de los TD son de defensa o equipos especiales y no se le dan a nadie.
Empate en tiempo regular: tiempo extra (FG 55%, TD 45%, 7% queda empatado).

En 2020-2025 (1,693 partidos, predicciones fuera de muestra del Ridge):

| | Simulador | Real o Ridge |
| --- | ---: | ---: |
| Brier del ganador | 0.2220 | 0.2215 (logística del Ridge) |
| Cobertura del rango 80% del margen | 81.9% | 80% nominal |
| Margen exacto de 7 | 7.3% | 7.7% |
| Margen exacto de 3 | 9.1% | 15.2% |
| Partidos por 8 o menos | 43.7% | 53.4% |
| Empates | 0.3% | 0.3% |

Lo flojo: los partidos cerrados. En la realidad el que va abajo busca el empate
y el que va arriba administra; el simulador no tiene reloj, así que da menos
partidos por 3 y por una posesión. Es el siguiente arreglo (un "cierre" como el
del NBA).

## Fase 3: props (`props.py`, `tasas.py`)

Los jugadores salen de cada simulación del marcador:

1. Targets y acarreos del equipo según el guion: `targets = 12.5 + 0.60·esperado
   − 0.135·margen` y `acarreos = 14.9 + 0.45·esperado + 0.28·margen` (2021-24).
2. Reparto multinomial con los shares de la fase 1; cada simulación mueve el share
   ±35% y la eficiencia ±15% (lognormal, con topes) porque el rol de hoy no se
   conoce exacto.
3. Recepciones binomiales; yardas por recepción gamma y por acarreo normales,
   con tasas encogidas hacia su posición. La eficiencia del equipo sube con los
   puntos de la simulación (elasticidad 0.32).
4. TD: los ofensivos de la simulación se dividen en pase y carrera (fracción del
   equipo) y se reparten por la parte de TD de cada jugador.
5. QB: yardas de pase = yardas recibidas de su equipo.
6. Mezcla final con el promedio de temporada y los últimos 4 (mínimos cuadrados
   en 2023-24): la simulación pesa 69-78%.

Prueba ciega 2025, error medio absoluto por jugador-partido:

| | Modelo | Promedio de temporada | Últimos 4 |
| --- | ---: | ---: | ---: |
| Yardas de pase (QB, 457) | **61.3** | 65.9 | 69.0 |
| Pases de TD (QB, 474) | **0.93** | 0.99 | 1.02 |
| Yardas por tierra (RB/QB, 1,869) | **15.7** | 16.1 | 16.6 |
| Recepciones (4,923) | **1.12** | 1.15 | 1.16 |
| Yardas por recepción (4,923) | **14.6** | 14.9 | 15.3 |
| TD anotados (4,923) | 0.290 | 0.286 | 0.286 |

En TD el error medio absoluto no sirve (casi todos son 0 o 1); el Brier de
"anota TD" es 0.123 y su calibración por bandas está en `reports/props.json`.
La P(over) está bien calibrada entre 30% y 60%; arriba de 65% el modelo se pasa
de confiado en recepciones (dice 76%, pasa 66%).

No hay líneas históricas de props: esto dice que le gana a los promedios, no a
las casas.

## Fase 4: operación (`emitir.py`, `scripts/props_partidos.py`)

Disponibles = jugadores del equipo en sus últimos 3 partidos, menos Out,
Doubtful, IR o suspendidos en el reporte de ESPN. Questionable juega y se marca
`en_duda`. Solo entra el QB probable del depth chart de ESPN. Se corre después
de la emisión y otra vez con los inactivos (~90 min antes).

El video (`video.py`) usa las mismas simulaciones: victorias, histograma del
margen, cinta, y por jugador sus probabilidades y cuánto gana su equipo cuando
pasa cada umbral. "Mejor partido" es el percentil 99 de su estadística principal
(1 de cada 100): el máximo de 10,000 apila todas las colas y da récords imposibles.

## Pendiente

- Probabilidad de jugar por estatus del reporte (Questionable, Doubtful) calibrada
  con `injuries` de nflverse, para servir antes de conocer los inactivos.
- Share de zona roja (play-by-play) para repartir TD.
- Cierre de partidos (margen de 3 y una posesión).
- Ajuste por rival en yardas por pase (probarlo a ciegas antes de usarlo).
- Mostrar los props en Edgebook y correr esto en el workflow.
