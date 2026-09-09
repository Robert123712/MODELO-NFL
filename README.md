# MODELO NFL · Edgebook

Primera versión ejecutable de predicción NFL, independiente de momios. Calcula
ganador con probabilidad, spread propio, total y marcador esperado. Usa Python,
pandas y scikit-learn; los datos gratuitos provienen de nflverse.

## Estado

Versión **0.3.0 experimental**, ejecutada con datos reales el 9 de septiembre UTC
(8 de septiembre en Chihuahua) de 2026. Generó 16 proyecciones próximas.
Tiene una pantalla básica independiente, con las proyecciones de la última
emisión publicada. Claude conectó el snapshot NFL con Edgebook de producción. El
repositorio contiene el motor, la pantalla, un exportador JSON, historial de
emisiones y evaluación.

La v0.3 añade ajuste de EPA por rival al total y captura actual de lesiones,
QB probable, plantillas, cambios de entrenador y clima. El ajuste por rival baja
el error histórico del total de 10.5411 a 10.5233 puntos, una mejora pequeña. Ver
`reports/opponent-adjustment.json`.

Desde esta versión, el **QB probable de hoy sí entra al total**: su historial
sustituye al del QB de referencia cuando difieren. No es un peso nuevo, es
corregir una entrada equivocada; la regresión conserva sus coeficientes y el
ganador, la probabilidad y el margen esperado no cambian. Lesiones, plantilla y
clima siguen archivándose **sin pesos numéricos validados**.

Medir ese cambio con noticias pasadas no es posible, así que se midió su techo:
sustituir la referencia por el titular real de cada partido de 2020–2025 movió el
error del total 0.0014 puntos, con intervalo del 95% [-0.0288, 0.0305]. **No se
distingue del azar**: la sustitución se hace porque la entrada anterior era falsa,
no porque mejore la precisión demostrada. Detalle en `reports/qb-news.json`.

Sobre la emisión publicada del 9 de septiembre el ajuste habría movido 12 QB en 9
de 16 partidos, con cambio medio de 0.6 puntos de total y máximo de 3.8 en
Denver–Kansas City, donde el modelo cargaba a Chris Oladokun y Jarrett Stidham en
lugar de Patrick Mahomes y Bo Nix. Ese repaso está en `reports/qb-news-preview.json`
y **no reemplaza la emisión**: se aplica al generar la siguiente.

El contrato de producción se documenta en [docs/CONTRACT.md](docs/CONTRACT.md).
El formato sigue siendo **1.1**. La versión del motor incluye una huella para no
mezclar matemática distinta bajo un mismo sello. Las emisiones v0.1/v0.2 se
conservan sin modificaciones.

## Pantalla visual

[Abrir pantalla NFL (privada)](https://edgebook-nfl-robert.rsapicks.chatgpt.site)

Presenta partidos por fecha, marcador esperado, ganador con probabilidad, spread
propio, total y contexto desplegable. Los horarios están en hora de Chihuahua.
Es una vista de la emisión publicada, no un marcador en vivo. Para renovar los
datos de la pantalla hay que generar otra emisión, reconstruirla y publicarla.

Para verla localmente:

```sh
python scripts/build_web.py
python -m http.server 8765 --bind 127.0.0.1 --directory dist
```

Después abre `http://127.0.0.1:8765/`. La pantalla usa HTML, CSS y JavaScript,
sin dependencias adicionales. El paquete web contiene solo la pantalla y el JSON
de predicciones, no código Python, archivos de entrenamiento ni credenciales.

## Ejecutar

Requiere Python 3.11 o superior.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
modelo-nfl backtest
modelo-nfl run
# Corte reproducible, sin usar resultados posteriores al corte:
modelo-nfl run --as-of 2026-09-09T00:00:00Z --days 8
```

`run` descarga datos, reconstruye las estadísticas previas a cada semana, entrena,
calibra y exporta `data/edgebook-latest.json`. También conserva cada emisión en
`data/history/` y el modelo en `artifacts/model.joblib`. No cargar archivos joblib
de orígenes no confiables. GitHub Actions permite ejecutar manualmente y descargar
el resultado como artefacto; no hay actualización periódica configurada.

## Qué aprende

Dos regresiones Ridge con estandarización estiman margen local y total. Las
entradas son puntos anotados/permitidos previos, forma reciente, rating Elo que
considera rivales, localía, descanso y avance de temporada. Los promedios usan
hasta 16 juegos, suavizado hacia un prior y menos peso para la temporada anterior.
La v0.2 añade para el total EPA ofensivo/defensivo por jugada, frecuencia de
pérdidas de balón, volumen de jugadas, EPA por dropback y CPOE del quarterback
histórico de referencia. El QB de referencia es el que tuvo más intentos en el
último juego observado del equipo; **no confirma el titular del siguiente juego**.
Al servir, si el contexto reporta otro QB probable, el total usa el historial de
ese quarterback: es la misma fórmula con la entrada corregida. La sustitución se
declara por partido en `qb_adjustment` y requiere identificar al jugador en
nflverse; si no se resuelve, la emisión sale sin tocar y con aviso. No se usan QB
retrospectivos del partido que se intenta predecir ni estadísticas de esa semana
antes de terminar de emitir sus features.

Las contribuciones por variable explican el margen y el total respecto al promedio
de entrenamiento; no son explicaciones causales.

Una regresión logística transforma el margen estimado en probabilidad, calibrada
en una temporada posterior al entrenamiento y anterior a la predicción. Para
2026: regresiones con 2005–2024, calibración con 2025. Las estadísticas de entrada
sí se actualizan semanalmente con nuevos resultados. Los intervalos empíricos del
80% se estiman con los errores de esa temporada de calibración.

El ganador probabilístico y el favorito por margen pueden diferir cerca de cero:
se estiman probabilidad de victoria y margen medio, magnitudes distintas.

## Evaluación temporal 2020–2025

Se compararon tres conjuntos de variables (base, EPA, EPA+QB) solamente en
2017–2019 para elegir cada regresión. Ganó **base para margen** y **EPA+QB para
total**. Ganador y spread conservan sus variables originales. Después se evaluó
la combinación elegida en 2020–2025 sin modificar candidatos. Este historial de
la versión base ya se había examinado: es confirmación retrospectiva, no una
prueba prospectiva ciega. Detalle completo: `reports/refinement.json`.

1,693 partidos, incluidos playoffs. Cinco empates excluidos solamente de las
métricas binarias. Cada temporada se evalúa con entrenamiento y calibración
anteriores; nunca se mezclan aleatoriamente partidos futuros con pasados.
Los hiperparámetros están fijados y este backtest no los optimiza.

| Métrica | Modelo | Referencia |
| --- | ---: | ---: |
| Acierto del ganador (1,688 juegos sin empate) | 65.70% | Elo 62.03%; siempre local 53.91% |
| Brier (menor es mejor) | 0.2215 | Elo 0.2282 |
| Error absoluto medio del margen | 10.10 puntos | — |
| Error absoluto medio del total | 10.54 puntos | v0.1: 10.66; media histórica 10.92 |
| Cobertura intervalo de margen al 80% | 79.56% | Objetivo nominal 80% |
| Cobertura intervalo de total al 80% | 80.92% | Objetivo nominal 80% |

Los resultados originales v0.1 se conservan en `reports/backtest.json`; los nuevos
en `reports/refinement.json` y `reports/refined-predictions.csv`. La mejora del
total es pequeña (0.12 puntos de error medio, alrededor de 1.2%). Las
probabilidades extremas tienen muestras pequeñas y algunas bandas muestran mala
calibración: no se presenta el porcentaje como certeza ni como rentabilidad.

## Convenciones

- **Spread local −3.5**: el modelo espera que el local gane por 3.5 puntos.
- **Total 44.5**: suma de puntos esperada. No significa apostar over o under.
- La probabilidad se condiciona a que el partido no termine empatado. La v0.1 no
  estima probabilidad de empate; los empates sí entrenan margen y total.
- Los puntos incluyen tiempo extra, como los resultados finales de la fuente.
- Marcadores con decimales representan valores esperados, no resultados exactos.

## Datos y límites

Se usa una lista explícita de columnas: ni momios ni spread ni total de mercado
entran como variables o etiquetas. Las etiquetas se calculan de los marcadores.
Se guardan URL, hora de descarga y SHA-256 junto a las predicciones; el CSV crudo
se conserva localmente en `data/raw/`, fuera de Git. Los IDs ESPN y nflverse
facilitan el futuro cruce con Edgebook.

La v0.2 combina marcadores con estadísticas semanales de equipo y QB calculadas
por nflverse a partir de jugadas. EPA se aproxima por jugada oficial (intentos +
sacks + carreras), sin filtrar kneel-downs ni garbage time. Desde v0.3 se ajusta
EPA por el rendimiento previo del rival; el rating Elo sigue aportando fuerza relativa.
El único dato del día que altera un número es el QB probable, y solo el total; no
incorpora lesiones, alineaciones completas, titulares confirmados por el equipo,
traspasos de offseason ni pronóstico meteorológico. El QB probable se cruza con
nflverse por identidad ESPN y, si esa vía falla, por nombre único entre
quarterbacks de las últimas tres temporadas. Un QB sin historial no se sustituye.
El historial individual del QB se conserva por ID, pero su referencia de equipo
requiere cautela cuando cambia la plantilla. Las ausencias se declaran en cada
proyección.

`scripts/evaluate_qb_news.py` mide el techo del ajuste por QB y escribe
`reports/qb-news.json`; `scripts/preview_qb_news.py` muestra qué haría ese ajuste
sobre la emisión ya publicada, sin emitir ni archivar historial.
`scripts/compare_models.py` reproduce selección, comparación e incertidumbre
pareada por semana. La selección se guarda en `config/model.json` y es la que
consume `modelo-nfl run`. `--baseline` permite ejecutar las variables v0.1. La
primera descarga avanzada incluye 2004–2026; temporadas anteriores se cachean con
SHA-256 verificado y la temporada actual se refresca. Las estadísticas históricas
pueden estar corregidas retrospectivamente; se guardan las fuentes utilizadas.

Los resultados del mismo día se excluyen para evitar marcadores parciales. Todos
los partidos de una semana se calculan antes de incorporar resultados de esa
semana. El backtest usa resultados históricos corregidos de la fuente actual;
no reconstruye versiones originales del calendario o correcciones publicadas
después. `--as-of` limita resultados, pero no constituye un archivo histórico de
toda la información disponible en ese instante.

Fuentes:

- [nflverse](https://nflverse.nflverse.com/)
- [Calendario y resultados](https://github.com/nflverse/nfldata/blob/master/data/games.csv)
- [Diccionario del calendario](https://nflreadr.nflverse.com/articles/dictionary_schedules.html)
- [Diccionario de estadísticas de equipo](https://nflreadr.nflverse.com/articles/dictionary_team_stats.html)
- [Estadísticas individuales](https://nflreadr.nflverse.com/reference/load_player_stats)
- [Disponibilidad de datos: la fuente de lesiones dejó de funcionar tras 2024](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)
- [Calibración en scikit-learn](https://scikit-learn.org/stable/modules/calibration.html)

Los datos pertenecen a sus respectivos proveedores y están sujetos a sus
condiciones; este repositorio no los relicencia.

## Integración con Edgebook

El JSON usa `schema_version: 1.1` y `sport: NFL`, conserva los campos básicos del
exportador MLB y agrega `spread`, `total_points`, intervalos y explicaciones.
Edgebook acepta MLB/1.0 y NFL/1.1. Los cambios futuros deben respetar
`docs/CONTRACT.md`; no subir la versión del formato sin coordinar al consumidor.
