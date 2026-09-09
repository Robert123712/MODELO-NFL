# MODELO NFL · Edgebook

Primera versión ejecutable de predicción NFL, independiente de momios. Calcula
ganador con probabilidad, spread propio, total y marcador esperado. Usa Python,
pandas y scikit-learn; los datos gratuitos provienen de nflverse.

## Estado

Versión **0.1.0 experimental**, ejecutada con datos reales el 9 de septiembre UTC
(8 de septiembre en Chihuahua) de 2026. Generó 16 proyecciones próximas.
Tiene una pantalla básica independiente, con las proyecciones de la última
emisión publicada. No está conectada todavía al Edgebook de producción. El
repositorio contiene el motor, la pantalla, un exportador JSON, historial de
emisiones y evaluación.

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

1,693 partidos, incluidos playoffs. Cinco empates excluidos solamente de las
métricas binarias. Cada temporada se evalúa con entrenamiento y calibración
anteriores; nunca se mezclan aleatoriamente partidos futuros con pasados.
Los hiperparámetros están fijados y este backtest no los optimiza.

| Métrica | Modelo | Referencia |
| --- | ---: | ---: |
| Acierto del ganador (1,688 juegos sin empate) | 65.70% | Elo 62.03%; siempre local 53.91% |
| Brier (menor es mejor) | 0.2215 | Elo 0.2282 |
| Error absoluto medio del margen | 10.10 puntos | — |
| Error absoluto medio del total | 10.66 puntos | Media histórica 10.92 |
| Cobertura intervalo de margen al 80% | 79.56% | Objetivo nominal 80% |
| Cobertura intervalo de total al 80% | 80.33% | Objetivo nominal 80% |

Los resultados por temporada, grupos de calibración y cada predicción están en
`reports/`. La mejora del total frente a su referencia es pequeña. Las
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

Esta versión usa estadísticas de **marcadores de equipo**, no eficiencia EPA por
jugada. No incorpora todavía rendimiento individual del quarterback, lesiones,
alineaciones, turnovers por jugada ni pronóstico meteorológico. Lo declara en
cada proyección. Es una base medida para añadir esas variables y comprobar su
aporte, no una réplica completa de todas las variables sugeridas en Instagram.

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
- [Disponibilidad de datos: la fuente de lesiones dejó de funcionar tras 2024](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)
- [Calibración en scikit-learn](https://scikit-learn.org/stable/modules/calibration.html)

Los datos pertenecen a sus respectivos proveedores y están sujetos a sus
condiciones; este repositorio no los relicencia.

## Integración con Edgebook

El JSON usa `schema_version: 1.1` y `sport: NFL`, conserva los campos básicos del
exportador MLB y agrega `spread`, `total_points`, intervalos y explicaciones.
**El contrato actual de Edgebook solo acepta MLB/1.0**, de modo que este archivo
todavía no se puede enchufar directamente a producción. Ver `docs/EDGEBOOK.md`
para el contrato y los cambios concretos del consumidor.
