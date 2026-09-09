# Contrato activo con Edgebook

Edgebook ya consume `data/edgebook-latest.json` en producción. Estas restricciones
se conservan al modificar el motor:

1. Cambiar `model_version` cuando cambia la matemática. Desde v0.3 se agrega una
   huella de código matemático, variables, coeficientes, escalas y calibración:
   `0.3.0-<12 caracteres>`. La huella cubre `model.py`, `features.py`,
   `advanced.py` y `qb_news.py`, así que tocar el ajuste por QB cambia el sello
   aunque una emisión concreta no lo aplique. Entrenamientos o configuraciones distintos no comparten
   sello aunque se ejecute `--baseline`. La huella completa va en `math_fingerprint`.
2. **Mantener `schema_version: "1.1"`.** Un formato futuro requiere coordinación
   con el consumidor antes de emitirlo. MLB conserva su contrato 1.0.
3. No quitar ni renombrar `league_game_id`, `espn_game_id`, `starts_at`,
   `home/away {name,code}`, `projection {home_score,away_score}`, `win_probability`,
   `pick {market,side,label}`, `data_warning`, `warnings`, `probability_basis`,
   `spread`, `total_points`, `explanations.margin.top_factors
   [{label,contribution_points}]`, `skipped_without_id`.
4. Avisos generales del motor van en `warnings` y pueden repetirse en
   `data_warning`. Un aviso específico de partido va en `data_warning` y **no**
   en `warnings`: Edgebook usa esa diferencia para decidir si mostrarlo en la tarjeta.
5. Campos adicionales son opcionales. La v0.3 agrega `model_inputs`,
   `pregame_context`, `game_warnings`, `math_fingerprint`, `emission_kind`,
   `context_observed_at`, `qb_adjustment` por partido y `qb_news_adjustment` en el
   sobre; los campos contratados siguen presentes.
6. No emitir partidos sin ID ESPN. Se contabilizan en `skipped_without_espn_id`;
   nunca se inventa un ID ni se cambia el nombre de los equipos para forzar un cruce.
7. `quarterback_reference` nombra el historial que **sí** entra en la fórmula.
   Cuando el total se ajusta al QB probable, ese campo pasa a nombrarlo y
   `qb_adjustment.teams.*.model_reference` conserva la referencia anterior con su
   ID. Los QB probables de hoy siguen en `pregame_context.teams.*.quarterback`:
   reemplazar solo el texto sin cambiar las entradas falsearía la explicación.
8. El ajuste por QB probable se aplica después del contexto y antes del sello de
   emisión. Cuando `model_inputs.total` no reproduce el marcador publicado o el
   historial reconstruido no explica sus variables de QB, la emisión sale sin
   tocar con `qb_adjustment.status` en `emision_no_reproducible` o
   `estado_no_coincide`. Un fallo del índice de jugadores tampoco cancela la
   emisión: se declara y se avisa por partido.

## Estado cuantitativo v0.3

El ajuste de EPA por rival previo se seleccionó para el **total** en 2017–2019 y
se comparó retrospectivamente en 2020–2025. Error absoluto medio del total:
v0.2 **10.5411**, v0.3 **10.5233** puntos. Es una diferencia muy pequeña; ganador
y spread mantienen su matemática anterior. Se preservan informes y emisiones
anteriores, sin relabelar archivos históricos.

## Ajuste por QB probable

Desde esta versión el QB probable de hoy sustituye al QB de referencia dentro de la
regresión del total cuando difieren. Es la **misma matemática con otra entrada**:
mismos coeficientes, mismas variables, ningún peso nuevo. Solo se mueve el total,
porque las variables de QB no entran en la regresión del margen; el spread
publicado puede variar 0.1 por el redondeo de los marcadores.

Para sustituir hay que identificar al jugador en nflverse, primero por ID de ESPN
y si no por nombre único entre quarterbacks de las últimas tres temporadas. Sin
historial utilizable no hay sustitución y el partido lo avisa. El puente de
identidad se descarga con URL, hora y hash en `qb_news_adjustment.source`.

Efecto medido: no se puede reconstruir qué decían las noticias de años pasados, así
que se midió el techo del ajuste sustituyendo por el titular real de cada partido de
2020–2025. El error absoluto medio del total cambió 0.0014 puntos, con intervalo del
95% [-0.0288, 0.0305]; el titular real difiere de la referencia en 25% de los
partidos y en 37.5% de las semanas 1 y 2. **La ganancia no se distingue del azar.**
La sustitución se aplica porque la entrada anterior era falsa, no por precisión
demostrada. Informe: `reports/qb-news.json`, generado por
`scripts/evaluate_qb_news.py`. El titular real del partido predicho es un oráculo de
medición y nunca alimenta una emisión.

`scripts/preview_qb_news.py` reprocesa una copia de la emisión publicada con el
contexto que esa emisión archivó y escribe `reports/qb-news-preview.json`. Es una
simulación: no escribe emisiones ni historial.

## Datos actuales nuevos

- ESPN: reporte de lesiones, depth charts, plantillas y entrenador, con fecha
  de consulta, URL y hash de respuesta. Se verifica equipo, ID y horario del juego.
- QB probable: primero disponible en depth chart y plantilla, excluyendo bajas
  explícitas. **No equivale a titular confirmado**. Cuando difiere del QB de
  referencia, su historial entra al total y el partido lo avisa. Si no puede
  resolverse, se declara faltante y la emisión sale sin ajuste.
- Cambios de plantilla: comparación con captura anterior; en la primera emisión
  se intenta recuperar la plantilla de la temporada anterior y se valida el año.
  Esa plantilla se consulta hoy, no se presenta como archivo previo al partido.
- Open-Meteo: pronóstico horario en la ciudad del estadio, unidades explícitas,
  viento, ráfagas, temperatura y precipitación para tres horas desde el inicio.
  No es una medición dentro del estadio; el estado del techo queda sin confirmar.
  Si faltan datos o no se resuelve ciudad/estado, se declara indisponible.

**Lesiones, plantilla, entrenador y clima siguen sin alterar ninguna cifra.** La
captura es el prerrequisito para estimar y validar sus pesos sin mezclar datos
conocidos antes del partido con información posterior. La única excepción es la
identidad del QB probable, que corrige una entrada del total como se describe
arriba: no se inventan efectos de lesión ni se presenta un titular probable como
confirmado. Tampoco se ha filtrado garbage time o kneel-downs: el EPA disponible
sigue agregado por partido.

## Archivos y ejecución

`modelo-nfl run` captura el contexto y lo archiva en `data/context/`, aplica el
ajuste por QB probable, registra las entradas numéricas por partido y emite un
histórico en `data/history/`. El sello `generated_at` se fija al terminar la
captura; los partidos ya iniciados se excluyen de una emisión prospectiva.
`--as-of` histórico deshabilita consultas vivas y la emisión se marca como
retrospectiva. `--no-live-context` permite una ejecución explícita solo del motor:
sin contexto tampoco hay ajuste por QB probable. Un fallo de un proveedor no debe impedir la
emisión de predicciones del motor; sí produce avisos de datos faltantes.

Todavía no hay programación automática ni conciliación automática con resultados.
Las fuentes gratuitas pueden cambiar; su ausencia se informa, no se reemplaza con
datos inventados. Open-Meteo tiene condiciones específicas para uso comercial:
https://open-meteo.com/en/terms
