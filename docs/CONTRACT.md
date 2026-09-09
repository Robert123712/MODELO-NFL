# Contrato activo con Edgebook

Edgebook ya consume `data/edgebook-latest.json` en producción. Estas restricciones
se conservan al modificar el motor:

1. Cambiar `model_version` cuando cambia la matemática. Desde v0.3 se agrega una
   huella de código matemático, variables, coeficientes, escalas y calibración:
   `0.3.0-<12 caracteres>`. Entrenamientos o configuraciones distintos no comparten
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
   `pregame_context`, `game_warnings`, `math_fingerprint`, `emission_kind` y
   `context_observed_at`; los campos contratados siguen presentes.
6. No emitir partidos sin ID ESPN. Se contabilizan en `skipped_without_espn_id`;
   nunca se inventa un ID ni se cambia el nombre de los equipos para forzar un cruce.
7. `quarterback_reference` conserva las referencias históricas usadas en la
   fórmula. Los QB probables de hoy viven en `pregame_context.teams.*.quarterback`:
   reemplazar solo el texto sin cambiar las entradas falsearía la explicación.

## Estado cuantitativo v0.3

El ajuste de EPA por rival previo se seleccionó para el **total** en 2017–2019 y
se comparó retrospectivamente en 2020–2025. Error absoluto medio del total:
v0.2 **10.5411**, v0.3 **10.5233** puntos. Es una diferencia muy pequeña; ganador
y spread mantienen su matemática anterior. Se preservan informes y emisiones
anteriores, sin relabelar archivos históricos.

## Datos actuales nuevos

- ESPN: reporte de lesiones, depth charts, plantillas y entrenador, con fecha
  de consulta, URL y hash de respuesta. Se verifica equipo, ID y horario del juego.
- QB probable: primero disponible en depth chart y plantilla, excluyendo bajas
  explícitas. **No equivale a titular confirmado**. Se avisa cuando difiere del
  QB de referencia del modelo. Si no puede resolverse, se declara faltante.
- Cambios de plantilla: comparación con captura anterior; en la primera emisión
  se intenta recuperar la plantilla de la temporada anterior y se valida el año.
  Esa plantilla se consulta hoy, no se presenta como archivo previo al partido.
- Open-Meteo: pronóstico horario en la ciudad del estadio, unidades explícitas,
  viento, ráfagas, temperatura y precipitación para tres horas desde el inicio.
  No es una medición dentro del estadio; el estado del techo queda sin confirmar.
  Si faltan datos o no se resuelve ciudad/estado, se declara indisponible.

**Esos datos actuales todavía no alteran las probabilidades ni suman/restan puntos.**
La captura es el prerrequisito para estimar y validar sus pesos sin mezclar datos
conocidos antes del partido con información posterior. No se inventan efectos de
lesión ni se presenta un titular probable como confirmado. Tampoco se ha filtrado
garbage time o kneel-downs: el EPA disponible sigue agregado por partido.

## Archivos y ejecución

`modelo-nfl run` captura el contexto y lo archiva en `data/context/`, registra las
entradas numéricas por partido y emite un histórico en `data/history/`. El sello
`generated_at` se fija al terminar la captura; los partidos ya iniciados se
excluyen de una emisión prospectiva. `--as-of` histórico deshabilita consultas
vivas y la emisión se marca como retrospectiva. `--no-live-context` permite una
ejecución explícita solo del motor. Un fallo de un proveedor no debe impedir la
emisión de predicciones del motor; sí produce avisos de datos faltantes.

Todavía no hay programación automática ni conciliación automática con resultados.
Las fuentes gratuitas pueden cambiar; su ausencia se informa, no se reemplaza con
datos inventados. Open-Meteo tiene condiciones específicas para uso comercial:
https://open-meteo.com/en/terms
