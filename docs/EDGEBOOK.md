# Contrato NFL 1.1 y conexión pendiente

Motor independiente. Edgebook presenta proyecciones; no entrena el modelo.
Archivo de intercambio: `data/edgebook-latest.json`. Es un sobre semanal con
`generated_at`, `as_of`, `sport`, versión, fuente y `games`. La fecha local del
sobre es fecha de generación, no fecha de todos los juegos.

Cada juego incluye:

- `league_game_id`: ID nflverse estable; `espn_game_id` cuando está disponible.
- `starts_at`: inicio UTC con zona horaria.
- `home` y `away`: nombre completo y código de equipo.
- `projection`: puntos esperados por equipo, coherentes con spread y total.
- `win_probability`: local/visitante, condicionada a que no haya empate.
- `pick`: moneyline, lado y etiqueta; expresa ganador, sin recomendar una cuota.
- `spread.home`: negativo si el local es favorito por margen; `away` es su opuesto.
- `total_points`: total esperado, independiente de la línea comercial.
- `intervals_80`: intervalo empírico de margen local y de total.
- `explanations`: contribuciones del margen/total en puntos.
- `data_warning` y `warnings`: datos ausentes e inicio de temporada.

Cambios necesarios en el consumidor, verificados en el checkout de EDGEBOOK:

1. Ampliar `lib/models/contract.ts` para NFL/1.1 y conservar los campos nuevos.
2. Registrar una fuente NFL adicional en `lib/server.ts`; aislar fallos por
   deporte. Hoy usa una sola URL y una sola descarga.
3. Resolver por ID ESPN cuando exista y validar equipos, deporte e inicio;
   usar nombres/fecha como respaldo. No cruzar solo por equipo sin ventana temporal.
4. Quitar la salida vacía incondicional para NFL en `ModeloView` de
   `components/edgebook/workspace.tsx`. Mostrar ganador, spread propio, total,
   fecha del modelo y advertencias. Mantener expertos separados.
5. Rechazar proyecciones posteriores al inicio para la evaluación prepartido y
   hacer visible la antigüedad. Archivar emisiones previas al partido.
6. Proveer una URL HTTPS accesible al consumidor o autenticación de servidor
   para repositorio privado. No colocar tokens en el navegador ni el JSON.

La integración y el despliegue de Edgebook no se realizan desde este repositorio.
La v0.1 genera resultados ejecutables y auditables; una pestaña existente por sí
sola no significa que el modelo ya esté activo en producción.
