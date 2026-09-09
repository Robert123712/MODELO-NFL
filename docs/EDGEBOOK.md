# Contrato NFL 1.1 y conexión pendiente

## Handoff para Claude

El usuario asigna la integración con Edgebook a Claude. El trabajo del modelo
NFL continúa por separado en este repositorio. No hace falta volver a construir
el motor para mostrar sus resultados.

- Repositorio privado del motor: `Robert123712/MODELO-NFL`, rama `main`.
- Versión actual del motor: **0.2.0**. No confundir con el contrato JSON **1.1**.
- Consumidor: `Robert123712/EDGEBOOK`. Los archivos mencionados abajo se
  verificaron en el checkout local; revisar su estado actual antes de editar.
- Snapshot real de ejemplo: `data/edgebook-latest.json`, con 16 proyecciones en
  la emisión actualmente guardada. El número de partidos cambiará en otras emisiones.
- Vista de referencia: https://edgebook-nfl-robert.rsapicks.chatgpt.site
  Es privada y **no es un endpoint público de datos**. No intentar usar su HTML
  ni asumir que un Worker de Edgebook podrá descargar su JSON sin autenticación.
- UI solicitada: conservar MLB y habilitar NFL en sus pestañas; tarjetas con
  ganador/probabilidad, spread propio y total. El contexto del modelo debe vivir
  en **un único apartado**, no en un desplegable por partido.
- El contexto NFL debe aclarar que el QB es una referencia histórica, no un
  titular confirmado, y que no se incorporan lesiones ni clima. Los porcentajes
  están condicionados a que no haya empate.
- No usar los momios como entrada del motor ni modificar la matemática al integrar.

### Entrega esperada de la integración

1. Seleccionar una fuente HTTPS de servidor que pueda leer el repositorio privado
   (por ejemplo, API autenticada de GitHub con secreto de mínimo alcance), o un
   mecanismo de publicación acordado con el usuario. No hacer público el repo ni
   cambiar permisos de Sites como atajo. Nunca exponer credenciales al cliente.
2. Mantener compatible la ingesta MLB/1.0 al añadir NFL/1.1. Un fallo de NFL no
   debe ocultar MLB ni los partidos. No quitar campos NFL durante la validación.
3. Mostrar datos del snapshot cruzados con el partido correcto y su fecha real;
   manejar juegos sin proyección, fechas distintas y fuentes desactualizadas.
4. Verificar ambos modelos con sus snapshots, el signo del spread y la ausencia
   de credenciales en la respuesta al navegador. Confirmar el estado del despliegue
   antes de decir que NFL ya está activo en Edgebook.

### Lo que todavía NO existe

No hay capturador automático programado ni cruce automático de emisiones con
resultados finales. `modelo-nfl run` guarda un histórico por ejecución; GitHub
Actions permite ejecución manual. La pantalla publicada es una emisión estática,
no un marcador en vivo ni una garantía de actualización. No presentar esos flujos
como implementados durante la integración.

## Campos y cambios técnicos

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
La v0.2 genera resultados ejecutables y auditables; una pestaña existente por sí
sola no significa que el modelo ya esté activo en producción.
