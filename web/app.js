const zone = 'America/Chihuahua';
const format = new Intl.NumberFormat('es-MX', {maximumFractionDigits:1,minimumFractionDigits:1});
const signed = value => `${value > 0 ? '+' : value < 0 ? '−' : ''}${format.format(Math.abs(value))}`;
const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const dateKey = date => new Intl.DateTimeFormat('en-CA',{timeZone:zone,year:'numeric',month:'2-digit',day:'2-digit'}).format(date);
const dateLabel = date => new Intl.DateTimeFormat('es-MX',{timeZone:zone,weekday:'long',day:'numeric',month:'long'}).format(date);
const clock = date => new Intl.DateTimeFormat('es-MX',{timeZone:zone,hour:'numeric',minute:'2-digit',hour12:true}).format(date);

function renderGame(game) {
  const start = new Date(game.starts_at);
  const side = game.pick.side;
  const winner = game[side];
  const prob = Math.round(game.win_probability[side]*100);
  const favoriteSide = game.spread.home <= 0 ? 'home' : 'away';
  const spread = game.spread.home === 0 ? 'Parejo' : `${escape(game[favoriteSide].code)} ${signed(game.spread[favoriteSide])}`;
  const state = start > new Date() ? 'Proyección previa' : 'Emisión previa · partido iniciado';
  const team = side => `<div class="team"><span class="code">${escape(game[side].code)}</span><span class="team-name">${escape(game[side].name)}</span><span class="score">${format.format(game.projection[`${side}_score`])}</span></div>`;
  const factors = (game.explanations?.margin?.top_factors ?? []).map(f => escape(f.label)).join(' · ');
  const totalFactors = (game.explanations?.total?.top_factors ?? []).map(f => escape(f.label)).join(' · ');
  const qb = game.quarterback_reference;
  const interval = game.intervals_80?.total;
  return `<article class="game" aria-label="${escape(game.away.name)} contra ${escape(game.home.name)}">
    <div class="game-top"><span class="status">${state}</span><time datetime="${escape(game.starts_at)}">${clock(start)}</time></div>
    <div class="teams">${team('away')}${team('home')}<p class="score-label">MARCADOR PROYECTADO · VISITANTE / LOCAL</p></div>
    <div class="prediction"><div><span class="winner-label">GANADOR ESTIMADO</span><span class="winner-name">${escape(winner.name)}</span></div><div class="prob">${prob}%<small>probabilidad</small></div></div>
    <div class="markets"><div class="market"><span class="market-label">Spread propio</span><span class="market-value">${spread}</span></div><div class="market"><span class="market-label">Total de puntos</span><span class="market-value">${format.format(game.total_points)}</span></div></div>
    <details><summary>Ver contexto del modelo</summary><p class="details-copy">${escape((game.warnings ?? []).join('. '))}.</p>${qb ? `<p class="details-copy">QB históricos de referencia: ${escape(qb.away ?? 'Sin datos')} / ${escape(qb.home ?? 'Sin datos')}. No son titulares confirmados.</p>` : ''}${factors ? `<p class="details-copy">Factores principales del margen: ${factors}.</p>` : ''}${totalFactors ? `<p class="details-copy">Factores principales del total: ${totalFactors}.</p>` : ''}${interval ? `<p class="details-copy">Intervalo estimado del total al 80%: ${format.format(interval[0])}–${format.format(interval[1])} puntos. La cobertura futura no está garantizada.</p>` : ''}</details>
  </article>`;
}

async function load() {
  const container = document.getElementById('games');
  try {
    const response = await fetch('./data/edgebook-latest.json',{cache:'no-store'});
    if (!response.ok) throw new Error('No se pudieron cargar las proyecciones');
    const data = await response.json();
    if (data.sport !== 'NFL' || !Array.isArray(data.games)) throw new Error('Datos no válidos');
    const games = [...data.games].sort((a,b)=>new Date(a.starts_at)-new Date(b.starts_at));
    document.getElementById('count').textContent = `${games.length} partidos · Temporada ${games[0]?.season ?? ''}`;
    document.getElementById('updated').textContent = `Actualizado ${dateLabel(new Date(data.generated_at))}, ${clock(new Date(data.generated_at))} · Chihuahua`;
    document.getElementById('version').textContent = `v${data.model_version}`;
    if (!games.length) {container.innerHTML = '<p class="state">No hay partidos en la ventana de esta emisión.</p>'; return;}
    const groups = new Map();
    for (const game of games) {const key=dateKey(new Date(game.starts_at));if(!groups.has(key))groups.set(key,[]);groups.get(key).push(game);}
    container.innerHTML = [...groups.values()].map(group=>`<section><h2 class="day-heading">${dateLabel(new Date(group[0].starts_at))}</h2><div class="game-grid">${group.map(renderGame).join('')}</div></section>`).join('');
  } catch (error) {
    document.getElementById('count').textContent = 'Proyecciones no disponibles';
    container.innerHTML = '<p class="state">No se pudieron cargar los partidos. Recarga la página para intentarlo de nuevo.</p>';
  }
}
load();
