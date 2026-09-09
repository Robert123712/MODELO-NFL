"""Contexto previo al partido, consultado y archivado con fecha; nunca se retrodata."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import unicodedata

import pandas as pd
import requests
from .weather import forecast

API = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl'
UNAVAILABLE = {'out', 'injured reserve', 'suspension', 'suspended', 'pup', 'reserve'}
QB_MISMATCH_NOTE = '{code}: QB probable distinto al usado por el modelo'


def normalized(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', value).lower() if c.isalnum())


def get_json(url):
    res = requests.get(url, timeout=25)
    res.raise_for_status()
    content = res.content
    return res.json(), {'url':url, 'observed_at':datetime.now(timezone.utc).isoformat(),
                        'sha256':hashlib.sha256(content).hexdigest()}


def parse_injuries(payload, team_id, as_of):
    rows = []
    found = False
    for item in payload.get('injuries', []):
        if str(item.get('team', {}).get('id', item.get('id'))) != str(team_id):
            continue
        found = True
        for injury in item.get('injuries', []):
            date = injury.get('date')
            # Nunca incorporar un aviso fechado después del momento observado.
            if date and pd.Timestamp(date) > as_of:
                continue
            athlete = injury.get('athlete', {})
            player_id = str(athlete.get('id', ''))
            if not player_id:
                for link in athlete.get('links',[]):
                    match = re.search(r'^https://www\.espn\.com/nfl/player/(?:[^/]+/)?_/id/(\d+)(?:/|$)',link.get('href',''))
                    if match:
                        player_id = match.group(1)
                        break
            rows.append({'espn_player_id':player_id,
                         'name':athlete.get('displayName', ''),
                         'position':athlete.get('position', {}).get('abbreviation'),
                         'status':injury.get('status'), 'reported_at':date,
                         'injury_type':injury.get('details', {}).get('type')})
    return {'status':'reported' if found else 'unavailable', 'players':rows,
            'coverage':'provider_list_not_guaranteed_complete',
            'meaning':'Reporte del proveedor; lista vacía no garantiza plantilla sana.'}


def parse_roster(payload):
    # Lista blanca: no se guardan biografías, fotos, enlaces ni estadísticas ajenas al contexto.
    athletes = [{'espn_player_id':str(a['id']), 'name':a.get('displayName', ''),
                 'position':a.get('position',{}).get('abbreviation'),
                 'status':a.get('status',{}).get('name')}
                for group in payload.get('athletes', []) for a in group.get('items', []) if a.get('id')]
    coach = payload.get('coach', [])
    if isinstance(coach, dict):
        coach = [coach]
    return {'players':athletes, 'coaches':[{'name':c.get('firstName','')+' '+c.get('lastName',''),
                                          'espn_coach_id':str(c.get('id',''))} for c in coach],
            'status':'available' if athletes else 'unavailable'}


def parse_quarterback(payload, injuries, roster):
    ordered = []
    for chart in payload.get('depthchart', []):
        for key, position in chart.get('positions', {}).items():
            if key.lower() == 'qb' or position.get('position',{}).get('abbreviation') == 'QB':
                ordered += position.get('athletes', [])
    distinct = {str(q.get('id')):q for q in ordered if q.get('id')}
    unavailable = {p['espn_player_id'] for p in injuries['players']
                   if (p.get('status') or '').lower() in UNAVAILABLE}
    roster_ids = {p['espn_player_id'] for p in roster['players']}
    unavailable |= {p['espn_player_id'] for p in roster['players'] if (p.get('status') or '').lower() in UNAVAILABLE}
    candidates = [{'espn_player_id':pid,'name':q.get('displayName',''),
                   'reported_unavailable':pid in unavailable,
                   'on_current_roster':pid in roster_ids} for pid,q in distinct.items()]
    eligible = [q for q in candidates if not q['reported_unavailable'] and q['on_current_roster']]
    return {'status':'projected_not_confirmed' if eligible else 'unavailable',
            'projected':eligible[0] if eligible else None, 'depth_order':candidates,
            'confirmed':False, 'basis':'Orden de depth chart y reporte de bajas; no anuncio oficial de titular.'}


def compare_roster(current, reference):
    if not reference:
        return {'status':'baseline_not_yet_available','added':[],'removed':[], 'coach_changed':None}
    before = {p['espn_player_id']:p for p in reference.get('players',[])}
    after = {p['espn_player_id']:p for p in current['players']}
    return {'status':'compared_to_previous_capture',
            'added':[after[k] for k in sorted(after.keys()-before.keys())],
            'removed':[before[k] for k in sorted(before.keys()-after.keys())],
            'coach_changed':current['coaches'] != reference.get('coaches', [])}


def previous_rosters(root, observed_at):
    found = {}
    for path in sorted((root/'data/context').glob('*.json'), reverse=True):
        stored = json.loads(path.read_text())
        if pd.Timestamp(stored['observed_at']) >= observed_at:
            continue
        for game in stored['games'].values():
            for team in game.get('teams',{}).values():
                tid = team['espn_team_id']
                if tid not in found and team.get('roster',{}).get('status') == 'available':
                    found[tid] = team['roster']
    return found


def collect_context(snapshot, root: Path):
    observed_at = pd.Timestamp.now(tz='UTC')
    old = previous_rosters(root, observed_at)
    # Un --as-of histórico no puede consultar el estado vivo de hoy como si fuera pasado.
    if abs((observed_at-pd.Timestamp(snapshot['as_of'])).total_seconds()) > 3600:
        return {'observed_at':observed_at.isoformat(),'status':'disabled_for_historical_as_of','games':{}}
    try:
        all_injuries, injuries_source = get_json(f'{API}/injuries?limit=1000')
        if int(all_injuries.get('season',{}).get('year',0)) != (snapshot['games'][0]['season'] if snapshot['games'] else 0):
            raise ValueError('Lesiones de otra temporada')
    except (requests.RequestException,ValueError,KeyError,TypeError):
        all_injuries, injuries_source = None, None

    def collect(game):
        gid = game['league_game_id']
        if not game.get('espn_game_id') or pd.Timestamp(game['starts_at']) <= observed_at:
            return gid, {'status':'unavailable','reason':'Sin ID ESPN o partido iniciado'}
        try:
            summary, provenance = get_json(f'{API}/summary?event={game["espn_game_id"]}')
            comp = summary['header']['competitions'][0]
            if str(comp['id']) != game['espn_game_id']:
                raise ValueError('ID ESPN no coincide')
            if abs((pd.Timestamp(comp['date'])-pd.Timestamp(game['starts_at'])).total_seconds()) > 5400:
                raise ValueError('Horario de ESPN cambió; actualizar calendario antes de cruzar')
            if comp.get('status',{}).get('type',{}).get('state') in ('in','post'):
                raise ValueError('El partido ya no está en prepartido')
            output = {'status':'available','observed_at':provenance['observed_at'],
                      'source':provenance,'teams':{},'quantitative_adjustment_applied':False}
            for competitor in comp['competitors']:
                side, tid = competitor['homeAway'], str(competitor['id'])
                if side not in ('home','away'):
                    continue
                expected = game[side]['code']
                actual = competitor['team']['abbreviation']
                aliases = {'WSH':'WAS','LAR':'LA','OAK':'LV','SD':'LAC'}
                if aliases.get(actual,actual) != expected:
                    raise ValueError('Equipo ESPN no coincide con snapshot')
                injuries = parse_injuries(all_injuries or summary, tid, pd.Timestamp.now(tz='UTC'))
                if all_injuries is None:
                    injuries['coverage'] = 'limited_game_summary'
                item = {'espn_team_id':tid,'injuries':injuries,'sources':[injuries_source] if injuries_source else []}
                try:
                    raw, source = get_json(f'{API}/teams/{tid}/roster')
                    if str(raw.get('team',{}).get('id')) != tid:
                        raise ValueError('Roster de otro equipo')
                    item['roster'] = parse_roster(raw)
                    item['sources'].append(source)
                    baseline = old.get(tid)
                    comparison_basis = 'previous_capture'
                    item['offseason_roster_changes'] = {'status':'unavailable'}
                    try:
                        past, past_source = get_json(f'{API}/teams/{tid}/roster?season={game["season"]-1}')
                        if int(past.get('season',{}).get('year',0)) == game['season']-1:
                            previous_season = parse_roster(past)
                            item['sources'].append(past_source)
                            item['offseason_roster_changes'] = compare_roster(item['roster'], previous_season)
                            item['offseason_roster_changes']['comparison_basis'] = 'previous_season_roster_retrieved_today'
                            if baseline is None:
                                baseline = previous_season
                                comparison_basis = 'previous_season_roster_retrieved_today'
                    except (requests.RequestException, ValueError, KeyError, TypeError):
                        pass
                    item['roster_changes'] = compare_roster(item['roster'], baseline)
                    item['roster_changes']['comparison_basis'] = comparison_basis
                except (requests.RequestException, ValueError, KeyError) as error:
                    item['roster'] = {'status':'unavailable','players':[],'coaches':[]}
                    item['roster_changes'] = {'status':'unavailable'}
                    item['roster_error'] = str(error)[:180]
                try:
                    raw, source = get_json(f'{API}/teams/{tid}/depthcharts')
                    if str(raw.get('team',{}).get('id')) != tid:
                        raise ValueError('Depth chart de otro equipo')
                    item['quarterback'] = parse_quarterback(raw, injuries, item['roster'])
                    item['sources'].append(source)
                except (requests.RequestException, ValueError, KeyError) as error:
                    item['quarterback'] = {'status':'unavailable','confirmed':False,'projected':None}
                    item['quarterback_error'] = str(error)[:180]
                ref = game.get('quarterback_reference') or {}
                projected = item['quarterback'].get('projected')
                item['quarterback']['differs_from_model_reference'] = (
                    normalized(projected['name']) != normalized(ref[side]) if projected and ref.get(side) else None)
                output['teams'][side] = item
            venue = summary.get('gameInfo',{}).get('venue',{})
            try:
                output['weather'] = forecast(venue, game['starts_at'], get_json)
            except (requests.RequestException, ValueError, KeyError, IndexError) as error:
                output['weather'] = {'status':'unavailable','reason':str(error)[:180]}
            output['observed_at'] = datetime.now(timezone.utc).isoformat()
            if pd.Timestamp(game['starts_at']) <= pd.Timestamp(output['observed_at']):
                return gid, {'status':'unavailable','reason':'El partido empezó durante la consulta'}
            return gid, output
        except (requests.RequestException, ValueError, KeyError, IndexError) as error:
            return gid, {'status':'unavailable','reason':str(error)[:180]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        games = dict(pool.map(collect, snapshot['games']))
    return {'observed_at':datetime.now(timezone.utc).isoformat(),'status':'collected','games':games}


def attach_context(snapshot, context):
    for game in snapshot['games']:
        info = context['games'].get(game['league_game_id'], {'status':'unavailable','reason':context['status']})
        game['pregame_context'] = info
        issues = []
        if info['status'] != 'available':
            issues.append('Contexto actual no disponible')
        for side,item in info.get('teams',{}).items():
            qb = item['quarterback']
            if qb.get('differs_from_model_reference'):
                issues.append(QB_MISMATCH_NOTE.format(code=game[side]['code']))
            if item['injuries']['status'] == 'unavailable':
                issues.append(f'{game[side]["code"]}: reporte de lesiones no disponible')
            if qb['status'] == 'unavailable':
                issues.append(f'{game[side]["code"]}: QB probable no disponible')
        if info.get('weather',{}).get('status') == 'unavailable':
            issues.append('Clima de este partido no disponible')
        # Solo límites GENERALES en warnings. Los avisos específicos no deben desaparecer en Edgebook.
        previous = game.get('data_warning')
        if previous and previous not in game['warnings']:
            issues.insert(0, previous)
        if issues:
            game['game_warnings'] = issues
            game['data_warning'] = '; '.join(issues)[:200]
        game['warnings'] = list(dict.fromkeys(game['warnings'] + [
            'Lesiones, plantilla y clima se registran como contexto; aún no tienen pesos numéricos validados.',
            'Un depth chart indica QB probable, no titular confirmado.']))[:10]
    snapshot['context_observed_at'] = context['observed_at']
    return snapshot
