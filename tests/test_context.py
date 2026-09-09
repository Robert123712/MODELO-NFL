import pandas as pd

from nfl_model.live_context import parse_injuries, parse_quarterback, compare_roster, attach_context, collect_context
from nfl_model.weather import forecast


def test_depth_chart_is_not_confirmation_and_excludes_out_qb():
    raw = {'depthchart':[{'positions':{'qb':{'athletes':[{'id':'1','displayName':'QB Out'}, {'id':'2','displayName':'QB Backup'}]}}}]}
    injuries = {'players':[{'espn_player_id':'1','status':'Out'}]}
    roster = {'players':[{'espn_player_id':str(i),'status':'Active'} for i in [1,2]]}
    result = parse_quarterback(raw,injuries,roster)
    assert result['projected']['espn_player_id']=='2'
    assert result['confirmed'] is False
    assert result['status']=='projected_not_confirmed'


def test_future_injury_reports_are_not_read():
    raw = {'injuries':[{'team':{'id':'1'},'injuries':[{'date':'2026-09-10T00:00Z','status':'Out'}]}]}
    result = parse_injuries(raw,'1',pd.Timestamp('2026-09-09T00:00Z'))
    assert result['players']==[]
    assert parse_injuries({},'1',pd.Timestamp.now(tz='UTC'))['status']=='unavailable'


def test_roster_changes_are_identity_based():
    old = {'players':[{'espn_player_id':'1','name':'Old'}],'coaches':[{'name':'Coach A'}]}
    new = {'players':[{'espn_player_id':'2','name':'New'}],'coaches':[{'name':'Coach B'}]}
    result = compare_roster(new,old)
    assert result['added'][0]['name']=='New'
    assert result['removed'][0]['name']=='Old'
    assert result['coach_changed']
    assert compare_roster(new,None)['status']=='baseline_not_yet_available'


def test_context_specific_warnings_are_not_general():
    snapshot = {'games':[{'league_game_id':'a','data_warning':'Historial limitado','warnings':['Límite general'],'home':{'code':'KC'},'away':{'code':'BUF'}}]}
    result = attach_context(snapshot,{'observed_at':'2026-09-09T00:00Z','status':'collected','games':{}})
    game = result['games'][0]
    assert game['data_warning'] not in game['warnings']
    assert 'Historial limitado' in game['data_warning']
    assert all('Historial limitado' not in w for w in game['warnings'])


def test_historical_as_of_never_fetches_live_data(tmp_path,monkeypatch):
    def blocked(*args):
        raise AssertionError('Consultó datos actuales para un corte histórico')
    monkeypatch.setattr('nfl_model.live_context.get_json',blocked)
    result = collect_context({'as_of':'2000-01-01T00:00:00Z','games':[]},tmp_path)
    assert result['status']=='disabled_for_historical_as_of'


def test_forecast_uses_kickoff_window_and_explicit_units():
    responses = [
        {'results':[{'name':'Seattle','country_code':'US','admin1':'Washington','latitude':47.6,'longitude':-122.3}]},
        {'hourly_units':{'temperature_2m':'°C','wind_speed_10m':'km/h','wind_gusts_10m':'km/h','precipitation':'mm'},
         'hourly':{'time':['2026-09-10T00:00','2026-09-10T01:00','2026-09-10T02:00'],
                   'temperature_2m':[20,19,18],'wind_speed_10m':[8,12,10],'wind_gusts_10m':[11,18,14],'precipitation':[0,1,2]}}
    ]
    def fake(url):
        return responses.pop(0), {'url':url,'observed_at':'2026-09-09T00:00:00Z'}
    result = forecast({'address':{'country':'USA','city':'Seattle','state':'WA'}},'2026-09-10T00:20:00Z',fake)
    assert result['wind_max_kmh']==12
    assert result['precipitation_mm_3h']==3
    assert result['forecast_valid_at']=='2026-09-10T00:00:00+00:00'
