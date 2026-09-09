"""Regresión del contrato de producción, sin necesitar el checkout de Edgebook."""
import json
from pathlib import Path

from nfl_model import __version__


def test_published_snapshot_keeps_edgebook_contract():
    raw = (Path(__file__).resolve().parents[1]/'data/edgebook-latest.json').read_text()
    assert len(raw.encode())<4_000_000
    data = json.loads(raw)
    assert data['schema_version']=='1.1'
    assert data['model_version'].startswith(__version__+'-')
    assert data['model_version'].endswith(data['math_fingerprint'][:12])
    assert len(data['games'])<=40
    assert isinstance(data['skipped_without_id'],int)
    for game in data['games']:
        required = {'league_game_id','espn_game_id','starts_at','home','away','projection',
                    'win_probability','pick','data_warning','warnings','probability_basis',
                    'spread','total_points','explanations'}
        assert required <= game.keys()
        assert isinstance(game['espn_game_id'],str) and game['espn_game_id'].isdigit()
        assert game['pick']['market']=='moneyline'
        assert game['pick']['side'] in ('home','away')
        assert abs(sum(game['win_probability'].values())-1)<1e-6
        assert len(game['warnings'])<=10 and all(len(w)<=300 for w in game['warnings'])
        assert game['data_warning'] is None or len(game['data_warning'])<=200
        for side in ('home','away'):
            assert all(isinstance(game[side][key],str) for key in ('name','code'))
            assert 0<=game['projection'][side+'_score']<=100
        for factor in game['explanations']['margin']['top_factors']:
            assert isinstance(factor['label'],str) and len(factor['label'])<=120
            assert isinstance(factor['contribution_points'],(int,float))
        if game.get('game_warnings'):
            assert game['data_warning'] not in game['warnings']


def test_history_is_not_relabelled_as_current_math():
    history = Path(__file__).resolve().parents[1]/'data/history'
    versions = {json.loads(p.read_text())['model_version'] for p in history.glob('*.json')}
    assert '0.1.0' in versions and '0.2.0' in versions
