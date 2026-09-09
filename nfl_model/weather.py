"""Pronóstico horario gratuito de Open-Meteo, para la ciudad del estadio."""
from urllib.parse import urlencode

import pandas as pd

STATES = {'WA':'Washington','CA':'California','AZ':'Arizona','CO':'Colorado','TX':'Texas',
          'FL':'Florida','GA':'Georgia','NC':'North Carolina','SC':'South Carolina','LA':'Louisiana',
          'TN':'Tennessee','MO':'Missouri','OH':'Ohio','PA':'Pennsylvania','MD':'Maryland',
          'NJ':'New Jersey','NY':'New York','MA':'Massachusetts','MI':'Michigan','IN':'Indiana',
          'IL':'Illinois','WI':'Wisconsin','MN':'Minnesota','NV':'Nevada','DC':'District of Columbia'}
COUNTRIES = {'USA':'US','Australia':'AU','United Kingdom':'GB','Germany':'DE','Spain':'ES','Ireland':'IE','Mexico':'MX','Brazil':'BR'}


def forecast(venue, starts_at, get_json):
    address = venue.get('address',{})
    city, state = address.get('city'), address.get('state')
    country = COUNTRIES.get(address.get('country'))
    if not country or not city or (country=='US' and state not in STATES):
        return {'status':'unavailable','reason':'Ciudad fuera de la cobertura validada'}
    url = 'https://geocoding-api.open-meteo.com/v1/search?' + urlencode({'name':city,'count':10,'countryCode':country,'language':'en'})
    geo, geo_source = get_json(url)
    matches = [r for r in geo.get('results',[]) if r.get('country_code')==country and r.get('name','').casefold()==city.casefold() and
               (r.get('admin1')==STATES[state] if country=='US' else r.get('name','').casefold()==city.casefold())]
    if len(matches)!=1:
        return {'status':'unavailable','reason':'No se pudo resolver la ciudad y estado del estadio'}
    point = matches[0]
    url = 'https://api.open-meteo.com/v1/forecast?' + urlencode({
        'latitude':point['latitude'],'longitude':point['longitude'],'timezone':'UTC','forecast_days':16,
        'hourly':'temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation'})
    raw, source = get_json(url)
    hourly = raw.get('hourly',{})
    times = pd.to_datetime(hourly.get('time',[]),utc=True)
    start = pd.Timestamp(starts_at).floor('h')
    indices = [i for i,t in enumerate(times) if start <= t < start+pd.Timedelta(hours=3)]
    units = raw.get('hourly_units',{})
    if len(indices)!=3 or units.get('temperature_2m')!='°C' or units.get('precipitation')!='mm' or units.get('wind_speed_10m')!='km/h' or units.get('wind_gusts_10m')!='km/h':
        return {'status':'unavailable','reason':'Pronóstico incompleto o unidades no reconocidas','source':source}
    values = {key:[hourly[key][i] for i in indices] for key in ('temperature_2m','wind_speed_10m','wind_gusts_10m','precipitation')}
    if any(v is None for array in values.values() for v in array):
        return {'status':'unavailable','reason':'Pronóstico con valores faltantes','source':source}
    return {'status':'available','source':source,'geocoding_source':geo_source,
            'forecast_valid_at':start.isoformat(),'venue':venue.get('fullName'),
            'location_basis':'Ciudad del estadio, no medición dentro del campo',
            'latitude':point['latitude'],'longitude':point['longitude'],
            'temperature_c':values['temperature_2m'][0],'wind_max_kmh':max(values['wind_speed_10m']),
            'gust_max_kmh':max(values['wind_gusts_10m']),'precipitation_mm_3h':sum(values['precipitation']),
            'roof_status':'not_confirmed','quantitative_adjustment_applied':False}
