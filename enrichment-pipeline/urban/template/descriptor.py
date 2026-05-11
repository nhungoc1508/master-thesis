"""
Template function turning enriched data into natural language descriptions (point level)

Slot-based cross-domain template:
    [AGENT]             transport mode + behavioral phase + speed
    [CORRIDOR]          road type + name + speed limit
    [INFRASTRUCTURE]    nearest POI cluster summary
    [REGULATORY_ZONE]   land use / functional zone
    [BEHAVIORAL_PHASE]  phase label
    [ENVIRONMENT]       weather conditions
    [TEMPORAL]          time of day + day type + season
    [SPATIAL_CONTEXT]   POI density summary

Implementation details:
    - All slots are always included
    - Use 'no data' when the underlying enrichment value is absent

Enriched columns from different phases:
    - Temporal: [x] hour_of_day, [x] day_of_week, [x] is_weekend,
        [ ] month, [x] season, [x] time_of_day_category
    - Behavioral: [x] speed_ms, [x] acceleration_ms2, [x] behavioral_phase
    - Road network: [x] road_type, [x] road_name, [x] speed_limit_kmh,
        [x] road_lanes, [x] road_oneway
    - POI: [x] poi_count_[50|200|500]m, [x] poi_categories_[50|200|500]m
    - Land use: [x] land_use
    - Weather: [x] temperature_c, [x] precipitation_mm, [x] wind_speed_kmh,
        [x] wind_direction_deg, [ ] weather_code, [x] weather_description
"""
from __future__ import annotations

import json
import math
from typing import Any

_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def generate_point_descriptor(row: dict[str, Any]) -> str:
    """Convert one enriched GPS point record to a slot-based natural language string"""
    slots = [
        _slot_agent(row),
        _slot_corridor(row),
        _slot_infrastructure(row),
        _slot_regulatory_zone(row),
        _slot_behavioral_phase(row),
        _slot_environment(row),
        _slot_temporal(row),
        _slot_spatial_context(row)
    ]
    return ' '.join(slots)

def _slot_agent(r: dict) -> str:
    mode = r.get('transport_mode') or 'vehicle'
    phase = r.get('behavioral_phase') or 'moving'
    spd = r.get('speed_ms')
    if _valid_float(spd):
        spd_str = f'{spd * 3.6:.2f} km/h'
        return f'[AGENT]: {mode}, {phase} at {spd_str}'
    return f'[AGENT]: {mode}, {phase}'

def _slot_corridor(r: dict) -> str:
    rt = str(r.get('road_type') or 'unknown')
    desc = rt.replace('_', ' ') # simple method: 'primary_link' -> 'primary link'
    name = str(r.get('road_name') or '').strip()
    lim = r.get('speed_limit_kmh')
    lanes = _safe_int(r.get('road_lanes'))
    one_way = bool(r.get('road_oneway'))
    parts = [desc]
    if name:
        parts.append(f'named {name}')
    if _valid_float(lim):
        parts.append(f'{lim:.0f} km/h limit')
    if lanes != 0:
        parts.append(f'has {lanes} lanes')
    parts.append(f'is{" not" if not one_way else ""} one-way')
    return f'[CORRIDOR]: {", ".join(parts)}.'

def _slot_infrastructure(r: dict) -> str:
    radii = [50, 200, 500]
    n_cats = 2
    # r_summary: dict[int, tuple[int, str]] = {} # { radius: (count, cat_str) }
    parts = []
    for radius in radii:
        count = _safe_int(r.get(f'poi_count_{radius}m'))
        cats = _top_categories(r.get(f'poi_categories_{radius}m') or '{}', n=n_cats)
        cat_str = f'({", ".join(cats)})' if cats else ''
        # r_summary[radius] = (count, cat_str)
        if count > 0:
            parts.append(f'{count} point(s) of interest within {radius}m {cat_str}')
    if len(parts) > 0:
        return f'[INFRASTRUCTURE]: {", ".join(parts)}.'
    return '[INFRASTRUCTURE]: no points of interest nearby.'

def _slot_regulatory_zone(r: dict) -> str:
    lu = str(r.get('land_use') or 'unknown').strip()
    if lu and lu not in ('unknown', 'other', ''):
        return f'[REGULATORY_ZONE]: {lu.replace("_", " ")} zone.'
    return '[REGULATORY_ZONE]: no designated land use zone.'

def _slot_behavioral_phase(r: dict) -> str:
    phase = str(r.get('behavioral_phase') or 'unknown')
    spd = r.get('speed_ms')
    acc = r.get('acceleration_ms2')
    parts = [phase]
    if _valid_float(spd):
        parts.append(f'{spd * 3.6:.2f} km/h')
    if _valid_float(acc) and abs(acc) > 0.5:
        direction = 'accelerating' if acc > 0 else 'decelerating'
        parts.append(f'{direction} at {abs(acc):.2f} m/s^2')
    return f'[BEHAVIORAL_PHASE]: {", ".join(parts)}.'

def _slot_environment(r: dict) -> str:
    temp = r.get('temperature_c')
    prec = r.get('precipitation_mm')
    wind = r.get('wind_speed_kmh')
    wdir = r.get('wind_direction_deg')
    wdesc = str(r.get('weather_description') or '').strip()
    parts = []
    if _valid_float(temp):
        parts.append(f'{temp:.1f} degrees Celcius')
    if wdesc and wdesc not in ('unknown', ''):
        parts.append(wdesc)
    if _valid_float(prec) and prec > 0.05:
        parts.append(f'{prec:.1f} mm/h precipitation')
    if _valid_float(wind):
        parts.append(f'wind {wind:.2f} km/h')
    if _valid_float(wdir):
        parts.append(f'wind direction {wdir:.1f} degrees')
    if not parts:
        return '[ENVIRONMENT]: weather data unavailable.'
    return f'[ENVIRONMENT]: {", ".join(parts)}'

def _slot_temporal(r: dict) -> str:
    hour = r.get('hour_of_day')
    dow = r.get('day_of_week')
    weekend = r.get('is_weekend')
    tcat = str(r.get('time_of_day_category') or '').replace('_', ' ')
    season = str(r.get('season') or '')
    parts = []
    if isinstance(dow, (int, float)) and 0 <= int(dow) <= 6:
        parts.append(_DAYS[int(dow)])
    if isinstance(hour, (int, float)):
        parts.append(f'{int(hour):02d}:00')
    if tcat:
        parts.append(tcat)
    if season:
        parts.append(season)
    if isinstance(weekend, bool) and weekend:
        parts.append('weekend')
    if not parts:
        return '[TEMPORAL]: time unknown.'
    return f'[TEMPORAL]: {", ".join(parts)}.'

def _slot_spatial_context(r: dict) -> str:
    c500 = _safe_int(r.get('poi_count_500m'))
    if c500 == 0:
        density = 'sparse'
    elif c500 < 10:
        density = 'low'
    elif c500 < 40:
        density = 'moderate'
    else:
        density = 'high'
    return f'[SPATIAL_CONTEXT]: {density} point of interest density ({c500} points of interest within 500 m).'

# ---------- Helper functions ----------

def _valid_float(v: Any) -> bool:
    return v is not None and isinstance(v, (int, float)) and not math.isnan(float(v))

def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0
    
def _top_categories(json_str: str, n: int = 2) -> list[str]:
    try:
        d = json.loads(json_str)
        return [k.replace('_', ' ') for k, _ in sorted(d.items(), key=lambda x: -x[1])[:n]]
    except (json.JSONDecodeError, AttributeError):
        return []