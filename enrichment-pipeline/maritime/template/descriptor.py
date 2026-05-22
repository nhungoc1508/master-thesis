"""
Template function turning enriched data into natural language descriptions (point level)

Slot-based cross-domain template:
    [AGENT]             vessel type + draught + behavioral phase + SOG
    [CORRIDOR]          sea area, TSS
    [INFRASTRUCTURE]    port proximity label + nearest port name
    [REGULATORY_ZONE]   EEZ jurisdiction + MPA membership
    [BEHAVIORAL_PHASE]  phase label + SOG + heading + navigational status
    [ENVIRONMENT]       wind + wave height
    [TEMPORAL]          time of day + day type + season
    [SPATIAL_CONTEXT]   water depth + geohash

Implementation details:
    - All slots are always included
    - Use 'no data' when the underlying enrichment value is absent

Enriched columns from different phases:
    - AIS: [x] SOG, [x] COG, [x] Heading, [ ] ROT, [x] Navigational status, [ ] Ship type,
        [x] Draught, [ ] Name, [ ] Destination
    - Annotation: [ ] annotation, [ ] computed_speed_kn, [ ] computed_heading_deg
    - Spatial: [x] nearest_port_nm, [x] nearest_port_name, [x] port_proximity_label, [ ] in_port_zone,
        [x] in_mpa, [x] mpa_name, [x] in_tss, [x] tss_name, [x] sea_area_name, [x] eez_country_iso, [x] in_territorial_sea
    - Temporal: [x] hour_of_day, [x] day_of_week, [ ] is_weekend, [ ] month,
        [x] season, [x] time_of_day_category
    - Kinematic: [x] behavioral_phase, [x] heading_to_cog_diff_deg
    - Bathymetry: [x] water_depth_m
    - Ocean: [x] wave_height_m, [x] current_speed_ms, [x] current_dir_deg
    - Geohash: [ ] geohash_5, [x] geohash_7
"""
from __future__ import annotations

import json
import math
from typing import Any

_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

_BEARING_LABELS = ['North', 'Northeast', 'East', 'Southeast', 'South', 'Southwest', 'West', 'Northwest']

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
    mode = _safe_get_val(r, 'transport_mode', 'vessel')
    phase = _safe_get_val(r, 'behavioral_phase', 'transiting')
    draught = _safe_get_val(r, 'Draught')
    sog = _safe_get_val(r, 'SOG')

    return f'[AGENT]: {mode}, {phase}, draught {draught}, speed over ground: {sog} kn.'

def _slot_corridor(r: dict) -> str:
    sea = _safe_get_val(r, 'sea_area_name', 'open sea')
    tss = _safe_get_val(r, 'in_tss', False)
    tss_name = _safe_get_val(r, 'tss_name')
    if tss:
        tss_str = f'inside traffic separation scheme (name: {tss_name})'
    else:
        tss_str = 'no traffic separation scheme'
    return f'[CORRIDOR]: {sea}, {tss_str}.'

def _slot_infrastructure(r: dict) -> str:
    label = _safe_get_val(r, 'port_proximity_label', 'offshore')
    name = _safe_get_val(r, 'nearest_port_name')
    nm = _safe_get_val(r, 'nearest_port_nm')
    if nm is not None and name:
        return f'[INFRASTRUCTURE]: {label}, closest to port {name} ({float(nm):.1f} nm).'
    return f'[INFRASTRUCTURE]: {label}, closest to port {name}.'

def _slot_regulatory_zone(r: dict) -> str:
    eez = _safe_get_val(r, 'eez_country_iso', 'high_seas')
    terr = _safe_get_val(r, 'in_territorial_sea', False)
    mpa = _safe_get_val(r, 'in_mpa', False)
    mname = _safe_get_val(r, 'mpa_name')

    eez_str = f'{eez} Exclusive Economic Zone' if eez != 'high_seas' else 'high seas'
    if terr:
        eez_str += ' (territorial sea)'
    if mpa and mname:
        mpa_str = f'marine protected area ({mname})'
    elif mpa:
        mpa_str = f'marine protected area'
    else:
        mpa_str = 'not within a marine protected area'
    return f'[REGULATORY_ZONE]: {eez_str}, {mpa_str}.'

def _slot_behavioral_phase(r: dict) -> str:
    phase = _safe_get_val(r, 'behavioral_phase', 'transiting').replace('_', ' ')
    sog = _safe_get_val(r, 'SOG', None)
    heading = _safe_get_val(r, 'Heading')
    cog = _safe_get_val(r, 'COG')
    nav = _safe_get_val(r, 'Navigational status')
    leeway = _safe_get_val(r, 'heading_to_cog_diff_deg')

    if sog is not None:
        sog_str = f'{float(sog):.1f} kn'
    else:
        sog_str = 'unknown speed'
    head_card = _bearing_to_cardinal(heading if heading is not None else cog)
    nav_label = _nav_status_text(nav)

    try:
        lv = float(leeway)
        leeway_str = f', leeway {lv:+.0f} degrees' if abs(lv) > 10 else ''
    except (TypeError, ValueError):
        leeway_str = ''

    return f'[BEHAVIORAL_PHASE]: {phase}, {sog_str}, heading {head_card}{leeway_str}, navigational status: {nav_label}.'

def _slot_environment(r: dict) -> str:
    wave = _safe_get_val(r, 'wave_height_m', None)
    curr_s = _safe_get_val(r, 'current_speed_ms', None)
    curr_d = _safe_get_val(r, 'current_dir_deg', None)

    if wave is not None:
        wave_str = f'waves {float(wave):.1f} m'
    else:
        wave_str = 'wave height unknown'
    if curr_s is not None and curr_d is not None:
        curr_card = _bearing_to_cardinal(curr_d)
        curr_str = f'current {float(curr_s):.2f} m/s {curr_card}'
    else:
        curr_str = 'current data unavailable'
    return f'[ENVIRONMENT]: {wave_str}, {curr_str}.'

def _slot_temporal(r: dict) -> str:
    dow = _safe_get_val(r, 'day_of_week')
    hour = _safe_get_val(r, 'hour_of_day')
    tod = _safe_get_val(r, 'time_of_day_category', '')
    season = _safe_get_val(r, 'season', '')

    if dow is not None:
        day_str = _DAYS[int(dow)]
    else:
        day_str = 'unknown day'
    if hour is not None:
        hour_str = f'{int(hour):02d}:00'
    else:
        hour_str = 'unknown time'
    return f'[TEMPORAL]: {day_str}, {hour_str}, {tod}, {season}.'

def _slot_spatial_context(r: dict) -> str:
    depth = _safe_get_val(r, 'water_depth_m', None)
    gh7 = _safe_get_val(r, 'geohash_7', '')

    if depth is not None:
        depth_str = f'water depth {float(depth):.0f} m'
    else:
        depth_str = 'water depth unknown'
    parts = [depth_str]
    if gh7:
        parts.append(f'geohash {gh7}')
    return f'[SPATIAL_CONTEXT]: {", ".join(parts)}.'

# ---------- Helper functions ----------

def _safe_get_val(row: dict, key: str, default='unknown'):
    val = row.get(key)
    if val is None or (isinstance(val, float) and val != val) or val == '':
        return default
    return val

def _bearing_to_cardinal(deg) -> str:
    try:
        d = float(deg) % 360
        return _BEARING_LABELS[int((d + 22.5) / 45) % 8]
    except (TypeError, ValueError):
        return 'unknown'
    
def _nav_status_text(nav) -> str | None:
    text = str(nav).strip().lower()
    if text and text not in ('nan', 'none', ''):
        return text
    return None