"""
Kinematic feature enrichment

Columns added:
    behavioral_phase            str         at_berth | anchored | maneuvering | fishing |
                                            slow_transit | transiting | communication_gap
    heading_to_cog_diff_deg     float       heading - COG in [-180, 180]

Phase classification combines:
    - AIS 'Navigational status'
    - C++ annotation labels
    - SOG kinematics
"""
from __future__ import annotations

import pandas as pd

_NAV_TEXT_TO_PHASE: dict[str, str] = {
    'moored':                               'at_berth',
    'at anchor':                            'anchored',
    'aground':                              'anchored',
    'engaged in fishing':                   'fishing',
    'not under command':                    'maneuvering',
    'restricted maneuverability':           'maneuvering',
    'restricted manoeuvrability':           'maneuvering',
    'restricted manoeuverability':          'maneuvering',
    'constrained by her draught':           'maneuvering',
    'power-driven vessel towing astern':    'maneuvering',
    'power-driven vessel pushing ahead or towing alongside': 'maneuvering',
    'ais-sart is active':                   'maneuvering',
    'ais-sart active':                      'maneuvering',
}

def _classify_phase(row: pd.Series, stop_kn: float, slow_kn: float) -> str:
    sog = row.get('SOG')
    nav = str(row.get('Navigational status', '')).strip().lower()
    anno = str(row.get('annotation', ''))
    labels = set(anno.split(';')) if anno.strip() else set()

    if 'GAP_START' in labels or 'GAP_END' in labels:
        return 'communication_gap'
    
    try:
        sog_val = float(sog)
    except (TypeError, ValueError):
        sog_val = None

    if 'STOP_START' in labels:
        return 'at_berth' if (sog_val is not None and sog_val < stop_kn) else 'anchored'

    if nav in _NAV_TEXT_TO_PHASE:
        return _NAV_TEXT_TO_PHASE[nav]
    
    # SOG-based fallbaccks for underway/unknown status
    if sog_val is None:
        return 'transiting'
    if sog_val < stop_kn:
        return 'anchored'
    if sog_val < slow_kn:
        return 'slow_transit'
    return 'transiting'

def _angle_diff(a, b) -> float | None:
    """Signed difference a - b wrapped to [-180, 180]"""
    try:
        diff = (float(a) - float(b) + 180) % 360 - 180
        return round(diff, 2)
    except (TypeError, ValueError):
        return None
    
def enrich(df: pd.DataFrame, stop_speed_kn: float = 0.5, slow_speed_kn: float = 3.0) -> pd.DataFrame:
    df = df.copy()

    df['behavioral_phase'] = df.apply(_classify_phase, axis=1, stop_kn=stop_speed_kn, slow_kn=slow_speed_kn)

    # Heading - COG: proxy for leeway/current set
    df['heading_to_cog_diff_deg'] = df.apply(lambda r: _angle_diff(r.get('Heading'), r.get('COG')), axis=1)

    return df