"""
STAGE 1: Ingest & preprocess raw AIS .csv with DuckDB

Output files:
    _full_ais_parquet       All columns, forward-filled, sorted by (MMSI, ts_unix)
                            Used in Stage 3 to recover AIS fields that the annotation tool strips
    _for_annotation.txt     Space-delimited ASCII: MMSI lon lat ts_unix
                            Sorted by ts_unix in ascending order. To be used
                            as input for the Archimedes' annotation binary
    _vessel_info.csv        Semicolon-delimited: MMSI;TYPE_CODE;TYPE;DESCRIPTION
                            One row per unique MMSI, used by Stage 2 annotation binary
                            to select per-type thresholds

Preprocessing steps (in DuckDB):
    1. Read raw .csv with auto-detected schema
    2. Filter Type_of_mobile to keep 'Class A' and 'Class B'
    3. Parse timestamp to Unix epoch seconds
    4. Deduplicate based in (MMSI, ts_unix)
    5. Forward-fill static fields with LAST_VALUE IGNORE NULLS
    6. Sort by (MMSI, ts_unix)
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import duckdb
import tempfile
import os

logger = logging.getLogger(__name__)

# Static features to be forward-filled
_DEFAULT_FILL_COLS = ['Name', 'Ship type', 'Draught', 'Length', 'Width', 'Destination', 'ETA']

# Mapping ship type to string
_ITU_TO_ARCHIMEDES_TYPE: dict[int, str] = {
    30: 'Fishing',
    31: 'Tug',   # towing
    32: 'Tug',   # towing long/wide
    36: 'Sailing',
    37: 'Pleasure Craft',
    52: 'Tug',
    53: 'Tug',   # port tender
}
for _code in range(40, 50):
    _ITU_TO_ARCHIMEDES_TYPE[_code] = 'High speed craft'
for _code in range(60, 70):
    _ITU_TO_ARCHIMEDES_TYPE[_code] = 'Passenger'
for _code in range(70, 80):
    _ITU_TO_ARCHIMEDES_TYPE[_code] = 'Cargo'
for _code in range(80, 90):
    _ITU_TO_ARCHIMEDES_TYPE[_code] = 'Tanker'

_TEXT_TO_ARCHIMEDES_TYPE: dict[str, str] = {
    'fishing':           'Fishing',
    'cargo':             'Cargo',
    'tanker':            'Tanker',
    'passenger':         'Passenger',
    'tug':               'Tug',
    'towing':            'Tug',
    'towing long/wide':  'Tug',
    'port tender':       'Tug',
    'sailing':           'Sailing',
    'pleasure':          'Pleasure Craft',
    'hsc':               'High speed craft',
    'dredging':          'Default',
    'pilot':             'Default',
    'sar':               'Default',
    'military':          'Default',
    'law enforcement':   'Default',
    'medical':           'Default',
    'diving':            'Default',
    'anti-pollution':    'Default',
    'other':             'Default',
    'undefined':         'Default',
    'reserved':          'Default',
    'spare 1':           'Default',
    'spare 2':           'Default',
    'not party to conflict': 'Default',
}

# AIS navigational-status codes (ITU-R M.1371). NOAA encodes these numerically;
# mapped to DMA-style text at ingest so the text-keyed phase classifier works.
_US_NAV_STATUS: dict[int, str] = {
    0: 'under way using engine',
    1: 'at anchor',
    2: 'not under command',
    3: 'restricted manoeuvrability',
    4: 'constrained by her draught',
    5: 'moored',
    6: 'aground',
    7: 'engaged in fishing',
    8: 'under way sailing',
    9: 'reserved for high speed craft',
    10: 'reserved for wing in ground',
    11: 'power-driven vessel towing astern',
    12: 'power-driven vessel pushing ahead',
    14: 'AIS-SART / MOB / EPIRB',
    15: 'undefined',
}

def _itu_to_type(code) -> str:
    """Map a Ship_type value (code or label) to an Archimedes type string"""
    if code is None:
        return 'Default'
    text_key = str(code).strip().lower()
    if text_key in _TEXT_TO_ARCHIMEDES_TYPE:
        return _TEXT_TO_ARCHIMEDES_TYPE[text_key]
    # Fall back to integer code lookup
    try:
        return _ITU_TO_ARCHIMEDES_TYPE.get(int(code), 'Default')
    except (TypeError, ValueError):
        return 'Default'
    
def run(raw_csv: Path | str, output_dir: Path | str, cfg: dict) -> tuple[Path, Path]:
    """Preprocess one raw AIS .csv file"""
    raw_csv = Path(raw_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = raw_csv.stem
    parquet_out = output_dir / f'{stem}_full_ais.parquet'
    txt_out = output_dir / f'{stem}_for_annotation.txt'
    vessel_info_out = output_dir / f'{stem}_vessel_info.csv'

    ingest_cfg = cfg.get('ingest', {})
    raw_cols = ingest_cfg.get('raw_columns', {})
    mobile_filter = ingest_cfg.get('mobile_type_filter')
    fill_cols = ingest_cfg.get('forward_fill_cols', _DEFAULT_FILL_COLS)
    source = ingest_cfg.get('source', 'dma').lower()
    ts_format = ingest_cfg.get('ts_format', '%Y-%m-%d %H:%M:%S')

    # Map config column names to actual .csv column names
    col_mmsi = raw_cols.get('mmsi', 'MMSI')
    col_ts = raw_cols.get('timestamp', '# Timestamp')
    col_lat = raw_cols.get('lat', 'Latitude')
    col_lon = raw_cols.get('lon', 'Longitude')
    col_mobile = raw_cols.get('mobile_type', 'Type of mobile')

    db_path = output_dir / f'{stem}_ingest.duckdb'
    con = duckdb.connect(str(db_path))
    con.execute("SET memory_limit='15GB'")
    con.execute("SET temp_directory='/tmp/duckdb_tmp/'")
    con.execute('INSTALL spatial; LOAD spatial;')
    
    logger.info('Reading %s', raw_csv)

    # ========== Load raw CSV ==========
    con.execute(f"""
        CREATE OR REPLACE TABLE raw AS
        SELECT * FROM read_csv_auto('{raw_csv}', header=true, sample_size=10000)
    """)

    # ========== Filter by mobile type & valid coordinates ==========
    coord_filter = (
        f'"{col_lat}" BETWEEN -90 AND 90 '
        f'AND "{col_lon}" BETWEEN -180 AND 180 '
        f'AND "{col_lat}" IS NOT NULL '
        f'AND "{col_lon}" IS NOT NULL'
    )
    if mobile_filter:
        values = ', '.join(f"'{v}'" for v in mobile_filter)
        con.execute(f"""
            CREATE OR REPLACE TABLE filtered AS
            SELECT * FROM raw
            WHERE "{col_mobile}" IN ({values})
                AND {coord_filter}
        """)
    else:
        con.execute(f"""
            CREATE OR REPLACE TABLE filtered AS
            SELECT * FROM raw
            WHERE {coord_filter}
        """)
    con.execute('DROP TABLE raw')

    # ========== US-specific: null AIS "not available" sentinels ==========
    # DMA leaves these blank; NOAA encodes the ITU sentinels numerically:
    #   SOG 102.3 (=1023, 0.1 kn) | COG 360.0 (=3600, 0.1 deg) | Heading 511
    if source == 'us':
        avail = {row[0] for row in con.execute('DESCRIBE filtered').fetchall()}
        col_sog = raw_cols.get('sog', 'SOG')
        col_cog = raw_cols.get('cog', 'COG')
        col_head = raw_cols.get('heading', 'Heading')
        if col_sog in avail:
            con.execute(f'UPDATE filtered SET "{col_sog}" = NULL WHERE "{col_sog}" >= 102.3')
        if col_cog in avail:
            con.execute(f'UPDATE filtered SET "{col_cog}" = NULL WHERE "{col_cog}" >= 360.0')
        if col_head in avail:
            con.execute(f'UPDATE filtered SET "{col_head}" = NULL '
                        f'WHERE "{col_head}" = 511 OR "{col_head}" > 359')
        logger.info('US sentinel cleaning: nulled SOG/COG/Heading not-available codes')

    # US-specific: normalize column names to the DMA canonical set
    if source == 'us':
        us_rename = {
            'Status':     'Navigational status',
            'Draft':      'Draught',
            'VesselType': 'Ship type',
            'VesselName': 'Name',
        }
        avail = {row[0] for row in con.execute('DESCRIBE filtered').fetchall()}
        for src_name, canon in us_rename.items():
            if src_name in avail and canon not in avail:
                con.execute(f'ALTER TABLE filtered RENAME COLUMN "{src_name}" TO "{canon}"')
        logger.info('US column normalization: Status/Draft/VesselType/VesselName '
                    '-> DMA canonical names')

        if 'Navigational status' in {row[0] for row in
                                     con.execute('DESCRIBE filtered').fetchall()}:
            nav_case = '\n'.join(f"                    WHEN {code} THEN '{txt}'"
                                 for code, txt in _US_NAV_STATUS.items())
            con.execute(f"""
                CREATE OR REPLACE TABLE filtered AS
                SELECT * EXCLUDE ("Navigational status"),
                    CASE TRY_CAST("Navigational status" AS INTEGER)
{nav_case}
                        ELSE NULL
                    END AS "Navigational status"
                FROM filtered
            """)
            logger.info('US nav-status codes mapped to text')

    # ========== Parse timestamp to Unix epoch seconds ==========
    con.execute(f"""
        CREATE OR REPLACE TABLE with_epoch AS
        SELECT *,
            CASE
                WHEN typeof("{col_ts}") IN ('BIGINT','INTEGER','HUGEINT')
                    THEN CAST("{col_ts}" AS BIGINT)
                WHEN typeof("{col_ts}") LIKE 'TIMESTAMP%'
                    THEN CAST(epoch("{col_ts}") AS BIGINT)
                ELSE
                    CAST(epoch(strptime(CAST("{col_ts}" AS VARCHAR),
                        '{ts_format}')) AS BIGINT)
            END AS ts_unix
        FROM filtered
    """)
    con.execute('DROP TABLE filtered')

    # ========== Deduplicate based on (MMSI, ts_unix) ==========
    con.execute(f"""
        CREATE OR REPLACE TABLE deduped AS
        SELECT * FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY "{col_mmsi}", ts_unix
                    ORDER BY ts_unix
                ) AS _rn
            FROM with_epoch
        )
        WHERE _rn = 1
    """)
    con.execute('DROP TABLE with_epoch')

    # ========== Forward-fill static fields ==========
    available = {row[0] for row in con.execute('DESCRIBE deduped').fetchall()}
    fill_exprs = []
    for col in fill_cols:
        if col in available:
            fill_exprs.append(
                f'LAST_VALUE("{col}" IGNORE NULLS) OVER '
                f'(PARTITION BY "{col_mmsi}" ORDER BY ts_unix '
                f'ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS "{col}"'
            )

    if fill_exprs:
        other_cols = [f'"{c}"' for c in available if c not in fill_cols and c != '_rn']
        select_list = ', '.join(other_cols + fill_exprs)
        con.execute(f"""
            CREATE OR REPLACE TABLE filled AS
            SELECT {select_list}
            FROM deduped
        """)
    else:
        con.execute("""
            CREATE OR REPLACE TABLE filled AS
            SELECT * EXCLUDE (_rn) FROM deduped
        """)
    con.execute('DROP TABLE deduped')

    # ========== Sort by (MMSI, ts_unix) ==========
    con.execute(f"""
        CREATE OR REPLACE TABLE sorted AS
        SELECT * FROM filled
        ORDER BY "{col_mmsi}", ts_unix
    """)
    con.execute('DROP TABLE filled')

    # ========== Output full AIS parquet ==========
    logger.info('Writing full AIS .parquet file: %s', parquet_out)
    con.execute(f"COPY sorted TO '{parquet_out}' (FORMAT PARQUET)")

    # ========== Output space-delimited ASCII for annotation ==========
    # Format: MMSI lon lat ts_unix (no header)
    logger.info('Writing annotation input: %s', txt_out)
    con.execute(f"""
        COPY (
            SELECT
                CAST("{col_mmsi}" AS BIGINT),
                CAST("{col_lon}" AS DOUBLE),
                CAST("{col_lat}" AS DOUBLE),
                CAST(ts_unix AS BIGINT)
            FROM sorted
        ) TO '{txt_out}' (DELIMITER ' ', HEADER FALSE)
    """)

    # ========== Output semicolon-delimited vessel info .csv file ==========
    # Format: MMSI;TYPE_CODE;TYPE;DESCRIPTION
    col_ship_type = raw_cols.get('ship_type', 'Ship type')
    ship_type_available = col_ship_type in {
        row[0] for row in con.execute('DESCRIBE sorted').fetchall()
    }

    if ship_type_available:
        rows = con.execute(f"""
            SELECT DISTINCT
                CAST("{col_mmsi}" AS BIGINT) AS mmsi,
                "{col_ship_type}" AS ship_type
            FROM sorted
            ORDER BY mmsi
        """).fetchall()
    else:
        rows = con.execute(f"""
        SELECT DISTINCT CAST("{col_mmsi}" AS BIGINT) AS mmsi
        FROM sorted
        ORDER BY mmsi
        """).fetchall()
        rows = [(r[0], None) for r in rows]

    logger.info('Writing vessel info .csv: %s', vessel_info_out)
    with open(vessel_info_out, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['MMSI', 'TYPE_CODE', 'TYPE', 'DESCRIPTION'])
        for mmsi, ship_type_raw in rows:
            arch_type = _itu_to_type(ship_type_raw)
            writer.writerow([mmsi, ship_type_raw or '', arch_type, arch_type])
    
    n_rows = con.execute('SELECT COUNT(*) FROM sorted').fetchone()[0]
    n_vessels = len(rows)
    logger.info('Ingest done: %d points, %d vessels -> %s', n_rows, n_vessels, output_dir)

    con.close()
    db_path.unlink(missing_ok=True)
    return parquet_out, txt_out, vessel_info_out