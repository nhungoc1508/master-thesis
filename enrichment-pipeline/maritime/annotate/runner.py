"""
STAGE 2: Run the Archimedes C++ annotation binary as a subprocess

Binary call signature:
    ./annotate [input_txt] [id_col] [ts_col] [params_json] [vessel_info_csv] [output_csv] [annotated_only]

Output: space-delimited .csv with header row:
    id lon lat t speed heading annotation
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

def run(input_txt: Path | str, vessel_info_csv: Path | str,
        output_dir: Path | str, cfg: dict) -> Path:
    """
    Annotate one sorted ASCII file produced by Stage 1
    Returns path to the output .csv written by the binary
    """
    input_txt = Path(input_txt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f'{input_txt.stem.replace("_for_annitation", "")}_annotated.csv'
    
    arch_cfg = cfg['archimedes']
    anno_cfg = cfg.get('annotate', {})
    binary = Path(arch_cfg['annotation_binary'])
    params_json = Path(arch_cfg['params_json'])
    vessel_info = Path(vessel_info_csv)
    id_col = str(anno_cfg.get('id_col', 1))
    ts_col = str(anno_cfg.get('ts_col', 4))
    anno_only = 'true' if anno_cfg.get('annotated_only', False) else 'false'

    cmd = [
        str(binary),
        str(input_txt),
        id_col,
        ts_col,
        str(params_json),
        str(vessel_info),
        str(output_csv),
        anno_only
    ]
    logger.info('Running: %s', ' '.join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.debug('stdout: %s', result.stdout.strip())
    if result.stderr:
        logger.debug('stderr: %s', result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f'Annotation binary exited {result.returncode}:\n{result.stderr}')
    
    n_lines = sum(1 for _ in open(output_csv)) - 1
    logger.info('Annotation done: %d annotated points -> %s', n_lines, output_csv)
    return output_csv