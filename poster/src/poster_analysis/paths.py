"""Resolve local report caches through the exploration catalogue."""
from pathlib import Path
import csv

REPORTS = {
    'responses': 'exploration_response_profiles_individual_snr_20261002',
    'chemistry': 'exploration_chemical_pattern_direct_report_20261003',
    'genus_distances': 'exploration_genus_within_between_20261003',
    'genus_patterns': 'exploration_genus_patterns_independent_20261003',
    'fixed_contrast': 'exploration_bacteroides_adf_ash_chemical_20261003',
    'local_poster': 'poster_local_chemical_neural_20261003',
    'local_pc1': 'exploration_bacteroides_local_model_20261003',
    'local_reliability': 'exploration_bacteroides_neural_reliability_20261003',
    'response_structure': 'response_structure_20260930',
    'pair_example': 'sample_comparison_draft_20261001',
}


def find_repo_root(start=None):
    """Resolve from a notebook working directory, or from this installed source."""
    starts = [Path(start).resolve()] if start is not None else [Path.cwd().resolve(), Path(__file__).resolve().parent]
    for origin in starts:
        for candidate in (origin, *origin.parents):
            if (candidate / 'pixi.toml').is_file() and (candidate / 'reports').is_dir() and (candidate / 'poster/src/poster_analysis').is_dir():
                return candidate
    raise FileNotFoundError('Open this notebook inside the bacteria_analysis repository containing poster/ and reports/.')


def report_path(name, root=None):
    """Resolve a poster alias or report name to its local, grouped result folder."""
    root = find_repo_root(root)
    with (root / 'analysis/explorations/catalogue.csv').open(newline='') as stream:
        reports = {row['report_name']: row['report_dir'] for row in csv.DictReader(stream)}
    report_name = REPORTS.get(name, name)
    if report_name not in reports:
        raise KeyError(f'Unknown source {name!r}; aliases: {list(REPORTS)}')
    result = root / reports[report_name]
    if not result.is_dir():
        raise FileNotFoundError(f'Missing local scientific inputs: {result}. See poster/docs/local_inputs.csv; generated data is not stored in Git.')
    return result


def prepared_response_path(root=None):
    """Require the explicit output of notebook 00; never fall back to old results."""
    import hashlib
    import json

    root = find_repo_root(root)
    result = root / 'poster/data/prepared/responses'
    manifest_path = result / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Run poster/notebooks/00_preparation.ipynb first to prepare response inputs.')
    manifest = json.loads(manifest_path.read_text())
    for relative, digest in manifest['output_sha256'].items():
        path = result / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Prepared response input is missing or changed: {relative}. Rerun notebook 00.')
    return result


def figure_dir(kind='main', root=None):
    if kind not in {'main', 'supporting'}:
        raise ValueError("kind must be 'main' or 'supporting'")
    result = find_repo_root(root) / 'poster/figures' / kind
    result.mkdir(parents=True, exist_ok=True)
    return result


def table_dir(root=None):
    result = find_repo_root(root) / 'poster/tables'
    result.mkdir(parents=True, exist_ok=True)
    return result
