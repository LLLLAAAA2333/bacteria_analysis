"""Shared visual settings and exports, with no import-time global mutations."""
from pathlib import Path
import matplotlib.pyplot as plt
from .paths import find_repo_root


def apply_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
        'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7,
        'figure.dpi': 110, 'savefig.dpi': 300,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'savefig.facecolor': 'white',
    })


def save_figure(fig, stem, formats=('png', 'svg', 'pdf'), dpi=300):
    """Export a live Figure only under poster/figures; keep it open for display."""
    stem = Path(stem).resolve()
    allowed = (find_repo_root() / 'poster/figures').resolve()
    if not stem.is_relative_to(allowed):
        raise ValueError(f'Poster exports must stay under {allowed}')
    if any(fmt not in {'png', 'svg', 'pdf'} for fmt in formats):
        raise ValueError('Supported export formats: png, svg, pdf')
    stem.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for fmt in formats:
        target = stem.with_suffix('.' + fmt)
        fig.savefig(target, dpi=dpi, bbox_inches='tight', facecolor='white')
        files.append(target)
    return files
