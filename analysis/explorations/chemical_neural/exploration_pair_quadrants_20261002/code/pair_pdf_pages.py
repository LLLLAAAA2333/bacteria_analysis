"""Draw one sample-pair inspection page on an existing ReportLab canvas.

The caller registers AtlasSans / AtlasBold, creates the PDF, orders the pair
rows, and calls showPage(). This module performs no file I/O and computes no
category summaries. The only feature selection is within the current pair.
"""

import math
import textwrap

import numpy as np
from reportlab.lib.colors import Color, HexColor, white
from reportlab.pdfbase.pdfmetrics import stringWidth


TOP_N = 18
PAGE_WIDTH, PAGE_HEIGHT = 841.89, 595.28
FONT, BOLD = "AtlasSans", "AtlasBold"
BLUE = HexColor("#226CA7")
RED = HexColor("#C64245")
INK = HexColor("#252C33")
GRAY = HexColor("#65717B")
LIGHT = HexColor("#E6E9ED")
MISSING = HexColor("#C17A23")
CATEGORY_LABELS = {
    "Cnear_Nnear": "Chemical near / neural near",
    "Cnear_Nfar": "Chemical near / neural far",
    "Cfar_Nnear": "Chemical far / neural near",
    "Cfar_Nfar": "Chemical far / neural far",
}


def _text(canvas, x, y, value, size=8, color=INK, bold=False, align="left"):
    canvas.setFillColor(color)
    canvas.setFont(BOLD if bold else FONT, size)
    method = {"left": canvas.drawString, "right": canvas.drawRightString,
              "center": canvas.drawCentredString}[align]
    method(x, y, str(value))


def _line(canvas, x1, y1, x2, y2, color=LIGHT, width=0.4):
    canvas.setStrokeColor(color)
    canvas.setLineWidth(width)
    canvas.line(x1, y1, x2, y2)


def _ticks(limits, target=4):
    low, high = map(float, limits)
    rough = (high - low) / target
    exponent = 10 ** math.floor(math.log10(rough))
    fraction = rough / exponent
    step = min((1, 2, 2.5, 5, 10), key=lambda x: abs(x - fraction)) * exponent
    return np.arange(math.ceil(low / step) * step, high + step * 1e-7, step)


def _map(value, limits, start, length):
    return start + (np.asarray(value) - limits[0]) / (limits[1] - limits[0]) * length


def _marker(canvas, x, y, color, filled=True, radius=2.1):
    canvas.setStrokeColor(color)
    canvas.setFillColor(color if filled else white)
    canvas.setLineWidth(0.8)
    canvas.circle(float(x), float(y), radius, stroke=1, fill=1)


def _wrap_width(value, width, size=8, max_lines=2, bold=False):
    """Wrap without dropping text; shrink font only for unusually long names."""
    value = str(value).strip()
    font = BOLD if bold else FONT
    for candidate_size in np.arange(size, 4.4, -0.25):
        lines = []
        line = ""
        for word in value.split():
            candidate = f"{line} {word}".strip()
            if line and stringWidth(candidate, font, candidate_size) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        if len(lines) <= max_lines and all(stringWidth(s, font, candidate_size) <= width for s in lines):
            return lines, float(candidate_size)
    # Full identity is retained even for a long unbroken identifier.
    chars = max(12, int(width / (4.5 * 0.60)))
    return textwrap.wrap(value, chars, break_long_words=True), 4.5


def _neural_panel(canvas, x, y, width, height, a, b, cells, limits, title, ylabel):
    _text(canvas, x, y + height + 14, title, size=9, bold=True)
    for tick in _ticks(limits, target=5):
        ty = float(_map(tick, limits, y, height))
        _line(canvas, x, ty, x + width, ty, color=LIGHT)
        _text(canvas, x - 6, ty - 2.5, f"{tick:g}", size=6.7, color=GRAY, align="right")
    zero = float(_map(0, limits, y, height))
    _line(canvas, x, zero, x + width, zero, color=GRAY, width=0.7)
    spacing = width / len(cells)
    bar_width = min(6.4, spacing * 0.34)
    for i, cell in enumerate(cells):
        center = x + (i + 0.5) * spacing
        for value, color, offset in ((a[i], BLUE, -bar_width - 0.35), (b[i], RED, 0.35)):
            if not np.isfinite(value):
                continue
            end = float(_map(value, limits, y, height))
            canvas.setFillColor(color)
            canvas.rect(center + offset, min(zero, end), bar_width, abs(end - zero), stroke=0, fill=1)
        canvas.saveState()
        canvas.translate(center, y - 6)
        canvas.rotate(55)
        _text(canvas, 0, 0, cell, size=6.6, align="right")
        canvas.restoreState()
    _line(canvas, x, y, x, y + height, color=GRAY, width=0.6)
    canvas.saveState()
    canvas.translate(x - 31, y + height / 2)
    canvas.rotate(90)
    _text(canvas, 0, 0, ylabel, size=7, align="center")
    canvas.restoreState()


def _scatter_panel(canvas, x, y, size, chemical_a, chemical_b, reported_a,
                   reported_b, limits, sample_a, sample_b):
    _text(canvas, x - 45, y + size + 14, f"Chemical profile ({len(chemical_a)} features)", size=9, bold=True)
    for tick in _ticks(limits, target=3):
        px = float(_map(tick, limits, x, size))
        py = float(_map(tick, limits, y, size))
        _line(canvas, px, y, px, y + size, color=LIGHT)
        _line(canvas, x, py, x + size, py, color=LIGHT)
        _text(canvas, px, y - 11, f"{tick:g}", size=6.7, color=GRAY, align="center")
        _text(canvas, x - 6, py - 2.3, f"{tick:g}", size=6.7, color=GRAY, align="right")
    _line(canvas, x, y, x + size, y + size, color=GRAY, width=0.65)
    _line(canvas, x, y, x + size, y, color=GRAY, width=0.6)
    _line(canvas, x, y, x, y + size, color=GRAY, width=0.6)
    xa = _map(chemical_a, limits, x, size)
    yb = _map(chemical_b, limits, y, size)
    finite = np.isfinite(chemical_a) & np.isfinite(chemical_b)
    both = reported_a & reported_b
    # Paths batch points with the same reporting status to keep large PDFs small.
    for mask, color, radius, filled in ((finite & both, BLUE, 1.15, True),
                                        (finite & ~both, MISSING, 1.45, False)):
        path = canvas.beginPath()
        for px, py in zip(xa[mask], yb[mask]):
            path.circle(float(px), float(py), radius)
        canvas.setStrokeColor(color)
        canvas.setFillColor(color)
        canvas.setLineWidth(0.5)
        canvas.drawPath(path, stroke=not filled, fill=filled)
    _text(canvas, x + size / 2, y - 23, f"{sample_a} log2FC", size=7.3, color=BLUE, align="center")
    canvas.saveState()
    canvas.translate(x - 28, y + size / 2)
    canvas.rotate(90)
    _text(canvas, 0, 0, f"{sample_b} log2FC", size=7.3, color=RED, align="center")
    canvas.restoreState()
    _marker(canvas, x - 45, y - 39, BLUE, True, radius=1.8)
    _text(canvas, x - 39, y - 41, "Both reported", size=6.5)
    _marker(canvas, x + 45, y - 39, MISSING, False, radius=1.8)
    _text(canvas, x + 51, y - 41, "Any report missing", size=6.5)


def _species(data, sample):
    taxonomy = data["taxonomy"]
    if sample not in taxonomy.index:
        return "Species unavailable"
    record = taxonomy.loc[sample]
    for field in ("species_clean", "genus_clean"):
        value = record.get(field)
        if value is not None and str(value).strip().lower() not in ("", "nan", "none"):
            return str(value).strip()
    return "Species unavailable"


def draw_pair_page(canvas, row, data, page_number):
    """Draw a landscape-A4 page; the caller retains page breaks and PDF output.

    Arrays are aligned through the chemical/neural DataFrame columns. Reporting
    masks refer to the original raw report, not a detection-limit assertion.
    Global limits are supplied by the caller and retained on every pair page.
    """
    canvas.saveState()
    sample_a, sample_b = str(row["strain_a"]), str(row["strain_b"])
    chemical_frame = data["chemical"]
    neural_frame = data["neural"]
    chemicals = chemical_frame.columns.astype(str).tolist()
    cells = neural_frame.columns.astype(str).tolist()
    ca = chemical_frame.loc[sample_a].to_numpy(float)
    cb = chemical_frame.loc[sample_b].to_numpy(float)
    na = neural_frame.loc[sample_a].to_numpy(float)
    nb = neural_frame.loc[sample_b].to_numpy(float)
    ra = data["reported"].loc[sample_a, chemical_frame.columns].to_numpy(bool)
    rb = data["reported"].loc[sample_b, chemical_frame.columns].to_numpy(bool)
    chemical_limits = tuple(data["chemical_limits"])
    neural_limits = tuple(data["neural_limits"])

    _text(canvas, 30, 571, f"{sample_a}  vs  {sample_b}", size=15, bold=True)
    category = CATEGORY_LABELS.get(str(row["category"]), str(row["category"]))
    _text(canvas, PAGE_WIDTH - 30, 573, category, size=10, align="right")
    for sample, x, color in ((sample_a, 30, BLUE), (sample_b, 433, RED)):
        lines, size = _wrap_width(f"{sample}: {_species(data, sample)}", 375, size=8.8)
        for i, line in enumerate(lines):
            _text(canvas, x, 551 - i * 10.5, line, size=size, color=color)
    chemical_pct = 100 * float(row["chemical_percentile"])
    neural_pct = 100 * float(row["neural_percentile"])
    _text(canvas, 30, 521,
          f"Chemical distance {float(row['chemical']):.3f}  (percentile {chemical_pct:.1f})"
          f"     |     Neural distance {float(row['neural']):.3f}  (percentile {neural_pct:.1f})",
          size=9)
    _line(canvas, 30, 509, PAGE_WIDTH - 30, 509, color=LIGHT, width=0.7)

    _neural_panel(canvas, 61, 351, 212, 130, na, nb, cells, neural_limits,
                   "Signed neural coefficients", "Coefficient (dF/F0)")
    norm_a, norm_b = np.linalg.norm(na), np.linalg.norm(nb)
    unit_a = na / norm_a if norm_a > 0 else np.full_like(na, np.nan)
    unit_b = nb / norm_b if norm_b > 0 else np.full_like(nb, np.nan)
    _neural_panel(canvas, 336, 351, 212, 130, unit_a, unit_b, cells, (-1, 1),
                   "Unit-normalized neural profiles", "Unit coefficient")
    _scatter_panel(canvas, 638, 351, 130, ca, cb, ra, rb, chemical_limits, sample_a, sample_b)

    _text(canvas, 30, 293, f"Largest {min(TOP_N, len(chemicals))} chemical differences for this pair", size=9, bold=True)
    _marker(canvas, 480, 295, BLUE)
    _text(canvas, 487, 292.5, sample_a, size=7)
    _marker(canvas, 544, 295, RED)
    _text(canvas, 551, 292.5, sample_b, size=7)
    _marker(canvas, 611, 295, GRAY, filled=False)
    _text(canvas, 618, 292.5, "Hollow: report missing", size=6.7)
    _text(canvas, 811, 292.5, "abs delta", size=6.5, color=GRAY, align="right")

    delta = np.abs(ca - cb)
    # Stable sorting makes ties reproducible in the existing feature order.
    selected = np.argsort(-np.nan_to_num(delta, nan=-np.inf), kind="stable")[:TOP_N]
    plot_x, plot_width = 345, 430
    first_y, spacing, axis_y = 276.5, 11.5, 72
    top_y = first_y + spacing / 2
    for tick in _ticks(chemical_limits, target=4):
        px = float(_map(tick, chemical_limits, plot_x, plot_width))
        _line(canvas, px, axis_y, px, top_y, color=LIGHT, width=0.4)
        _text(canvas, px, axis_y - 11, f"{tick:g}", size=6.8, color=GRAY, align="center")
    if chemical_limits[0] < 0 < chemical_limits[1]:
        zero = float(_map(0, chemical_limits, plot_x, plot_width))
        _line(canvas, zero, axis_y, zero, top_y, color=HexColor("#B8C0C8"), width=0.6)
    for position, index in enumerate(selected):
        y = first_y - position * spacing
        label = chemicals[index].strip()
        bold = chemicals[index] in data.get('bold_chemicals', set())
        text_lines, label_size = _wrap_width(label, 295, size=6.5, max_lines=2, bold=bold)
        text_y = y + (len(text_lines) - 1) * 3.15 - 2.1
        for i, line in enumerate(text_lines):
            _text(canvas, 30, text_y - i * 6.3, line, size=label_size,
                  color=RED if bold else INK, bold=bold)
        xa = float(_map(ca[index], chemical_limits, plot_x, plot_width))
        xb = float(_map(cb[index], chemical_limits, plot_x, plot_width))
        _line(canvas, xa, y, xb, y, color=HexColor("#A2ABB5"), width=0.75)
        _marker(canvas, xa, y, BLUE, filled=bool(ra[index]), radius=2.2)
        _marker(canvas, xb, y, RED, filled=bool(rb[index]), radius=2.2)
        _text(canvas, 811, y - 2.1, f"{delta[index]:.2f}", size=6.7, color=GRAY, align="right")
    _line(canvas, plot_x, axis_y, plot_x + plot_width, axis_y, color=GRAY, width=0.6)
    _text(canvas, plot_x + plot_width / 2, 46, "Chemical value (log2FC relative to each sample's reference)",
          size=7.2, align="center")
    _text(canvas, 30, 27,
          f"Top {len(selected)} by abs(log2FC A - B), all {len(chemicals)} features. "
          "Red names: recurrent in >=2 categories; red dots: sample B. Hollow: report missing.", size=6.6, color=GRAY)
    coverage = float(row.get("bootstrap_valid_fraction", np.nan))
    coverage_text = f"{100 * coverage:.1f}%" if np.isfinite(coverage) else "unavailable"
    ref_a, ref_b = str(row.get("reference_a", "unavailable")), str(row.get("reference_b", "unavailable"))
    _text(canvas, 30, 14,
          f"References: {sample_a}={ref_a}; {sample_b}={ref_b}     |     "
          f"Valid bootstrap comparisons: {coverage_text}     |     Pair {row['pair_id']}",
          size=6.6, color=GRAY)
    _text(canvas, PAGE_WIDTH - 30, 14, f"Page {page_number}", size=6.6, color=GRAY, align="right")
    canvas.restoreState()
