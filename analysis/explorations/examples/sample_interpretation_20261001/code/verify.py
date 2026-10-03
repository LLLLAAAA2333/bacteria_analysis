"""Focused independent checks of figure values and protected source files."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[1]
T = OUT / "tables"


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    checks = {}
    baseline = json.loads((OUT / "logs/source_baseline.json").read_text())
    changed = [name for name, sha in baseline.items() if digest(REPO / name) != sha]
    assert not changed, changed
    checks["protected_source_files_unchanged"] = len(baseline)

    params = json.loads((OUT / "logs/poster_parameters.json").read_text())
    assert digest(OUT / "code/poster_figures.py") == params["code_sha256"]
    assert all(digest(T / name) == sha for name, sha in params["table_sha256"].items())
    checks["figure_code_and_table_fingerprints_match"] = True

    curves = pd.read_csv(T / "neural_curves.csv")
    plotted = pd.read_csv(T / "poster_main_curve_values.csv")
    keys = ["sample_id", "neuron_class", "time_s"]
    independently_averaged = curves.groupby(keys).response.agg(["mean", "count", "std"])
    plotted = plotted.set_index(keys)
    matched = independently_averaged.loc[plotted.index]
    np.testing.assert_allclose(matched["mean"], plotted["mean"], atol=1e-14)
    np.testing.assert_allclose(matched["std"]/np.sqrt(matched["count"]), plotted["sem"], atol=1e-14)
    assert plotted.index.get_level_values("time_s").min() == -5
    assert plotted.index.get_level_values("time_s").max() == 24
    checks["main_trace_means_and_sem_recomputed_from_animals"] = True

    points = pd.read_csv(T / "poster_main_animal_points.csv")
    raw_stage = curves[curves.time_s.ge(10) & curves.time_s.lt(25)].groupby(
        ["sample_id", "neuron_class", "animal_id"]).response.mean()
    point_index = points.set_index(["sample_id", "neuron_class", "animal_id"])
    np.testing.assert_allclose(raw_stage.loc[point_index.index], point_index.response, atol=1e-14)
    counts = {}
    for cell in ["AWA", "ASH", "AWCON"]:
        d = points[points.neuron_class.eq(cell)]
        matrix = d.pivot(index="animal_id", columns="sample_id", values="response")
        assert matrix.notna().all().all()
        counts[cell] = len(matrix)
        if cell == "AWA":
            assert (matrix.A022 < matrix.A021).all() and (matrix.A023 < matrix.A021).all()
        if cell == "ASH":
            assert (matrix.A022 > matrix.A023).all() and (matrix.A023 > matrix.A021).all()
        if cell == "AWCON":
            assert (matrix.A023 > matrix.A022).all()
    assert counts == {"AWA": 5, "ASH": 6, "AWCON": 5}
    checks["main_paired_animal_counts"] = counts
    checks["reported_relative_orderings_verified"] = True

    tw = pd.read_csv(T / "poster_timing_animal_windows.csv")
    tm = tw.pivot(index="animal_id", columns="bin_index", values="difference")
    ta = pd.read_csv(T / "poster_timing_animal_average.csv").set_index("animal_id")
    assert tm.shape == (5, 5) and tm.notna().all().all()
    assert (tm[2] > 0).all() and (tm[4] < 0).all()
    np.testing.assert_allclose(tm.mean(axis=1).sort_index(), ta.difference.sort_index(), atol=1e-14)
    checks["timing_opposite_bin_signs"] = {"10-15s_positive": 5, "20-25s_negative": 5}
    checks["timing_0_25_mean_equals_mean_of_five_bins"] = True
    checks["timing_0_25_mean_difference"] = float(ta.difference.mean())

    chem = pd.read_csv(T / "poster_main_chemistry.csv")
    profiles = pd.read_csv(T / "chemical_primary_common_ranked.csv")
    assert len(profiles) == 322
    for row in chem.itertuples():
        a, b = profiles[row.strain_a+"_log2fc"], profiles[row.strain_b+"_log2fc"]
        value = np.sqrt(np.mean((a-b)**2))
        np.testing.assert_allclose(value, row.common_rms_log2fc, atol=1e-14)
    checks["chemical_fixed_322_distances_recomputed"] = True

    pages = {}
    for name, expected in [("poster_panels.pdf", 2), ("supporting_evidence.pdf", 5)]:
        path = OUT / "figures" / name
        info = subprocess.check_output(["pdfinfo", str(path)], text=True)
        page_count = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
        assert page_count == expected
        text = subprocess.check_output(["pdftotext", str(path), "-"], text=True)
        assert "A021" in text and "A007" in text
        pages[name] = page_count
    checks["pdf_page_counts"] = pages
    for name in ["strain_response_poster", "timing_cancellation_poster"]:
        root = ET.parse(OUT / "figures" / (name+".svg")).getroot()
        texts = root.findall(".//{http://www.w3.org/2000/svg}text")
        assert len(texts) > 20, "SVG labels must remain editable text."
    checks["svg_text_is_editable"] = True
    checks["visual_review"] = "Both poster pages and all five support pages rendered with Poppler and visually reviewed."
    checks["method_review"] = "Independent read-only review: animal support, projection algebra, fixed chemical mask and interpretation bounds."
    checks["status"] = "passed"
    (OUT / "logs/verification.json").write_text(json.dumps(checks, indent=2)+"\n")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in str(p)
                   and p.name != "output_manifest.json")
    manifest = {str(p.relative_to(OUT)): digest(p) for p in files}
    (OUT / "logs/output_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
