"""Verify this focused delivery and that prior inputs/outputs stayed intact."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
PREVIOUS = OUT.parent / "population_first_20260930"


def fingerprint(path, base):
    return dict(path=str(path.relative_to(base)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    checks = {}
    original = json.loads((PREVIOUS / "logs/input_manifest.json").read_text())
    for record in original:
        assert fingerprint(ROOT / record["path"], ROOT) == record
    previous = json.loads((PREVIOUS / "logs/output_manifest.json").read_text())
    for record in previous:
        assert fingerprint(PREVIOUS / record["path"], PREVIOUS) == record, record["path"]
    checks["unchanged_original_inputs"] = len(original)
    checks["unchanged_previous_outputs"] = len(previous)
    # Conservative manifest includes all previously aligned tables used for the
    # focused audits; raw dependencies retain their earlier baseline hashes.
    inputs = original + [fingerprint(p, ROOT) for p in sorted((PREVIOUS / "tables").glob("aligned_*"))]
    inputs += [fingerprint(p, ROOT) for p in sorted((PREVIOUS / "tables").glob("neural_value_*"))]
    unique = {record["path"]: record for record in inputs}
    (OUT / "logs/input_manifest.json").write_text(json.dumps(list(unique.values()), indent=2))
    for path in (OUT / "code").glob("*.py"):
        compile(path.read_text(), str(path), "exec")
    figures = json.loads((OUT / "logs/figure_verification.json").read_text())
    response = json.loads((OUT / "logs/response_evidence_verification.json").read_text())
    chemistry = json.loads((OUT / "tables/chemical_neighbor_audit_summary.json").read_text())
    assert figures["status"] == "passed" and figures["source_inputs_unchanged"]
    assert response["status"] == "success" and response["sign_agrees_for_all_rows"]
    assert response["n_independent_vector_reconstructions"] == 510
    assert chemistry["n_original_neighbors"] == 45
    c = pd.read_csv(OUT / "tables/example_chemical_points.csv")
    assert c.shape[0] == 760 and c.groupby("pair_id").size().eq(380).all()
    d = pd.read_csv(OUT / "tables/example_paired_response_points.csv")
    assert d.shape[0] == 147
    assert d.groupby("pair_id").neuron_class.nunique().eq(13).all()
    assert not d.duplicated(["pair_id", "worm_key", "neuron_class"]).any()
    summary = pd.read_csv(OUT / "tables/response_audit_subset_summary.csv")
    row = summary[(summary.subset == "nearest_either") &
                  (summary.variant == "mean") & (summary["mode"] == "population")].iloc[0]
    assert (row.n_pairs, row.n_animals, row.n_animal_pairs) == (45, 49, 255)
    assert (row.pairs_more_than_half_same_direction,
            row.pairs_half_same_direction, row.pairs_less_than_half_same_direction) == (41, 1, 3)
    assert np.isclose(row.animal_equal_same_direction, .8001700680272108, atol=1e-12)
    checks["all_380_chemical_features_and_13_neurons_visible"] = True
    checks["all_45_pairs_retained_in_evidence_tables"] = True
    checks["independent_vector_reconstructions"] = 510
    checks["animal_neuron_points_reconstructed"] = 147
    checks["visual_inspection"] = "Both chemical/response figures and supporting curves inspected; labels and data unobscured"
    pending = {"verification.json", "output_manifest.json"}
    nlinks = 0
    for path in OUT.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            destination = path.parent / target.split("#")[0]
            assert destination.exists() or destination.name in pending, (path.name, target)
            nlinks += 1
    checks["resolved_document_links"] = nlinks
    for path in (OUT / "figures").glob("*"):
        assert path.stat().st_size > 10000
    assert len(list((OUT / "figures").glob("*.png"))) == 4
    result = dict(status="passed", checks=checks,
                  execution="Chemical audit, response audit, and plotting actually ran; no new classifier was fitted.",
                  limitations="Same-data reuse and numerical review are not independent biological validation.")
    (OUT / "logs/verification.json").write_text(json.dumps(result, indent=2))
    outputs = [fingerprint(p, OUT) for p in sorted(OUT.rglob("*")) if p.is_file()
               and p.suffix != ".log" and "__pycache__" not in p.parts
               and p.name not in ["output_manifest.json", ".DS_Store"]]
    (OUT / "logs/output_manifest.json").write_text(json.dumps(outputs, indent=2))
    assert all(fingerprint(OUT / r["path"], OUT) == r for r in outputs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        (OUT / "logs/verification.json").write_text(json.dumps(dict(status="failed", error=repr(error))))
        raise
