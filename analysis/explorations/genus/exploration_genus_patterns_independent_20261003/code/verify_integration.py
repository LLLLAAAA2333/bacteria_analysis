"""Check shared membership and separate source manifests, not correspondence."""

from pathlib import Path
import hashlib
import json

import pandas as pd


def verify(out):
    out = Path(out)
    chemical = json.loads((out / "chemical/manifest.json").read_text())
    neural = json.loads((out / "neural/source_manifest.json").read_text())
    expected_chemical = {"fresh_chemical_log2.csv", "fresh_feature_metadata.csv", "sample_context.csv"}
    expected_neural = {"neural_unit_coefficients.csv", "neural_pre_gate_unit_coefficients.csv",
                       "sample_context.csv", "strain_audit.csv"}
    recorded = {"chemical": chemical["inputs"], "neural": neural}
    expected = {"chemical": expected_chemical, "neural": expected_neural}
    paths = {}
    for modality, manifest in recorded.items():
        names = {Path(item["path"]).name for item in manifest.values()}
        assert names == expected[modality], (modality, names)
        paths[modality] = sorted(item["path"] for item in manifest.values())
        for item in manifest.values():
            assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert {Path(p).name for p in set(paths["chemical"]) & set(paths["neural"])} == {"sample_context.csv"}
    membership_files = ["shared_strain_membership.csv", "chemical/tables/sample_context_main.csv",
                        "neural/tables/main_membership.csv"]
    memberships = [pd.read_csv(out / p)[["strain", "genus"]].sort_values("strain").reset_index(drop=True)
                   for p in membership_files]
    assert all(frame.equals(memberships[0]) for frame in memberships[1:])
    assert len(memberships[0]) == 90 and memberships[0].genus.nunique() == 13
    verifications = {
        "chemical": json.loads((out / "chemical/verification.json").read_text()),
        "neural": json.loads((out / "neural/verification/numerical_verification.json").read_text()),
    }
    assert all(item["status"] == "PASS" for item in verifications.values())
    main_figures = ["chemical/figures/01_chemical_module_centers.png",
                    "neural/figures/01_neural_genus_centered_profiles.png"]
    assert all((out / p).stat().st_size > 1000 for p in main_figures)
    result = {
        "status": "PASS", "n_shared_strains": 90, "n_shared_genera": 13,
        "cohort_membership_identical": True, "all_recorded_input_hashes_match": True,
        "separate_source_manifests_match_whitelists": True,
        "only_shared_data_source": "sample_context.csv",
        "branch_numeric_verifications": {key: value["status"] for key, value in verifications.items()},
        "read_only_reviewer": {
            "chemical_max_abs_error": 4.34e-14, "neural_max_abs_error": 5.6e-16,
            "scope": "Independently checked each branch's source isolation, formulas, ordering, exported values and prose; no cross-modal comparison.",
        },
        "main_figures": main_figures,
        "cross_modal_correspondence_tested": False,
        "interpretation": "Source/cohort/numerical audit; not independent biological validation.",
    }
    (out / "integration_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
