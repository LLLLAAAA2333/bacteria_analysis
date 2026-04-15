import importlib.util
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest
from openpyxl import Workbook

import bacteria_analysis.model_space_seed as model_space_seed_module
from bacteria_analysis.model_space import (
    build_stimulus_sample_map as shared_build_stimulus_sample_map,
    resolve_model_inputs,
)
from bacteria_analysis.model_space_seed import (
    IDENTITY_EVIDENCE_COLUMNS,
    TAXONOMY_ENRICHMENT_COLUMNS,
    build_metabolite_annotation_from_evidence,
    build_model_membership_from_rules,
    build_model_space,
    build_normalized_header_table,
    load_taxonomy_enrichment_cache,
    merge_identity_and_taxonomy_evidence,
    normalize_metabolite_header,
    read_raw_metabolite_headers,
    resolve_identity_from_cached_candidates,
)


def _load_builder_cli_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_model_space.py"
    spec = importlib.util.spec_from_file_location("build_model_space_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


BUILD_MODEL_SPACE_CLI = _load_builder_cli_module()


@pytest.fixture
def fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "model_space_seed"


def _write_matrix(path: Path, *, headers: list[str], rows: list[list[object]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["sample_id", *headers])
    for row in rows:
        sheet.append(row)
    workbook.save(path)


def _write_registry(path: Path) -> None:
    pd.DataFrame.from_records(
        [
            {
                "model_id": "bile_acid",
                "model_label": "Bile Acid",
                "model_tier": "primary",
                "model_status": "draft",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "Bile acid family",
                "authority": "user",
                "notes": "",
            }
        ]
    ).to_csv(path, index=False)


def _write_tiny_builder_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    matrix_path = tmp_path / "matrix.xlsx"
    preprocess_root = tmp_path / "preprocess"
    trial_level_dir = preprocess_root / "trial_level"
    registry_path = tmp_path / "model_registry.csv"
    trial_level_dir.mkdir(parents=True, exist_ok=True)

    _write_matrix(
        matrix_path,
        headers=["Cholic acid (CA)", "Mystery metabolite"],
        rows=[
            ["A226", 1.0, 0.5],
            ["A228", 2.0, 0.0],
        ],
    )
    pd.DataFrame.from_records(
        [
            {"stimulus": "b34_0", "stim_name": "A226 stationary"},
            {"stimulus": "b35_0", "stim_name": "A228 stationary"},
            {"stimulus": "b34_0", "stim_name": "A226 stationary"},
        ]
    ).to_parquet(trial_level_dir / "trial_metadata.parquet", index=False)
    _write_registry(registry_path)
    return matrix_path, preprocess_root, registry_path


def test_read_raw_metabolite_headers_reads_first_row_only(tmp_path):
    matrix_path = tmp_path / "matrix.xlsx"
    _write_matrix(
        matrix_path,
        headers=[" Cholic acid (CA) ", "Beta-Muricholic acid (β-MCA)"],
        rows=[["A001", 1.0, 2.0]],
    )

    assert read_raw_metabolite_headers(matrix_path) == [" Cholic acid (CA) ", "Beta-Muricholic acid (β-MCA)"]


def test_normalize_metabolite_header_preserves_original_and_extracts_alias():
    record = normalize_metabolite_header(" Beta-Muricholic acid (β-MCA) ")

    assert record["original_header"] == " Beta-Muricholic acid (β-MCA) "
    assert record["normalized_name"] == "Beta-Muricholic acid (beta-MCA)"
    assert record["stage3_metabolite_name"] == "Beta-Muricholic acid (beta-MCA)"
    assert record["alias_in_parentheses"] == "beta-MCA"
    assert record["query_candidates"] == [
        "Beta-Muricholic acid (beta-MCA)",
        "Beta-Muricholic acid",
        "beta-MCA",
    ]


def test_build_normalized_header_table_rejects_duplicate_normalized_names():
    with pytest.raises(ValueError, match="normalized metabolite names must be unique"):
        build_normalized_header_table(
            ["Beta-Muricholic acid (β-MCA)", "Beta-Muricholic acid (beta-MCA)"]
        )


def test_build_normalized_header_table_rejects_blank_headers():
    with pytest.raises(ValueError, match="metabolite headers must be non-empty"):
        build_normalized_header_table(["", "feature_2"])


def test_normalize_metabolite_header_normalizes_prime_characters():
    record = normalize_metabolite_header("Riboflavin-5′-monophosphate")

    assert record["normalized_name"] == "Riboflavin-5'-monophosphate"


def test_build_stimulus_sample_map_extracts_sample_id_from_stationary_labels():
    assert model_space_seed_module.build_stimulus_sample_map is shared_build_stimulus_sample_map

    metadata = pd.DataFrame.from_records(
        [
            {"stimulus": "b34_0", "stim_name": "A226 stationary"},
            {"stimulus": "b35_0", "stim_name": "A228 stationary"},
            {"stimulus": "b34_0", "stim_name": "A226 stationary"},
        ]
    )

    mapping = model_space_seed_module.build_stimulus_sample_map(
        metadata,
        matrix_sample_ids=pd.Index(["A226", "A228"]),
    )

    assert mapping.to_dict(orient="records") == [
        {"stimulus": "b34_0", "stim_name": "A226 stationary", "sample_id": "A226"},
        {"stimulus": "b35_0", "stim_name": "A228 stationary", "sample_id": "A228"},
    ]


def test_build_stimulus_sample_map_rejects_conflicting_stim_name():
    metadata = pd.DataFrame.from_records(
        [
            {"stimulus": "b34_0", "stim_name": "A226 stationary"},
            {"stimulus": "b34_0", "stim_name": "A227 stationary"},
        ]
    )

    with pytest.raises(ValueError, match="exactly one stim_name"):
        model_space_seed_module.build_stimulus_sample_map(metadata, matrix_sample_ids=pd.Index(["A226", "A227"]))


def test_build_stimulus_sample_map_rejects_blank_or_missing_values():
    metadata = pd.DataFrame.from_records(
        [
            {"stimulus": None, "stim_name": "A226 stationary"},
            {"stimulus": "b35_0", "stim_name": ""},
        ]
    )

    with pytest.raises(ValueError, match="must be non-empty"):
        model_space_seed_module.build_stimulus_sample_map(
            metadata,
            matrix_sample_ids=pd.Index(["A226", "A228"]),
        )


def test_build_stimulus_sample_map_rejects_duplicate_derived_sample_ids():
    metadata = pd.DataFrame.from_records(
        [
            {"stimulus": "b34_0", "stim_name": "A226 stationary"},
            {"stimulus": "b35_0", "stim_name": "A226 stationary"},
        ]
    )

    with pytest.raises(ValueError, match="must be unique"):
        model_space_seed_module.build_stimulus_sample_map(metadata, matrix_sample_ids=pd.Index(["A226"]))


def test_build_stimulus_sample_map_rejects_sample_ids_absent_from_matrix():
    metadata = pd.DataFrame.from_records([{"stimulus": "b34_0", "stim_name": "A999 stationary"}])

    with pytest.raises(ValueError, match="must exist in the matrix"):
        model_space_seed_module.build_stimulus_sample_map(metadata, matrix_sample_ids=pd.Index(["A226"]))


def test_resolve_identity_marks_exact_pubchem_match_high_confidence(fixture_path):
    record = {
        "original_header": "Cholic acid (CA)",
        "normalized_name": "Cholic acid (CA)",
        "stage3_metabolite_name": "Cholic acid (CA)",
        "alias_in_parentheses": "CA",
        "query_candidates": ["Cholic acid (CA)", "Cholic acid", "CA"],
    }
    payload = json.loads((fixture_path / "pubchem_cholic_acid.json").read_text(encoding="utf-8"))

    result = resolve_identity_from_cached_candidates(record, cached_payloads=[payload])

    assert result["resolution_status"] == "resolved_high_confidence"
    assert result["pubchem_cid"] == "221493"
    assert result["inchikey"]
    assert "exact" in result["match_reason"]


def test_resolve_identity_marks_multi_hit_for_equal_candidates(fixture_path):
    record = {
        "original_header": "Muricholic acid",
        "normalized_name": "Muricholic acid",
        "stage3_metabolite_name": "Muricholic acid",
        "alias_in_parentheses": "",
        "query_candidates": ["Muricholic acid"],
    }
    payload = json.loads((fixture_path / "pubchem_multi_hit.json").read_text(encoding="utf-8"))

    result = resolve_identity_from_cached_candidates(record, cached_payloads=[payload])

    assert result["resolution_status"] == "multi_hit_needs_review"
    assert result["match_reason"] == "multiple_equal_candidates"


def test_resolve_identity_marks_low_confidence_for_loose_match(fixture_path):
    record = {
        "original_header": "Cholic acid derivative",
        "normalized_name": "Cholic acid derivative",
        "stage3_metabolite_name": "Cholic acid derivative",
        "alias_in_parentheses": "",
        "query_candidates": ["Cholic acid derivative"],
    }
    payload = json.loads((fixture_path / "pubchem_cholic_acid.json").read_text(encoding="utf-8"))

    result = resolve_identity_from_cached_candidates(record, cached_payloads=[payload])

    assert result["resolution_status"] == "resolved_low_confidence"
    assert result["match_score"] == 60.0


def test_resolve_identity_marks_unresolved_when_no_candidates():
    record = {
        "original_header": "Mystery metabolite",
        "normalized_name": "Mystery metabolite",
        "stage3_metabolite_name": "Mystery metabolite",
        "alias_in_parentheses": "",
        "query_candidates": ["Mystery metabolite"],
    }

    result = resolve_identity_from_cached_candidates(record, cached_payloads=[])

    assert result["resolution_status"] == "unresolved"
    assert result["match_reason"] == "no_identity_candidates"


def test_resolve_identity_normalizes_greek_symbols_in_candidate_payload():
    record = {
        "original_header": "Beta-Muricholic acid (β-MCA)",
        "normalized_name": "Beta-Muricholic acid (beta-MCA)",
        "stage3_metabolite_name": "Beta-Muricholic acid (beta-MCA)",
        "alias_in_parentheses": "beta-MCA",
        "query_candidates": [
            "Beta-Muricholic acid (beta-MCA)",
            "Beta-Muricholic acid",
            "beta-MCA",
        ],
    }
    payload = {
        "query": "Beta-Muricholic acid",
        "source": "pubchem",
        "candidates": [
            {
                "pubchem_cid": "1234",
                "candidate_name": "Beta-Muricholic acid (β-MCA)",
                "synonyms": ["β-MCA"],
                "smiles": "C",
                "inchikey": "BETA-KEY",
            }
        ],
    }

    result = resolve_identity_from_cached_candidates(record, cached_payloads=[payload])

    assert result["resolution_status"] == "resolved_high_confidence"
    assert result["pubchem_cid"] == "1234"


def test_load_taxonomy_enrichment_cache_returns_schema_when_missing():
    enrichment = load_taxonomy_enrichment_cache(None)

    assert list(enrichment.columns) == list(TAXONOMY_ENRICHMENT_COLUMNS)
    assert enrichment.empty


def test_load_taxonomy_enrichment_cache_reads_cached_fields(fixture_path):
    enrichment = load_taxonomy_enrichment_cache(fixture_path / "taxonomy_enrichment.csv")

    row = enrichment.loc[enrichment["normalized_name"] == "Cholic acid (CA)"].iloc[0]
    assert row["chebi_parent_terms"] == "bile acid"
    assert row["hmdb_super_class"] == "Lipids and lipid-like molecules"


def test_build_metabolite_annotation_from_evidence_flattens_bile_acid_taxonomy(fixture_path):
    normalized_headers = build_normalized_header_table(["Cholic acid (CA)"])
    identity = pd.read_csv(fixture_path / "identity_resolution_evidence.csv").fillna("")
    taxonomy = pd.read_csv(fixture_path / "taxonomy_enrichment.csv").fillna("")
    merged = merge_identity_and_taxonomy_evidence(identity, taxonomy)

    annotation = build_metabolite_annotation_from_evidence(normalized_headers, merged)

    assert annotation.to_dict(orient="records") == [
        {
            "metabolite_name": "Cholic acid (CA)",
            "superclass": "Lipids and lipid-like molecules",
            "subclass": "Bile acids, alcohols and derivatives",
            "pathway_tag": "bile_acid",
            "annotation_source": "auto_identity_resolution",
            "review_status": "auto_high_confidence",
            "ambiguous_flag": False,
            "notes": "compound=Cholic acid; reason=exact_candidate_name_match",
        }
    ]


def test_build_metabolite_annotation_marks_low_confidence_as_ambiguous():
    normalized_headers = build_normalized_header_table(["Cholic acid derivative"])
    evidence = pd.DataFrame.from_records(
        [
            {
                "original_header": "Cholic acid derivative",
                "normalized_name": "Cholic acid derivative",
                "stage3_metabolite_name": "Cholic acid derivative",
                "canonical_compound_name": "Cholic acid",
                "resolution_status": "resolved_low_confidence",
                "source": "pubchem",
                "pubchem_cid": "221493",
                "chebi_id": "",
                "hmdb_id": "",
                "inchikey": "LOW-CONFIDENCE",
                "smiles": "C",
                "synonyms": "Cholic acid",
                "match_score": 60.0,
                "match_reason": "loose_name_match",
                "evidence_json": "{}",
                **{column: "" for column in TAXONOMY_ENRICHMENT_COLUMNS if column != "normalized_name"},
            }
        ]
    )

    annotation = build_metabolite_annotation_from_evidence(normalized_headers, evidence)

    assert annotation.loc[0, "review_status"] == "needs_review"
    assert bool(annotation.loc[0, "ambiguous_flag"])


def test_build_metabolite_annotation_keeps_non_matching_names_out_of_pathway_tags():
    normalized_headers = build_normalized_header_table(["Campesterol"])
    evidence = pd.DataFrame.from_records(
        [
            {
                "original_header": "Campesterol",
                "normalized_name": "Campesterol",
                "stage3_metabolite_name": "Campesterol",
                "canonical_compound_name": "Campesterol",
                "resolution_status": "resolved_high_confidence",
                "source": "pubchem",
                "pubchem_cid": "173183",
                "chebi_id": "",
                "hmdb_id": "",
                "inchikey": "CAMP-KEY",
                "smiles": "C",
                "synonyms": "Campesterol",
                "match_score": 100.0,
                "match_reason": "exact_candidate_name_match",
                "evidence_json": "{}",
                **{column: "" for column in TAXONOMY_ENRICHMENT_COLUMNS if column != "normalized_name"},
            }
        ]
    )

    annotation = build_metabolite_annotation_from_evidence(normalized_headers, evidence)

    assert annotation.loc[0, "pathway_tag"] == ""


def test_build_model_membership_from_rules_routes_high_confidence_and_unresolved_rows():
    annotation = pd.DataFrame.from_records(
        [
            {
                "metabolite_name": "Cholic acid (CA)",
                "superclass": "Lipids and lipid-like molecules",
                "subclass": "Bile acids, alcohols and derivatives",
                "pathway_tag": "bile_acid",
                "annotation_source": "auto_identity_resolution",
                "review_status": "auto_high_confidence",
                "ambiguous_flag": False,
                "notes": "compound=Cholic acid",
            },
            {
                "metabolite_name": "Mystery metabolite",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "unresolved_identity",
                "review_status": "unresolved",
                "ambiguous_flag": True,
                "notes": "reason=no_identity_candidates",
            },
        ]
    )
    registry = pd.DataFrame.from_records(
        [
            {
                "model_id": "bile_acid",
                "model_label": "Bile Acid",
                "model_tier": "primary",
                "model_status": "draft",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "Bile acid family",
                "authority": "user",
                "notes": "",
            }
        ]
    )

    membership, review_queue, evidence = build_model_membership_from_rules(annotation, registry)

    assert membership[["model_id", "metabolite_name", "review_status"]].to_dict(orient="records") == [
        {"model_id": "bile_acid", "metabolite_name": "Cholic acid (CA)", "review_status": "auto_high_confidence"}
    ]
    assert review_queue.loc[0, "model_id"] == ""
    assert review_queue.loc[0, "review_status"] == "unresolved"
    assert not evidence.empty


def test_build_model_membership_from_rules_rejects_registry_rows_without_rules():
    annotation = pd.DataFrame.from_records(
        [
            {
                "metabolite_name": "Feature 1",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "auto_identity_resolution",
                "review_status": "auto_high_confidence",
                "ambiguous_flag": False,
                "notes": "",
            }
        ]
    )
    registry = pd.DataFrame.from_records(
        [
            {
                "model_id": "custom_model",
                "model_label": "Custom",
                "model_tier": "primary",
                "model_status": "draft",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "Custom model",
                "authority": "user",
                "notes": "",
            }
        ]
    )

    with pytest.raises(ValueError, match="missing membership rules"):
        build_model_membership_from_rules(annotation, registry)


def test_build_model_membership_from_rules_avoids_short_substring_false_positive():
    annotation = pd.DataFrame.from_records(
        [
            {
                "metabolite_name": "Campesterol",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "auto_identity_resolution",
                "review_status": "auto_high_confidence",
                "ambiguous_flag": False,
                "notes": "",
            }
        ]
    )
    registry = pd.DataFrame.from_records(
        [
            {
                "model_id": "nucleotide_energy",
                "model_label": "Nucleotide",
                "model_tier": "primary",
                "model_status": "draft",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "Nucleotide model",
                "authority": "user",
                "notes": "",
            }
        ]
    )

    membership, review_queue, evidence = build_model_membership_from_rules(annotation, registry)

    assert membership.empty
    assert review_queue.empty
    assert evidence.empty


def test_build_model_space_writes_csvs_review_queue_cache_and_manifest(tmp_path, fixture_path):
    matrix_path, preprocess_root, registry_path = _write_tiny_builder_inputs(tmp_path)
    output_root = tmp_path / "model_space_auto"

    result = build_model_space(
        matrix_path=matrix_path,
        preprocess_root=preprocess_root,
        registry_path=registry_path,
        output_root=output_root,
        identity_evidence_path=fixture_path / "identity_resolution_evidence.csv",
        taxonomy_enrichment_path=fixture_path / "taxonomy_enrichment.csv",
        cache_version="test-cache-v1",
        refresh_pubchem_cache=False,
    )

    assert (output_root / "stimulus_sample_map.csv").exists()
    assert (output_root / "metabolite_annotation.csv").exists()
    assert (output_root / "model_registry.csv").exists()
    assert (output_root / "model_membership.csv").exists()
    assert (output_root / "model_membership_review_queue.csv").exists()
    assert (output_root / "model_space_manifest.json").exists()
    assert (output_root / "cache" / "normalized_headers.csv").exists()
    assert (output_root / "cache" / "identity_resolution_evidence.csv").exists()
    assert (output_root / "cache" / "taxonomy_enrichment_evidence.csv").exists()
    assert (output_root / "cache" / "membership_rule_evidence.csv").exists()
    assert result["output_root"] == output_root

    review_queue = pd.read_csv(output_root / "model_membership_review_queue.csv").fillna("")
    assert "Mystery metabolite" in set(review_queue["metabolite_name"])
    manifest = json.loads((output_root / "model_space_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cache_version"] == "test-cache-v1"
    assert manifest["matrix_path"] == str(matrix_path)
    assert manifest["trial_metadata_path"] == str(preprocess_root / "trial_level" / "trial_metadata.parquet")
    assert manifest["outputs"]["model_membership_review_queue"] == "model_membership_review_queue.csv"


def test_build_model_space_uses_matrix_canonical_names_for_known_overrides(tmp_path):
    matrix_path = tmp_path / "matrix.xlsx"
    preprocess_root = tmp_path / "preprocess"
    registry_path = tmp_path / "model_registry.csv"
    output_root = tmp_path / "model_space_auto"
    (preprocess_root / "trial_level").mkdir(parents=True, exist_ok=True)
    _write_matrix(
        matrix_path,
        headers=["Tauro-α-muricholic acid (ω-TMCA)"],
        rows=[["A226", 1.0]],
    )
    pd.DataFrame.from_records([{"stimulus": "b34_0", "stim_name": "A226 stationary"}]).to_parquet(
        preprocess_root / "trial_level" / "trial_metadata.parquet",
        index=False,
    )
    _write_registry(registry_path)

    build_model_space(
        matrix_path=matrix_path,
        preprocess_root=preprocess_root,
        registry_path=registry_path,
        output_root=output_root,
        identity_evidence_path=None,
        taxonomy_enrichment_path=None,
        cache_version="override-v1",
        refresh_pubchem_cache=False,
    )

    annotation = pd.read_csv(output_root / "metabolite_annotation.csv").fillna("")
    assert annotation.loc[0, "metabolite_name"] == "Tauro-omega-muricholic acid (omega-TMCA)"


def test_generated_model_space_resolves_with_existing_stage3_loader(tmp_path, fixture_path):
    matrix_path, preprocess_root, registry_path = _write_tiny_builder_inputs(tmp_path)
    output_root = tmp_path / "model_space_auto"

    build_model_space(
        matrix_path=matrix_path,
        preprocess_root=preprocess_root,
        registry_path=registry_path,
        output_root=output_root,
        identity_evidence_path=fixture_path / "identity_resolution_evidence.csv",
        taxonomy_enrichment_path=fixture_path / "taxonomy_enrichment.csv",
        cache_version="test-cache-v1",
        refresh_pubchem_cache=False,
    )

    resolved = resolve_model_inputs(output_root, matrix_path)
    non_global_membership = resolved["model_membership_resolved"].loc[
        resolved["model_membership_resolved"]["model_id"].astype(str) != "global_profile"
    ]

    assert not resolved["stimulus_sample_map"].empty
    assert "global_profile" in set(resolved["model_registry_resolved"]["model_id"])
    assert set(non_global_membership["review_status"]) <= {"auto_high_confidence"}


def test_build_model_space_cli_smoke(tmp_path, fixture_path):
    matrix_path, preprocess_root, registry_path = _write_tiny_builder_inputs(tmp_path)
    output_root = tmp_path / "model_space_auto"

    completed = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/build_model_space.py",
            "--matrix",
            str(matrix_path),
            "--preprocess-root",
            str(preprocess_root),
            "--registry",
            str(registry_path),
            "--output-root",
            str(output_root),
            "--identity-evidence-cache",
            str(fixture_path / "identity_resolution_evidence.csv"),
            "--taxonomy-enrichment-cache",
            str(fixture_path / "taxonomy_enrichment.csv"),
            "--cache-version",
            "cli-cache-v1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "model_space_auto" in completed.stdout
    assert (output_root / "model_space_manifest.json").exists()


def test_build_model_space_cli_resolves_repo_relative_defaults(tmp_path):
    resolved = BUILD_MODEL_SPACE_CLI.resolve_repo_path("data/matrix.xlsx")

    assert resolved.name == "matrix.xlsx"


def test_identity_and_taxonomy_fixture_schemas(fixture_path):
    identity = pd.read_csv(fixture_path / "identity_resolution_evidence.csv").fillna("")
    taxonomy = pd.read_csv(fixture_path / "taxonomy_enrichment.csv").fillna("")

    assert list(identity.columns) == list(IDENTITY_EVIDENCE_COLUMNS)
    assert list(taxonomy.columns) == list(TAXONOMY_ENRICHMENT_COLUMNS)


def test_merge_identity_and_taxonomy_evidence_rejects_conflicting_ids():
    identity = pd.DataFrame.from_records(
        [
            {
                "original_header": "Cholic acid (CA)",
                "normalized_name": "Cholic acid (CA)",
                "stage3_metabolite_name": "Cholic acid (CA)",
                "canonical_compound_name": "Cholic acid",
                "resolution_status": "resolved_high_confidence",
                "source": "pubchem",
                "pubchem_cid": "221493",
                "chebi_id": "",
                "hmdb_id": "",
                "inchikey": "MATCHING-KEY",
                "smiles": "C",
                "synonyms": "Cholic acid | CA",
                "match_score": 100.0,
                "match_reason": "exact_candidate_name_match",
                "evidence_json": "{}",
            }
        ],
        columns=IDENTITY_EVIDENCE_COLUMNS,
    )
    taxonomy = pd.DataFrame.from_records(
        [
            {
                "normalized_name": "Cholic acid (CA)",
                "pubchem_cid": "999999",
                "chebi_id": "",
                "hmdb_id": "",
                "inchikey": "OTHER-KEY",
                "chebi_parent_terms": "bile acid",
                "hmdb_super_class": "",
                "hmdb_class": "",
                "hmdb_sub_class": "",
                "classyfire_kingdom": "",
                "classyfire_superclass": "",
                "classyfire_class": "",
                "classyfire_subclass": "",
                "pathway_tags": "",
                "annotation_confidence": "high",
                "annotation_status": "resolved_high_confidence",
                "annotation_evidence": "fixture",
            }
        ],
        columns=TAXONOMY_ENRICHMENT_COLUMNS,
    )

    with pytest.raises(ValueError, match="taxonomy enrichment conflicts"):
        merge_identity_and_taxonomy_evidence(identity, taxonomy)


def test_build_model_space_refresh_pubchem_cache_overrides_existing_rows(tmp_path, fixture_path, monkeypatch):
    matrix_path, preprocess_root, registry_path = _write_tiny_builder_inputs(tmp_path)
    output_root = tmp_path / "model_space_auto"
    refresh_cache_path = tmp_path / "refresh_identity.csv"
    pd.DataFrame.from_records(
        [
            {
                "original_header": "Cholic acid (CA)",
                "normalized_name": "Cholic acid (CA)",
                "stage3_metabolite_name": "Cholic acid (CA)",
                "canonical_compound_name": "",
                "resolution_status": "unresolved",
                "source": "",
                "pubchem_cid": "",
                "chebi_id": "",
                "hmdb_id": "",
                "inchikey": "",
                "smiles": "",
                "synonyms": "",
                "match_score": 0.0,
                "match_reason": "no_identity_candidates",
                "evidence_json": "{}",
            },
            {
                "original_header": "Mystery metabolite",
                "normalized_name": "Mystery metabolite",
                "stage3_metabolite_name": "Mystery metabolite",
                "canonical_compound_name": "",
                "resolution_status": "unresolved",
                "source": "",
                "pubchem_cid": "",
                "chebi_id": "",
                "hmdb_id": "",
                "inchikey": "",
                "smiles": "",
                "synonyms": "",
                "match_score": 0.0,
                "match_reason": "no_identity_candidates",
                "evidence_json": "{}",
            },
        ]
    ).to_csv(refresh_cache_path, index=False)

    def fake_fetch(query: str, *, timeout_seconds: float = 20.0):
        if query == "Cholic acid (CA)":
            return json.loads((fixture_path / "pubchem_cholic_acid.json").read_text(encoding="utf-8"))
        return {"query": query, "source": "pubchem", "candidates": []}

    monkeypatch.setattr(model_space_seed_module, "fetch_pubchem_payload", fake_fetch)

    build_model_space(
        matrix_path=matrix_path,
        preprocess_root=preprocess_root,
        registry_path=registry_path,
        output_root=output_root,
        identity_evidence_path=refresh_cache_path,
        taxonomy_enrichment_path=fixture_path / "taxonomy_enrichment.csv",
        cache_version="refresh-cache-v1",
        refresh_pubchem_cache=True,
    )

    identity_output = pd.read_csv(output_root / "cache" / "identity_resolution_evidence.csv").fillna("")
    cholic_row = identity_output.loc[identity_output["normalized_name"] == "Cholic acid (CA)"].iloc[0]
    assert cholic_row["resolution_status"] == "resolved_high_confidence"
    assert str(cholic_row["pubchem_cid"]).startswith("221493")


def test_build_model_space_refresh_pubchem_cache_tries_all_query_candidates(tmp_path, monkeypatch):
    matrix_path = tmp_path / "matrix.xlsx"
    preprocess_root = tmp_path / "preprocess"
    registry_path = tmp_path / "model_registry.csv"
    output_root = tmp_path / "model_space_auto"
    refresh_cache_path = tmp_path / "refresh_identity.csv"
    (preprocess_root / "trial_level").mkdir(parents=True, exist_ok=True)
    _write_matrix(
        matrix_path,
        headers=["Beta-Muricholic acid (β-MCA)"],
        rows=[["A226", 1.0]],
    )
    pd.DataFrame.from_records([{"stimulus": "b34_0", "stim_name": "A226 stationary"}]).to_parquet(
        preprocess_root / "trial_level" / "trial_metadata.parquet",
        index=False,
    )
    _write_registry(registry_path)
    pd.DataFrame.from_records(
        [
            {
                "original_header": "Beta-Muricholic acid (β-MCA)",
                "normalized_name": "Beta-Muricholic acid (beta-MCA)",
                "stage3_metabolite_name": "Beta-Muricholic acid (beta-MCA)",
                "canonical_compound_name": "",
                "resolution_status": "unresolved",
                "source": "",
                "pubchem_cid": "",
                "chebi_id": "",
                "hmdb_id": "",
                "inchikey": "",
                "smiles": "",
                "synonyms": "",
                "match_score": 0.0,
                "match_reason": "no_identity_candidates",
                "evidence_json": "{}",
            }
        ]
    ).to_csv(refresh_cache_path, index=False)

    def fake_fetch(query: str, *, timeout_seconds: float = 20.0):
        if query == "Beta-Muricholic acid":
            return {
                "query": query,
                "source": "pubchem",
                "candidates": [
                    {
                        "pubchem_cid": "5678",
                        "candidate_name": "Beta-Muricholic acid",
                        "synonyms": ["β-MCA"],
                        "smiles": "C",
                        "inchikey": "BETA-REFRESH",
                    }
                ],
            }
        return {"query": query, "source": "pubchem", "candidates": []}

    monkeypatch.setattr(model_space_seed_module, "fetch_pubchem_payload", fake_fetch)

    build_model_space(
        matrix_path=matrix_path,
        preprocess_root=preprocess_root,
        registry_path=registry_path,
        output_root=output_root,
        identity_evidence_path=refresh_cache_path,
        taxonomy_enrichment_path=None,
        cache_version="refresh-query-v1",
        refresh_pubchem_cache=True,
    )

    identity_output = pd.read_csv(output_root / "cache" / "identity_resolution_evidence.csv").fillna("")
    row = identity_output.loc[
        identity_output["normalized_name"] == "Beta-Muricholic acid (beta-MCA)"
    ].iloc[0]
    assert row["resolution_status"] == "resolved_high_confidence"
    assert str(row["pubchem_cid"]).startswith("5678")
