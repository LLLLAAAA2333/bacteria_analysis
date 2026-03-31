import pandas as pd
import pytest

from bacteria_analysis.model_space import (
    build_metabolite_annotation_skeleton,
    load_stimulus_sample_map,
    load_model_registry,
    read_metabolite_matrix,
    resolve_model_inputs,
)

MODEL_REGISTRY_COLUMNS = [
    "model_id",
    "model_label",
    "model_tier",
    "model_status",
    "feature_kind",
    "distance_kind",
    "description",
    "authority",
    "notes",
]
MODEL_MEMBERSHIP_COLUMNS = [
    "model_id",
    "metabolite_name",
    "membership_source",
    "review_status",
    "ambiguous_flag",
    "notes",
]
METABOLITE_ANNOTATION_COLUMNS = [
    "metabolite_name",
    "superclass",
    "subclass",
    "pathway_tag",
    "annotation_source",
    "review_status",
    "ambiguous_flag",
    "notes",
]


def _write_stage3_model_space_files(root, *, registry_rows, membership_rows, annotation_rows):
    pd.DataFrame.from_records(registry_rows, columns=MODEL_REGISTRY_COLUMNS).to_csv(
        root / "model_registry.csv", index=False
    )
    pd.DataFrame.from_records(membership_rows, columns=MODEL_MEMBERSHIP_COLUMNS).to_csv(
        root / "model_membership.csv", index=False
    )
    pd.DataFrame.from_records(annotation_rows, columns=METABOLITE_ANNOTATION_COLUMNS).to_csv(
        root / "metabolite_annotation.csv", index=False
    )


def test_load_stimulus_sample_map_requires_unique_sample_ids(tmp_path):
    root = tmp_path / "model_space"
    root.mkdir()
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "stimulus": "stimulus_a", "stim_name": "Stimulus A"},
            {"sample_id": "A001", "stimulus": "stimulus_b", "stim_name": "Stimulus B"},
            {"sample_id": "A003", "stimulus": "stimulus_c", "stim_name": "Stimulus C"},
        ]
    ).to_csv(root / "duplicate_stimulus_sample_map.csv", index=False)

    with pytest.raises(ValueError, match="sample_id values must be unique"):
        load_stimulus_sample_map(root / "duplicate_stimulus_sample_map.csv")


def test_load_stimulus_sample_map_requires_unique_stimulus_values(tmp_path):
    root = tmp_path / "model_space"
    root.mkdir()
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "stimulus": "stimulus_a", "stim_name": "Stimulus A"},
            {"sample_id": "A002", "stimulus": "stimulus_a", "stim_name": "Stimulus B"},
            {"sample_id": "A003", "stimulus": "stimulus_c", "stim_name": "Stimulus C"},
        ]
    ).to_csv(root / "duplicate_stimulus_values_map.csv", index=False)

    with pytest.raises(ValueError, match="stimulus values must be unique"):
        load_stimulus_sample_map(root / "duplicate_stimulus_values_map.csv")


def test_load_stimulus_sample_map_allows_duplicate_stim_name_values(tmp_path):
    root = tmp_path / "model_space"
    root.mkdir()
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "stimulus": "stimulus_a", "stim_name": "Shared Stimulus"},
            {"sample_id": "A002", "stimulus": "stimulus_b", "stim_name": "Shared Stimulus"},
            {"sample_id": "A003", "stimulus": "stimulus_c", "stim_name": "Shared Stimulus"},
        ]
    ).to_csv(root / "duplicate_stim_name_map.csv", index=False)

    frame = load_stimulus_sample_map(root / "duplicate_stim_name_map.csv")
    assert frame["stim_name"].tolist() == ["Shared Stimulus", "Shared Stimulus", "Shared Stimulus"]


def test_load_stimulus_sample_map_rejects_blank_stim_name_values(tmp_path):
    root = tmp_path / "model_space"
    root.mkdir()
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "stimulus": "stimulus_a", "stim_name": "Stimulus A"},
            {"sample_id": "A002", "stimulus": "stimulus_b", "stim_name": ""},
            {"sample_id": "A003", "stimulus": "stimulus_c", "stim_name": "Stimulus C"},
        ]
    ).to_csv(root / "blank_stim_name_map.csv", index=False)

    with pytest.raises(ValueError, match="stim_name values must be non-empty"):
        load_stimulus_sample_map(root / "blank_stim_name_map.csv")


def test_read_metabolite_matrix_loads_expected_sample_ids(tmp_path):
    matrix_path = tmp_path / "matrix.xlsx"
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "feature_1": 0.1, "feature_2": 1.0},
            {"sample_id": "A002", "feature_1": 0.2, "feature_2": 0.8},
            {"sample_id": "A003", "feature_1": 0.3, "feature_2": 0.6},
        ]
    ).to_excel(matrix_path, index=False, engine="openpyxl")

    matrix = read_metabolite_matrix(matrix_path)
    assert matrix.index.tolist() == ["A001", "A002", "A003"]


def test_read_metabolite_matrix_rejects_blank_sample_id_cells(tmp_path):
    matrix_path = tmp_path / "blank_matrix.xlsx"
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "feature_1": 0.1, "feature_2": 1.0},
            {"sample_id": None, "feature_1": 0.2, "feature_2": 0.8},
            {"sample_id": "A003", "feature_1": 0.3, "feature_2": 0.6},
        ]
    ).to_excel(matrix_path, index=False, engine="openpyxl")

    with pytest.raises(ValueError, match="sample_id values must be non-empty"):
        read_metabolite_matrix(matrix_path)


def test_read_metabolite_matrix_rejects_duplicate_metabolite_columns(tmp_path):
    matrix_path = tmp_path / "duplicate_metabolite_columns.xlsx"
    pd.DataFrame(
        [
            ["A001", 0.1, 1.0],
            ["A002", 0.2, 0.8],
            ["A003", 0.3, 0.6],
        ],
        columns=["sample_id", "feature_1", "feature_1"],
    ).to_excel(matrix_path, index=False, engine="openpyxl")

    with pytest.raises(ValueError, match="metabolite column names must be unique"):
        read_metabolite_matrix(matrix_path)


def test_resolve_model_inputs_accepts_broad_union_text_when_registry_metadata_is_valid(tmp_path):
    root = tmp_path / "model_space"
    root.mkdir()
    matrix_path = root / "matrix.xlsx"
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "feature_1": 0.1, "feature_2": 1.0},
            {"sample_id": "A002", "feature_1": 0.2, "feature_2": 0.8},
            {"sample_id": "A003", "feature_1": 0.3, "feature_2": 0.6},
        ]
    ).to_excel(matrix_path, index=False, engine="openpyxl")

    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "stimulus": "stimulus_a", "stim_name": "Stimulus A"},
            {"sample_id": "A002", "stimulus": "stimulus_b", "stim_name": "Stimulus B"},
            {"sample_id": "A003", "stimulus": "stimulus_c", "stim_name": "Stimulus C"},
        ]
    ).to_csv(root / "stimulus_sample_map.csv", index=False)

    _write_stage3_model_space_files(
        root,
        registry_rows=[
            {
                "model_id": "global_profile",
                "model_label": "Global Metabolite Profile",
                "model_tier": "primary",
                "model_status": "primary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "All matrix metabolites",
                "authority": "user",
                "notes": "",
            },
            {
                "model_id": "broad_union",
                "model_label": "Broad Union Model",
                "model_tier": "primary",
                "model_status": "supplementary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "Exploratory broad union",
                "authority": "exploratory",
                "notes": "",
            },
        ],
        membership_rows=[
            {
                "model_id": "global_profile",
                "metabolite_name": "feature_1",
                "membership_source": "seed",
                "review_status": "reviewed",
                "ambiguous_flag": False,
                "notes": "",
            }
        ],
        annotation_rows=[
            {
                "metabolite_name": "feature_1",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "",
                "review_status": "",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "metabolite_name": "feature_2",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "",
                "review_status": "",
                "ambiguous_flag": False,
                "notes": "",
            },
        ],
    )

    resolved = resolve_model_inputs(root, matrix_path)
    broad_union_row = resolved["model_registry_resolved"].loc[
        resolved["model_registry_resolved"]["model_id"] == "broad_union"
    ].iloc[0]

    assert broad_union_row["model_tier"] == "primary"
    assert broad_union_row["model_status"] == "supplementary"


def test_resolve_model_inputs_preserves_case_variant_global_profile_rows(tmp_path):
    root = tmp_path / "model_space"
    root.mkdir()
    matrix_path = root / "matrix.xlsx"
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "feature_1": 0.1, "feature_2": 1.0},
            {"sample_id": "A002", "feature_1": 0.2, "feature_2": 0.8},
            {"sample_id": "A003", "feature_1": 0.3, "feature_2": 0.6},
        ]
    ).to_excel(matrix_path, index=False, engine="openpyxl")

    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "stimulus": "stimulus_a", "stim_name": "Stimulus A"},
            {"sample_id": "A002", "stimulus": "stimulus_b", "stim_name": "Stimulus B"},
            {"sample_id": "A003", "stimulus": "stimulus_c", "stim_name": "Stimulus C"},
        ]
    ).to_csv(root / "stimulus_sample_map.csv", index=False)

    _write_stage3_model_space_files(
        root,
        registry_rows=[
            {
                "model_id": "GLOBAL_PROFILE",
                "model_label": "Global Metabolite Profile",
                "model_tier": "primary",
                "model_status": "primary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "All matrix metabolites",
                "authority": "user",
                "notes": "",
            }
        ],
        membership_rows=[
            {
                "model_id": "GLOBAL_PROFILE",
                "metabolite_name": "feature_1",
                "membership_source": "seed",
                "review_status": "reviewed",
                "ambiguous_flag": False,
                "notes": "",
            }
        ],
        annotation_rows=[
            {
                "metabolite_name": "feature_1",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "",
                "review_status": "",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "metabolite_name": "feature_2",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "",
                "review_status": "",
                "ambiguous_flag": False,
                "notes": "",
            },
        ],
    )

    resolved = resolve_model_inputs(root, matrix_path)

    assert resolved["model_registry_resolved"]["model_id"].tolist() == ["GLOBAL_PROFILE"]
    assert resolved["model_membership_resolved"]["model_id"].tolist() == ["GLOBAL_PROFILE", "GLOBAL_PROFILE"]
    assert set(resolved["model_membership_resolved"]["metabolite_name"]) == {"feature_1", "feature_2"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_tier", "secondary"),
        ("model_status", "pending"),
        ("feature_kind", "rank_abundance"),
        ("distance_kind", "cosine"),
    ],
)
def test_load_model_registry_rejects_invalid_classification_values(tmp_path, field, value):
    root = tmp_path / "model_space"
    root.mkdir()
    registry_rows = [
        {
            "model_id": "global_profile",
            "model_label": "Global Metabolite Profile",
            "model_tier": "primary",
            "model_status": "primary",
            "feature_kind": "continuous_abundance",
            "distance_kind": "correlation",
            "description": "All matrix metabolites",
            "authority": "user",
            "notes": "",
        },
        {
            "model_id": "broad_union",
            "model_label": "Broad Union Model",
            "model_tier": "primary",
            "model_status": "supplementary",
            "feature_kind": "continuous_abundance",
            "distance_kind": "correlation",
            "description": "Exploratory broad union",
            "authority": "exploratory",
            "notes": "",
        },
    ]
    registry_rows[1][field] = value
    pd.DataFrame.from_records(registry_rows, columns=MODEL_REGISTRY_COLUMNS).to_csv(
        root / "model_registry.csv", index=False
    )

    with pytest.raises(ValueError, match=f"{field} values must be one of"):
        load_model_registry(root / "model_registry.csv")


@pytest.mark.parametrize("blank_field", ["model_tier", "model_status"])
def test_load_model_registry_rejects_blank_required_fields(tmp_path, blank_field):
    root = tmp_path / "model_space"
    root.mkdir()
    registry_rows = [
        {
            "model_id": "global_profile",
            "model_label": "Global Metabolite Profile",
            "model_tier": "primary",
            "model_status": "primary",
            "feature_kind": "continuous_abundance",
            "distance_kind": "correlation",
            "description": "All matrix metabolites",
            "authority": "user",
            "notes": "",
        },
        {
            "model_id": "broad_union",
            "model_label": "Broad Union Model",
            "model_tier": "primary",
            "model_status": "supplementary",
            "feature_kind": "continuous_abundance",
            "distance_kind": "correlation",
            "description": "Exploratory broad union",
            "authority": "exploratory",
            "notes": "",
        },
    ]
    registry_rows[1][blank_field] = ""
    pd.DataFrame.from_records(registry_rows, columns=MODEL_REGISTRY_COLUMNS).to_csv(
        root / "model_registry.csv", index=False
    )

    with pytest.raises(ValueError, match=f"{blank_field} values must be non-empty"):
        load_model_registry(root / "model_registry.csv")


def test_load_model_registry_canonicalizes_allowed_value_fields(tmp_path):
    root = tmp_path / "model_space"
    root.mkdir()
    registry_rows = [
        {
            "model_id": "global_profile",
            "model_label": "Global Metabolite Profile",
            "model_tier": "PRIMARY",
            "model_status": "Draft",
            "feature_kind": "Continuous_Abundance",
            "distance_kind": "Euclidean",
            "description": "All matrix metabolites",
            "authority": "user",
            "notes": "",
        },
        {
            "model_id": "broad_union",
            "model_label": "Broad Union Model",
            "model_tier": "Supplementary",
            "model_status": "EXCLUDED",
            "feature_kind": "Binary_Presence",
            "distance_kind": "JACCARD",
            "description": "Exploratory broad union",
            "authority": "exploratory",
            "notes": "",
        },
    ]
    pd.DataFrame.from_records(registry_rows, columns=MODEL_REGISTRY_COLUMNS).to_csv(
        root / "model_registry.csv", index=False
    )

    frame = load_model_registry(root / "model_registry.csv")

    assert frame["model_tier"].tolist() == ["primary", "supplementary"]
    assert frame["model_status"].tolist() == ["draft", "excluded"]
    assert frame["feature_kind"].tolist() == ["continuous_abundance", "binary_presence"]
    assert frame["distance_kind"].tolist() == ["euclidean", "jaccard"]


def test_resolve_model_inputs_accepts_draft_broad_union_text_without_keyword_inference(tmp_path):
    root = tmp_path / "model_space"
    root.mkdir()
    matrix_path = root / "matrix.xlsx"
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "feature_1": 0.1, "feature_2": 1.0},
            {"sample_id": "A002", "feature_1": 0.2, "feature_2": 0.8},
            {"sample_id": "A003", "feature_1": 0.3, "feature_2": 0.6},
        ]
    ).to_excel(matrix_path, index=False, engine="openpyxl")

    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "stimulus": "stimulus_a", "stim_name": "Stimulus A"},
            {"sample_id": "A002", "stimulus": "stimulus_b", "stim_name": "Stimulus B"},
            {"sample_id": "A003", "stimulus": "stimulus_c", "stim_name": "Stimulus C"},
        ]
    ).to_csv(root / "stimulus_sample_map.csv", index=False)

    _write_stage3_model_space_files(
        root,
        registry_rows=[
            {
                "model_id": "draft_broad_union",
                "model_label": "Draft Broad Union Model",
                "model_tier": "primary",
                "model_status": "draft",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "Draft broad union candidate",
                "authority": "user",
                "notes": "",
            }
        ],
        membership_rows=[],
        annotation_rows=[
            {
                "metabolite_name": "feature_1",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "",
                "review_status": "",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "metabolite_name": "feature_2",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "",
                "review_status": "",
                "ambiguous_flag": False,
                "notes": "",
            },
        ],
    )

    resolved = resolve_model_inputs(root, matrix_path)
    draft_row = resolved["model_registry_resolved"].loc[
        resolved["model_registry_resolved"]["model_id"] == "draft_broad_union"
    ].iloc[0]

    assert draft_row["model_tier"] == "primary"
    assert draft_row["model_status"] == "draft"


def test_resolve_model_inputs_rejects_unknown_membership_metabolite(tmp_path):
    root = tmp_path / "model_space"
    root.mkdir()
    matrix_path = root / "matrix.xlsx"
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "feature_1": 0.1, "feature_2": 1.0},
            {"sample_id": "A002", "feature_1": 0.2, "feature_2": 0.8},
            {"sample_id": "A003", "feature_1": 0.3, "feature_2": 0.6},
        ]
    ).to_excel(matrix_path, index=False, engine="openpyxl")

    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "stimulus": "stimulus_a", "stim_name": "Stimulus A"},
            {"sample_id": "A002", "stimulus": "stimulus_b", "stim_name": "Stimulus B"},
            {"sample_id": "A003", "stimulus": "stimulus_c", "stim_name": "Stimulus C"},
        ]
    ).to_csv(root / "stimulus_sample_map.csv", index=False)

    _write_stage3_model_space_files(
        root,
        registry_rows=[
            {
                "model_id": "global_profile",
                "model_label": "Global Metabolite Profile",
                "model_tier": "primary",
                "model_status": "primary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "All matrix metabolites",
                "authority": "user",
                "notes": "",
            }
        ],
        membership_rows=[
            {
                "model_id": "global_profile",
                "metabolite_name": "feature_typo",
                "membership_source": "seed",
                "review_status": "reviewed",
                "ambiguous_flag": False,
                "notes": "",
            }
        ],
        annotation_rows=[
            {
                "metabolite_name": "feature_1",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "",
                "review_status": "",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "metabolite_name": "feature_2",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "",
                "review_status": "",
                "ambiguous_flag": False,
                "notes": "",
            },
        ],
    )

    with pytest.raises(ValueError, match="model_membership metabolites must exist in the matrix: feature_typo"):
        resolve_model_inputs(root, matrix_path)


def test_resolve_model_inputs_allows_primary_supplementary_status_when_not_broad_union(tmp_path):
    root = tmp_path / "model_space"
    root.mkdir()
    matrix_path = root / "matrix.xlsx"
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "feature_1": 0.1, "feature_2": 1.0},
            {"sample_id": "A002", "feature_1": 0.2, "feature_2": 0.8},
            {"sample_id": "A003", "feature_1": 0.3, "feature_2": 0.6},
        ]
    ).to_excel(matrix_path, index=False, engine="openpyxl")

    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "stimulus": "stimulus_a", "stim_name": "Stimulus A"},
            {"sample_id": "A002", "stimulus": "stimulus_b", "stim_name": "Stimulus B"},
            {"sample_id": "A003", "stimulus": "stimulus_c", "stim_name": "Stimulus C"},
        ]
    ).to_csv(root / "stimulus_sample_map.csv", index=False)

    _write_stage3_model_space_files(
        root,
        registry_rows=[
            {
                "model_id": "bile_acid",
                "model_label": "Bile Acid",
                "model_tier": "primary",
                "model_status": "supplementary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "Narrow bile acid family",
                "authority": "user",
                "notes": "",
            }
        ],
        membership_rows=[],
        annotation_rows=[
            {
                "metabolite_name": "feature_1",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "",
                "review_status": "",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "metabolite_name": "feature_2",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "",
                "review_status": "",
                "ambiguous_flag": False,
                "notes": "",
            },
        ],
    )

    resolved = resolve_model_inputs(root, matrix_path)
    row = resolved["model_registry_resolved"].loc[resolved["model_registry_resolved"]["model_id"] == "bile_acid"].iloc[0]
    assert row["model_status"] == "supplementary"
    assert row["is_primary_family"]


def test_resolve_model_inputs_seeds_global_profile_for_header_only_empty_membership(tmp_path):
    root = tmp_path / "model_space"
    root.mkdir()
    matrix_path = root / "matrix.xlsx"
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "feature_1": 0.1, "feature_2": 1.0},
            {"sample_id": "A002", "feature_1": 0.2, "feature_2": 0.8},
            {"sample_id": "A003", "feature_1": 0.3, "feature_2": 0.6},
        ]
    ).to_excel(matrix_path, index=False, engine="openpyxl")

    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "stimulus": "stimulus_a", "stim_name": "Stimulus A"},
            {"sample_id": "A002", "stimulus": "stimulus_b", "stim_name": "Stimulus B"},
            {"sample_id": "A003", "stimulus": "stimulus_c", "stim_name": "Stimulus C"},
        ]
    ).to_csv(root / "stimulus_sample_map.csv", index=False)

    _write_stage3_model_space_files(
        root,
        registry_rows=[
            {
                "model_id": "bile_acid",
                "model_label": "Bile Acid",
                "model_tier": "primary",
                "model_status": "draft",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "Narrow bile acid family",
                "authority": "user",
                "notes": "",
            }
        ],
        membership_rows=[],
        annotation_rows=[
            {
                "metabolite_name": "feature_1",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "",
                "review_status": "",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "metabolite_name": "feature_2",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "",
                "review_status": "",
                "ambiguous_flag": False,
                "notes": "",
            },
        ],
    )

    resolved = resolve_model_inputs(root, matrix_path)

    global_profile_rows = resolved["model_membership_resolved"].loc[
        resolved["model_membership_resolved"]["model_id"] == "global_profile"
    ]
    assert set(global_profile_rows["metabolite_name"]) == {"feature_1", "feature_2"}


def test_resolve_model_inputs_backfills_missing_global_profile_metabolites(tmp_path):
    root = tmp_path / "model_space"
    root.mkdir()
    matrix_path = root / "matrix.xlsx"
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "feature_1": 0.1, "feature_2": 1.0},
            {"sample_id": "A002", "feature_1": 0.2, "feature_2": 0.8},
            {"sample_id": "A003", "feature_1": 0.3, "feature_2": 0.6},
        ]
    ).to_excel(matrix_path, index=False, engine="openpyxl")

    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "stimulus": "stimulus_a", "stim_name": "Stimulus A"},
            {"sample_id": "A002", "stimulus": "stimulus_b", "stim_name": "Stimulus B"},
            {"sample_id": "A003", "stimulus": "stimulus_c", "stim_name": "Stimulus C"},
        ]
    ).to_csv(root / "stimulus_sample_map.csv", index=False)

    _write_stage3_model_space_files(
        root,
        registry_rows=[
            {
                "model_id": "bile_acid",
                "model_label": "Bile Acid",
                "model_tier": "primary",
                "model_status": "draft",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "Narrow bile acid family",
                "authority": "user",
                "notes": "",
            }
        ],
        membership_rows=[
            {
                "model_id": "global_profile",
                "metabolite_name": "feature_1",
                "membership_source": "manual",
                "review_status": "reviewed",
                "ambiguous_flag": False,
                "notes": "",
            }
        ],
        annotation_rows=[
            {
                "metabolite_name": "feature_1",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "",
                "review_status": "",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "metabolite_name": "feature_2",
                "superclass": "",
                "subclass": "",
                "pathway_tag": "",
                "annotation_source": "",
                "review_status": "",
                "ambiguous_flag": False,
                "notes": "",
            },
        ],
    )

    resolved = resolve_model_inputs(root, matrix_path)

    global_profile_rows = resolved["model_membership_resolved"].loc[
        resolved["model_membership_resolved"]["model_id"] == "global_profile"
    ]
    assert set(global_profile_rows["metabolite_name"]) == {"feature_1", "feature_2"}
    seeded_row = global_profile_rows.loc[global_profile_rows["metabolite_name"] == "feature_2"].iloc[0]
    assert seeded_row["membership_source"] == "matrix_all_columns"


def test_build_annotation_skeleton_emits_all_matrix_metabolites(tmp_path):
    matrix_path = tmp_path / "matrix.xlsx"
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "Cholic acid (CA)": 0.10, "Palmitic acid": 0.30},
            {"sample_id": "A002", "Cholic acid (CA)": 0.20, "Palmitic acid": 0.40},
            {"sample_id": "A003", "Cholic acid (CA)": 0.30, "Palmitic acid": 0.50},
        ]
    ).to_excel(matrix_path, index=False, engine="openpyxl")

    annotation = build_metabolite_annotation_skeleton(matrix_path)
    assert {"metabolite_name", "review_status", "ambiguous_flag"}.issubset(annotation.columns)
    assert "Cholic acid (CA)" in set(annotation["metabolite_name"])
