import numpy as np
import pandas as pd
import pytest

from bacteria_analysis.chemical_features import build_chemical_class_rdms, build_chemical_rdm
from bacteria_analysis.io import AnalysisDataset


def _dataset(matrix, metadata, mapping=None):
    if mapping is None:
        mapping = pd.DataFrame(
            {
                "stimulus": ["odor_a", "odor_b", "odor_c"],
                "stim_name": ["a", "b", "c"],
                "sample_id": ["sample_a", "sample_b", "sample_c"],
            }
        )
    return AnalysisDataset(
        neural=pd.DataFrame(),
        matrix=matrix,
        metadata=metadata,
        stimulus_sample_map=mapping,
        included_dates=(),
        excluded_dates=(),
        parameters={},
    )


def _matrix():
    return pd.DataFrame(
        {
            "m1": [1.0, 2.0, 4.0],
            "m2": [4.0, 8.0, 16.0],
            "m3": [10.0, 10.0, 10.0],
            "m4": [5.0, 6.0, 7.0],
        },
        index=["sample_a", "sample_b", "sample_c"],
    )


def test_build_chemical_rdm_maps_samples_to_stimuli_and_applies_qc_log2():
    metadata = pd.DataFrame(
        {
            "metabolite_name": ["m1", "m2", "m3", "m4"],
            "QCRSD": [5, 20, 21, 40],
        }
    )

    result = build_chemical_rdm(_dataset(_matrix(), metadata), qc_threshold=0.2)

    assert result.matrix.index.tolist() == ["odor_a", "odor_b", "odor_c"]
    assert result.matrix.columns.tolist() == ["odor_a", "odor_b", "odor_c"]
    assert result.matrix.loc["odor_a", "odor_a"] == pytest.approx(0.0)
    assert result.matrix.loc["odor_a", "odor_b"] == pytest.approx(np.sqrt(2.0))
    assert result.matrix.loc["odor_b", "odor_a"] == pytest.approx(np.sqrt(2.0))
    assert result.metadata["retained_features"] == ("m1", "m2")
    assert result.metadata["feature_count"] == 2
    assert result.metadata["qc_threshold"] == 0.2
    assert result.metadata["transform"] == "log2"
    assert result.metadata["distance"] == "euclidean"
    assert result.metadata["qcrsd_filter_applied"] is True


def test_build_chemical_rdm_supports_common_metadata_spellings_and_correlation_distance():
    metadata = pd.DataFrame(
        {
            "name": ["m1", "m2", "m3"],
            "qcrsd": [0.1, 0.15, 0.1],
        }
    )

    result = build_chemical_rdm(
        _dataset(_matrix().loc[:, ["m1", "m2", "m3"]], metadata),
        qc_threshold=0.2,
        transform="none",
        distance="correlation",
    )

    assert result.metadata["retained_features"] == ("m1", "m2", "m3")
    assert result.matrix.loc["odor_a", "odor_a"] == pytest.approx(0.0)
    assert np.isfinite(result.matrix.loc["odor_a", "odor_b"])


def test_build_chemical_rdm_rejects_nonpositive_values_before_log_transform():
    matrix = _matrix()
    matrix.loc["sample_a", "m1"] = 0.0
    metadata = pd.DataFrame({"metabolite_name": ["m1", "m2"], "QCRSD": [0.1, 0.1]})

    with pytest.raises(ValueError, match="log2 transform requires positive values"):
        build_chemical_rdm(_dataset(matrix, metadata), qc_threshold=0.2, transform="log2")


def test_build_chemical_rdm_treats_present_missing_qcrsd_as_failed_qc():
    metadata = pd.DataFrame({"metabolite_name": ["m1", "m2"], "QCRSD": [np.nan, ""]})

    with pytest.raises(ValueError, match="no chemical features passed QC"):
        build_chemical_rdm(_dataset(_matrix(), metadata), qc_threshold=0.2)


def test_build_chemical_rdm_rejects_missing_sample_mapping():
    metadata = pd.DataFrame({"metabolite_name": ["m1"], "QCRSD": [0.1]})
    mapping = pd.DataFrame({"stimulus": ["odor_a"], "sample_id": ["missing_sample"]})

    with pytest.raises(ValueError, match="sample_ids missing from matrix"):
        build_chemical_rdm(_dataset(_matrix(), metadata, mapping=mapping))


def test_build_chemical_class_rdms_groups_by_taxonomy_and_filters_after_qc():
    metadata = pd.DataFrame(
        {
            "metabolite_name": ["m1", "m2", "m3", "m4"],
            "QCRSD": [0.1, 0.1, 0.3, 0.1],
            "Class": ["lipid", "lipid", "lipid", "unknown"],
        }
    )

    results = build_chemical_class_rdms(
        _dataset(_matrix(), metadata),
        taxonomy_level="Class",
        qc_threshold=0.2,
        min_features=2,
    )

    assert list(results) == ["lipid"]
    lipid = results["lipid"]
    assert lipid.metadata["taxonomy_level"] == "class"
    assert lipid.metadata["category"] == "lipid"
    assert lipid.metadata["retained_features"] == ("m1", "m2")
    assert lipid.metadata["feature_count"] == 2
    assert lipid.matrix.loc["odor_a", "odor_b"] == pytest.approx(np.sqrt(2.0))


def test_build_chemical_class_rdms_accepts_superclass_aliases_and_skips_small_categories():
    matrix = _matrix()
    metadata = pd.DataFrame(
        {
            "name": ["m1", "m2", "m3", "m4"],
            "qcrsd": [10, 10, 10, 10],
            "SuperClass": ["organic", "organic", "small", "small"],
        }
    )

    results = build_chemical_class_rdms(
        _dataset(matrix, metadata),
        taxonomy_level="super_class",
        qc_threshold=0.2,
        min_features=3,
    )

    assert results == {}


def test_build_chemical_class_rdms_requires_taxonomy_metadata():
    metadata = pd.DataFrame({"metabolite_name": ["m1"], "QCRSD": [0.1]})

    with pytest.raises(ValueError, match="taxonomy column"):
        build_chemical_class_rdms(_dataset(_matrix(), metadata), taxonomy_level="Class")
