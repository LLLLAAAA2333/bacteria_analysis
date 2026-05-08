import numpy as np
import pandas as pd

from bacteria_analysis.features.biological_subspace import (
    build_chemical_rdm,
    coerce_rdm_heatmap_frame,
    prepare_display_frames,
)


def test_build_chemical_rdm_uses_log2_euclidean_distance():
    matrix = pd.DataFrame(
        {
            "m1": [1.0, 2.0, 4.0],
            "m2": [8.0, 8.0, 8.0],
        },
        index=["sample_a", "sample_b", "sample_c"],
    )
    stimulus_sample_map = pd.DataFrame(
        {
            "stimulus": ["a", "b", "c"],
            "sample_id": ["sample_a", "sample_b", "sample_c"],
        }
    )

    rdm = build_chemical_rdm(matrix, stimulus_sample_map, ["m1", "m2"])
    square = coerce_rdm_heatmap_frame(rdm)

    assert square.loc["a", "b"] == 1.0
    assert square.loc["a", "c"] == 2.0
    assert square.loc["b", "c"] == 1.0
    assert np.diag(square.to_numpy(float)).tolist() == [0.0, 0.0, 0.0]


def test_prepare_display_frames_reuses_neural_order_for_model_rdms():
    neural = pd.DataFrame(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ],
        index=["s1", "s2", "s3"],
        columns=["s1", "s2", "s3"],
    )
    model = pd.DataFrame(
        {
            "stimulus_row": ["s3", "s2", "s1"],
            "s3": [0.0, 4.0, 5.0],
            "s2": [4.0, 0.0, 6.0],
            "s1": [5.0, 6.0, 0.0],
        }
    )
    stimulus_sample_map = pd.DataFrame(
        {
            "stimulus": ["s1", "s2", "s3"],
            "sample_id": ["sample_1", "sample_2", "sample_3"],
        }
    )

    order, displays = prepare_display_frames(neural, {"model": model}, stimulus_sample_map)

    assert set(order) == {"s1", "s2", "s3"}
    assert displays["model"].index.tolist() == displays["neural"].index.tolist()
    assert displays["model"].columns.tolist() == displays["neural"].columns.tolist()
    assert all(label.startswith("sample_") for label in displays["neural"].index)
