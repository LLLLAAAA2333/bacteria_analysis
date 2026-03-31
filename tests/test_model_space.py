import pytest

from bacteria_analysis.model_space import load_stimulus_sample_map, read_metabolite_matrix


def test_load_stimulus_sample_map_requires_unique_sample_ids(stage3_model_input_root):
    with pytest.raises(ValueError, match="sample_id values must be unique"):
        load_stimulus_sample_map(stage3_model_input_root / "duplicate_stimulus_sample_map.csv")


def test_read_metabolite_matrix_loads_expected_sample_ids(stage3_matrix_path):
    matrix = read_metabolite_matrix(stage3_matrix_path)
    assert matrix.index.tolist() == ["A001", "A002", "A003"]


def test_read_metabolite_matrix_rejects_blank_sample_id_cells(stage3_blank_matrix_path):
    with pytest.raises(ValueError, match="sample_id values must be non-empty"):
        read_metabolite_matrix(stage3_blank_matrix_path)
