from bacteria_analysis import analysis_dataset
from bacteria_analysis import analysis_results
from bacteria_analysis import io
from bacteria_analysis.analyses import rdm
from bacteria_analysis.features import anchor, chemical, neural, taxonomy


def test_rdm_package_reexports_current_helpers():
    assert rdm.build_chemical_rdm.__name__ == "build_chemical_rdm"
    assert rdm.build_neural_rdm.__name__ == "build_neural_rdm"
    assert rdm.run_rdm_alignment.__name__ == "run_rdm_alignment"


def test_feature_facades_reexport_current_helpers():
    assert chemical.build_chemical_feature_matrix.__name__ == "build_chemical_feature_matrix"
    assert neural.build_trial_feature_matrix.__name__ == "build_trial_feature_matrix"
    assert anchor.merge_neurons.__name__ == "merge_neurons"
    assert taxonomy.ClassCandidate.__name__ == "ClassCandidate"


def test_io_is_public_analysis_io_boundary():
    assert io.AnalysisDataset is analysis_dataset.AnalysisDataset
    assert io.save_analysis_result is analysis_results.save_analysis_result
