from bacteria_analysis import analysis_plotting
from bacteria_analysis import chemical_features
from bacteria_analysis import neural_features
from bacteria_analysis import plotting
from bacteria_analysis.features import anchor, biological_subspace, chemical, neural, taxonomy


def test_plotting_facade_reexports_current_helpers():
    assert plotting.create_rdm_panel_figure is analysis_plotting.create_rdm_panel_figure
    assert plotting.write_rdm_heatmap_pair is analysis_plotting.write_rdm_heatmap_pair


def test_feature_facades_reexport_current_helpers():
    assert chemical.build_chemical_rdm is chemical_features.build_chemical_rdm
    assert neural.build_neural_rdm is neural_features.build_neural_rdm
    assert anchor.merge_neurons.__name__ == "merge_neurons"
    assert biological_subspace.prepare_display_frames.__name__ == "prepare_display_frames"
    assert taxonomy.ClassCandidate.__name__ == "ClassCandidate"
