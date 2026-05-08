import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bacteria_analysis.analysis_results import AnalysisResult, save_analysis_result


def _rdm():
    return pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], index=["s1", "s2"], columns=["s1", "s2"])


def _figure():
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    return figure


def _result(source_path):
    return AnalysisResult(
        analysis_id="rdm_alignment",
        parameters={
            "neural_path": str(source_path),
            "seed": 7,
            "permutations": 99,
            "nested": {"subset_count": 5},
        },
        summary={"all_pairs_rsa": 0.5, "n_pairs_all": 1},
        tables={"pair_summary": pd.DataFrame({"scope": ["all"], "n_pairs": [1]})},
        rdms={"neural": _rdm(), "chemical": _rdm()},
        figures={"overview": _figure()},
        audit={
            "aligned_stimulus_order": ["s1", "s2"],
            "retained_features": ["m1", "m2"],
            "n_pairs_by_scope": {"all": 1},
            "date_coverage": pd.DataFrame({"date": ["20260401"], "n_stimuli": [2]}),
        },
        diagnostics={"caveat": "cross-date RSA is descriptive"},
        debug_tables={"pair_values": pd.DataFrame({"stimulus_left": ["s1"], "stimulus_right": ["s2"]})},
    )


def test_analysis_result_stores_all_artifact_groups(tmp_path):
    source_path = tmp_path / "raw.csv"
    result = _result(source_path)

    assert result.parameters["seed"] == 7
    assert result.summary["all_pairs_rsa"] == 0.5
    assert "pair_summary" in result.tables
    assert "neural" in result.rdms
    assert "overview" in result.figures
    assert result.audit["aligned_stimulus_order"] == ["s1", "s2"]
    assert result.diagnostics["caveat"] == "cross-date RSA is descriptive"
    assert "pair_values" in result.debug_tables


def test_analysis_result_constructor_does_not_write_files(tmp_path):
    output_root = tmp_path / "not_written"

    _result(tmp_path / "raw.csv")

    assert not output_root.exists()


def test_save_analysis_result_writes_final_artifacts_without_debug_by_default(tmp_path):
    source_path = tmp_path / "raw.csv"
    source_path.write_text("sample,value\ns1,1\n", encoding="utf-8")
    output_root = tmp_path / "result"

    written = save_analysis_result(_result(source_path), output_root)

    assert (output_root / "summary.json").exists()
    assert (output_root / "summary.md").exists()
    assert (output_root / "parameters.json").exists()
    assert (output_root / "diagnostics.json").exists()
    assert (output_root / "tables" / "pair_summary.csv").exists()
    assert (output_root / "rdms" / "neural.csv").exists()
    assert (output_root / "rdms" / "chemical.csv").exists()
    assert (output_root / "figures" / "overview.png").exists()
    assert not (output_root / "audit").exists()
    assert not (output_root / "debug").exists()
    assert "debug.pair_values" not in written

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_pairs_all"] == 1
    diagnostics = json.loads((output_root / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["caveat"] == "cross-date RSA is descriptive"


def test_save_analysis_result_writes_audit_only_when_requested(tmp_path):
    source_path = tmp_path / "raw.csv"
    source_path.write_text("sample,value\ns1,1\n", encoding="utf-8")
    output_root = tmp_path / "result"

    written = save_analysis_result(_result(source_path), output_root, include_audit=True)

    assert (output_root / "audit" / "aligned_stimulus_order.json").exists()
    assert (output_root / "audit" / "retained_features.json").exists()
    assert (output_root / "audit" / "date_coverage.csv").exists()
    assert (output_root / "audit" / "source_manifest.json").exists()
    assert written["audit.source_manifest"] == output_root / "audit" / "source_manifest.json"

    manifest = json.loads((output_root / "audit" / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_id"] == "rdm_alignment"
    assert manifest["seeds"] == {"seed": 7}
    assert manifest["permutation_counts"] == {"permutations": 99}
    assert manifest["resampling_counts"] == {"nested.subset_count": 5}
    assert manifest["source_paths"][0]["parameter"] == "neural_path"
    assert manifest["source_paths"][0]["exists"] is True
    assert "sha256" in manifest["source_paths"][0]


def test_save_analysis_result_writes_debug_tables_when_requested(tmp_path):
    source_path = tmp_path / "raw.csv"
    source_path.write_text("sample,value\ns1,1\n", encoding="utf-8")
    output_root = tmp_path / "result"

    written = save_analysis_result(_result(source_path), output_root, include_debug=True)

    assert (output_root / "debug" / "pair_values.csv").exists()
    assert written["debug.pair_values"] == output_root / "debug" / "pair_values.csv"


def test_save_analysis_result_accepts_figure_writer_with_png_name(tmp_path):
    def write_figure(path):
        path.write_bytes(b"figure")

    result = AnalysisResult(
        analysis_id="writer",
        parameters={},
        summary={},
        figures={"custom_panel.png": write_figure},
    )

    written = save_analysis_result(result, tmp_path / "result")

    assert (tmp_path / "result" / "figures" / "custom_panel.png").read_bytes() == b"figure"
    assert not (tmp_path / "result" / "figures" / "custom_panel.png.png").exists()
    assert written["figures.custom_panel.png"] == tmp_path / "result" / "figures" / "custom_panel.png"


def test_save_analysis_result_accepts_dataframe_summary(tmp_path):
    result = AnalysisResult(
        analysis_id="class_rsa",
        parameters={},
        summary=pd.DataFrame({"class": ["lipid"], "score": [0.8]}),
    )

    save_analysis_result(result, tmp_path / "result")

    summary = json.loads((tmp_path / "result" / "summary.json").read_text(encoding="utf-8"))
    assert summary["records"] == [{"class": "lipid", "score": 0.8}]
