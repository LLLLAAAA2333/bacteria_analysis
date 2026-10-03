"""Refresh only the requested figure layouts from saved analysis and HMDS fits."""
from pathlib import Path
import hashlib
import json

from individual_profile_display import plot_profile_and_model
from individual_comparison_display import plot_comparisons
from individual_repeatability_display import plot_repeatability


def refresh_individual_figures(output_dir):
    out = Path(output_dir).resolve()
    folder = out / "figures"
    plot_profile_and_model(out)
    repeatability = plot_repeatability(out)
    comparisons = plot_comparisons(out, scope="all")
    # Keep unchanged filtering/repeatability captions, replacing revised panels.
    replaced = ("01_response_profile_5bin:", "01b_response_model:",
                "02_repeatability_distribution:",
                "Saved individual-SNR ≥",
                "03_neural_chemical_rdm:", "03b_neural_chemical_hmds",
                "03c_neural_chemical_hmds")
    old = (folder / "captions.txt").read_text().split("\n\n")
    kept = [p.strip() for p in old if p.strip() and not p.strip().startswith(replaced)]
    repeatability_caption = (folder / "repeatability_caption.txt").read_text().strip()
    if not repeatability_caption.startswith("02_repeatability_distribution:"):
        repeatability_caption = "02_repeatability_distribution: " + repeatability_caption
    updated = [*(folder / "profile_model_captions.txt").read_text().strip().split("\n\n"),
               *kept, repeatability_caption,
               (folder / "comparison_captions.txt").read_text().strip()]
    (folder / "captions.txt").write_text("\n\n".join(updated)+"\n")
    settings = json.loads((folder / "plot_parameters.json").read_text())
    settings.update(response_colormap="RdBu_r", rdm_colormap="RdBu_r", embedding_point_colormap="turbo",
                    profile_layout="Notebook 02 Panel A; templates shown in separate model figure",
                    model_figure="01b_response_model", hmds_dimensions=[2, 3],
                    hmds_shepard=True, hmds_scope="all_samples", repeatability_layout="Notebook short-wide",
                    figure_revision_parameters=["profile_display_parameters.json",
                                                "comparison_display_parameters.json",
                                                "repeatability_display_parameters.json"])
    (folder / "plot_parameters.json").write_text(json.dumps(settings, indent=2)+"\n")
    baseline = out / "full106_revision_input_hashes.json"
    if not baseline.exists():
        baseline = out / "figure_revision_input_hashes.json"
    unchanged = {}
    if baseline.exists():
        unchanged = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
                     for p, digest in json.loads(baseline.read_text()).items()}
        if not all(unchanged.values()):
            raise RuntimeError("An existing analysis input, 2D fit or original notebook changed")
    files = [folder / "profile_display_parameters.json", folder / "comparison_display_parameters.json",
             folder / "repeatability_display_parameters.json",
             folder / "captions.txt", *Path(__file__).parent.glob("*.py")]
    n_hmds = next(f["n_samples"] for f in comparisons["figures"] if f.get("dimension") == 2)
    verification = dict(unchanged_existing_inputs=unchanged,
                        reused_response_analysis=True, display_only_refresh=True,
                        hmds_fit_source="hmds_full106; both dimensionalities checked separately",
                        new_fit=f"Neural 2D and 3D HMDS on all {n_hmds} samples",
                        n_samples_hmds=n_hmds,
                        repeatability_checks=repeatability["checks"],
                        source_and_output_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in files},
                        figures=[f["filename"] for f in comparisons["figures"]])
    (out / "figure_revision_verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    return verification


if __name__ == "__main__":
    print(json.dumps(refresh_individual_figures(Path(__file__).resolve().parents[1]), indent=2))
