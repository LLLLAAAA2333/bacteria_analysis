"""Explain the SNR change using existing condition summaries; no refitting.

Only two audit outputs are written in this exploration. Both source reports
remain unchanged. Rows are strain × recording date × cell, not independent
biological replicates.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1]
REPORTS = OUT.parent
KEYS = ["strain", "block", "cell"]


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def quantiles(values):
    probabilities = [0., .1, .25, .5, .75, .9, 1.]
    return {f"{p:g}": float(v) for p, v in zip(probabilities, np.quantile(values, probabilities))}


def main():
    paths = {
        "individual": REPORTS / "exploration_response_profiles_20261001/tables/condition_metrics.csv",
        "trial": REPORTS / "exploration_response_profiles_trial_snr_20261001/tables/condition_metrics.csv",
    }
    inputs = {name: pd.read_csv(path, dtype={"block": str}).set_index(KEYS).sort_index()
              for name, path in paths.items()}
    animal, trial = inputs["individual"], inputs["trial"]
    if (not animal.index.is_unique or not trial.index.is_unique
            or not animal.index.equals(trial.index) or len(animal) != 1456):
        raise ValueError("Expected the same 1,456 unique condition × cell records")
    shared = ["signal_power", "mean_rms", "baseline_snr", "n_animals"]
    shared_errors = {name: float((animal[name] - trial[name]).abs().max()) for name in shared}
    if any(value != 0. for value in shared_errors.values()):
        raise ValueError("The two cached definitions no longer share identical signal means/coverage")
    if not animal.n_animals.ge(2).all():
        raise ValueError("This fixed-coverage audit requires at least two animals in every matched row")

    n = animal.n_animals.to_numpy()
    vi, vt = animal.scatter_power.to_numpy(), trial.scatter_power.to_numpy()
    p = animal.signal_power.to_numpy()
    q = trial.weight_sum_squared.to_numpy()
    ai, st = animal.snr.to_numpy(), trial.snr.to_numpy()
    between = (n - 1.) / n * vi
    within = (1. - q) * vt - between
    fraction = within / (within + between)
    if np.any(within < -1e-12):
        raise ValueError("Negative within-animal trial power beyond numerical tolerance")
    individual_pass, trial_pass, individual_half = ai >= 1., st >= 1., ai >= .5
    table = animal.index.to_frame(index=False)
    values = dict(
        n_animals=n, n_trials=trial.n_trials.to_numpy(), effective_n_trial=trial.effective_n.to_numpy(),
        signal_power_individual=p, signal_power_trial=trial.signal_power.to_numpy(),
        scatter_power_individual=vi, scatter_power_trial=vt,
        snr_individual=ai, snr_trial=st, variance_ratio_trial_over_individual=vt / vi,
        between_animal_population_power=between, within_animal_trial_population_power=within,
        within_trial_fraction=fraction, trial_weight_sum_squared=q,
        squared_snr_variance_change=p / vt - p / vi,
        squared_snr_bias_correction_change=1. / n - q,
        pass_individual_1=individual_pass, pass_individual_0_5=individual_half,
        pass_trial_1=trial_pass, eligible_old_minimum_3=n >= 3,
        saved_individual_status=animal.status.to_numpy(), saved_trial_status=trial.status.to_numpy(),
    )
    for name, value in values.items():
        table[name] = value

    counts = {"individual_1": int(individual_pass.sum()),
              "individual_0_5": int(individual_half.sum()), "trial_1": int(trial_pass.sum())}
    summary = dict(
        analysis="Read-only arithmetic comparison of cached SNR definitions; no fitting or threshold selection",
        input_sha256={str(path): sha256(path) for path in paths.values()}, code_sha256=sha256(__file__),
        matched_records=len(table), row_definition="strain × recording date × cell",
        common_min_animals=2, shared_fields_max_abs_difference=shared_errors,
        fixed_coverage_pass_counts=counts,
        fixed_coverage_pass_percent={name: 100. * count / len(table) for name, count in counts.items()},
        threshold_1_transitions=dict(
            both_pass=int((individual_pass & trial_pass).sum()),
            individual_only=int((individual_pass & ~trial_pass).sum()),
            trial_only=int((~individual_pass & trial_pass).sum()),
            both_fail=int((~individual_pass & ~trial_pass).sum())),
        individual_half_vs_trial_one=dict(
            both_pass=int((individual_half & trial_pass).sum()),
            individual_half_only=int((individual_half & ~trial_pass).sum()),
            trial_one_only=int((~individual_half & trial_pass).sum()),
            both_fail=int((~individual_half & ~trial_pass).sum())),
        coverage_confound=dict(
            saved_individual_min_animals=3, saved_trial_min_animals=2,
            saved_individual_retained=int(animal.status.eq("retained").sum()),
            saved_trial_retained=int(trial.status.eq("retained").sum()),
            old_limited_records=int(animal.status.eq("limited_n").sum()),
            two_animal_records=int((n == 2).sum()),
            two_animal_individual_1_pass=int(((n == 2) & individual_pass).sum()),
            two_animal_individual_half_pass=int(((n == 2) & individual_half).sum()),
            individual_half_with_minimum_3=int(((n >= 3) & individual_half).sum()),
            explanation="Of 39 records excluded by the previous three-animal requirement, 20 pass individual SNR≥1 and 29 pass SNR≥0.5 when coverage is restored to two animals."),
        variance_ratio_quantiles=quantiles(vt / vi), individual_snr_quantiles=quantiles(ai),
        trial_snr_quantiles=quantiles(st), effective_trial_n_quantiles=quantiles(trial.effective_n),
        trial_variance_greater_count=int((vt > vi).sum()), trial_snr_lower_count=int((st < ai).sum()),
        decomposition=dict(
            formula="B=(m−1)/m * V_individual; W=(1−sw2)*V_trial−B; fraction=W/(B+W)",
            interpretation="B is population scatter of animal means; W is the animal-equal average of within-animal trial population variance",
            within_power_min=float(within.min()), within_fraction_quantiles=quantiles(fraction),
            within_fraction_above_half_count=int((fraction > .5).sum()),
            individual_pass_trial_fail_within_fraction_quantiles=quantiles(fraction[individual_pass & ~trial_pass]),
            variance_term_change_quantiles=quantiles(p / vt - p / vi),
            bias_correction_change_quantiles=quantiles(1. / n - q)),
        formula_max_abs_error=dict(
            individual=float(np.max(np.abs(ai**2 - np.maximum(p / vi - 1. / n, 0.)))),
            trial=float(np.max(np.abs(st**2 - np.maximum(p / vt - q, 0.))))),
        interpretation="The signal means are identical. The extra trial scatter increases the variance denominator; the smaller trial bias-correction term works in the opposite direction. Trial scatter is not assumed to be pure recording noise.",
        limitations=["Effective trial n describes weights, not independent biological replication.",
                     "Trial-to-trial differences can include order effects, adaptation and drift.",
                     "SNR cutoffs are exploratory, not significance levels.",
                     "Matched condition-cell rows and pairwise transitions are descriptive counts, not independent replicates."])
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "tables/snr_definition_comparison.csv", index=False)
    (OUT / "snr_definition_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"records": len(table), "fixed_coverage_pass_counts": counts,
                      "shared_fields_max_abs_difference": shared_errors}, indent=2))


if __name__ == "__main__":
    main()
