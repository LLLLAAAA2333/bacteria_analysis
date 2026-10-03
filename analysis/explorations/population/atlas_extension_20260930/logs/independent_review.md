# Independent read-only review of targeted chemical analyses

Reviewed by atlas_roles agent, independently of the author of code 04–07. Code and existing outputs were read; no analysis code or results were modified. Focused calculations were actually executed with the project Python environment. This is a computational/methodological review, not independent biological validation.

## Judgment

No high-priority implementation defect was found in `04_targeted_chemistry.py`, `05_chemical_checks.py`, `06_within_genus_chemistry.py` or `07_population_bridge.py`. The saved predictions, split logic, weights, common-cell target construction, and chemical bridge membership agree with independent calculations below. The scientific claims still need to respect selection, support changes, and the distinction between cross-block prediction and strain/date identifiability.

## Actual checks and results

1. **Outer splits and repeated identities.** Verified that all 9 outer test blocks share neither sample identity nor `(date,worm_key)` animal identity with training. All six repeated strains are absent from targeted predictions, leaving 100 independent sample IDs. Prediction rows are unique by model × neuron × phase × animal × strain.

2. **Nested selection.** Read all 56,160 inner risk rows, verified outer block differs from inner validation block, and independently averaged the inner MSE per candidate penalty. All 1,404 saved model × neuron × phase × outer-block penalties equal the independent minimizing choice. Code inspection confirms training excludes both outer and inner validation blocks when estimating scaling, imputation and coefficients.

3. **Numerical fit replication.** Recomputed three held-out predictions with ordinary weighted normal-equation solves, independently of the eigendecomposition fitter: ASI/post/panel162 in 20260311; ASJ/post/relative162 in 20260414; ASK/stim/panel248 in 20260609. Maximum difference from saved predictions was **1.21e−16 dF/F0**. These are focused algorithm checks, not a redundant full rerun.

4. **Shared-animal handling and biological weights.** Every strain × cell × block has total animal weight 1; maximum rounding deviation **4.44e−16**. Animal-centering is done before regression, and complete blocks of animals are left out in outer and inner prediction. The remaining within-training animal correlation is not explicitly a covariance model; that does not invalidate the descriptive held-out prediction task, but no OLS-like inferential independence or significance should be claimed.

5. **ASI/ASJ common-measurement fairness.** Independently formed the physical-unit mean of raw ASI and ASJ only where both were measured, then centered on the same animal's measured stimulus set. This matched `chemical_checks_axis_predictions.csv` with maximum error **1.87e−16**. Coverage is **40 animals**, 100 strains, 439 co-measured animal×strain units (878 rows when counting stim and post). Averaging separately centered source channels happens to equal centering their co-measured average in the current data; the check establishes this rather than assuming it.

6. **Same-genus centering.** Independently re-centered saved target and prediction within model × neuron × phase × animal × genus, requiring ≥2 strains, and re-averaged by strain. Maximum discrepancy with code 05 was **2.22e−16**. For code 06, independently constructed raw ASK and the co-measured ASI/ASJ mean, then centered within animal × genus; maximum target discrepancy **9.97e−17**. Weight sums again differed from 1 by at most **4.44e−16**. The same-genus models therefore genuinely train on within-genus variation, rather than retaining a genus intercept and relabeling it.

7. **Bridge membership and visible contrasts.** The full matched neural response figure uses **32 animals**; chemical bridge filtering retains **16 animals**, 4 blocks, **33 distinct strains**. For every retained animal, both groups contain exactly 3 distinct strains, all 5 neurons are jointly observed, no repeated strain remains, and all six chemical models use the same observed contrast. Membership is exactly the corresponding subset of the earlier neural selection: no chemical-output-dependent reselection occurred. Independent high-minus-low differences reconstructed directly from raw animal post-response means match bridge observations with maximum error **2.78e−16**.

8. **No hidden independent-validation claim in code.** Targeted prediction methods call this exploratory reuse; code 06 is explicitly a bounded response to a failed within-genus check; code 07 calls the ASI/ASJ target data-discovered and the displayed uncertainty conditional/descriptive. This wording must also be retained in the reports.

## Interpretation issues to enforce in the report

- **Do not merge support sets.** The ASI/ASJ prediction score uses 40 animals/100 strains; the main matched neural curves use 32 animals; chemical reconstruction of the same five-cell contrast uses a selected 16-animal/33-strain subset. The bridge cannot be described as explaining all 32 animals or the full atlas. These are not merely different display sample counts: exclusion depends on the selected strains and cell availability.
- **`relative162` is not a chemistry-total-free model.** It retains median chemical-level covariates and appends features minus median standardized intensity. Relative and raw panel designs span essentially the same covariate space, with different ridge parameterization. This is a sensitivity analysis, not an independent proof that only chemical composition matters. The defensible information comparison is panel prediction against the level-only baseline, subject to held-out performance and alternatives.
- **Post hoc phenotype discovery remains post hoc.** ASI/ASJ and the post window were selected after a 13-cell/78-pair/three-phase neural screen, and subsequent chemical routes were inspected in this same dataset. Holding out animals within the neural contrast and entire blocks in chemical models does not undo that phenotype/model-family selection. Use “discovery,” “internal check,” and “held-out prediction” precisely; do not call it independent validation or attach confirmatory p values.
- **Zero-relative predictive R².** These scores compare predictions of animal-centered responses against zero relative response, often after strain averaging. They do not quantify explained raw fluorescence variance, absolute response to one isolated exposure, or causal chemical effect.
- **Same-genus failure has a specific scope.** Negative same-genus R², including in code 06's retraining, limits the proposed bridge. It does not establish absence of chemical effects, and it does not by itself condemn the neural atlas's reproducibility.
- **Uncertainty definitions must remain narrow.** Block/strain deletion ranges are sensitivity ranges without refitting. Bridge bootstrap intervals resample animals within existing blocks with selections and predictions fixed; they do not include target selection, model selection or new-block uncertainty.

## Review execution note

The first focused audit command stopped during a reviewer-only pandas assertion because GroupBy `.observed` accesses a Boolean option; the corrected check used `['observed']` and passed. This was an audit-script accessor mistake, not a project-analysis failure. All substantive checks listed above executed successfully after that correction. No source or result file was changed by the review.

## REVIEW.md / REPORT.md review

Read both reports and checked the main numerical claims against their saved tables. The following matched: 32-animal five-cell means and directional counts; ASI/ASJ cross-animal transfer counts and median correlations; deleted-strain correlation checks; 19 genera in the matched higher group; four-cell classification and within-genus increments; chemical prediction and bridge values; and AWB, ASER, AWCOFF and ASEL threshold counts. The displayed main neural figure was also inspected: it uses a common y-axis, visible individual curves, and no date labels.

No additional material numerical error was found beyond corrections already identified by the author (figure filename and explicit support/class-model details). Two interpretation edits were sent to the author:

- Qualify the loss of ASI later-trial prediction as **post response**. Panel162's later/post R² is −0.00631, but later/stim is +0.01731; the unqualified sentence would erase that distinction.
- Replace “failure is not only caused by between-genus variation dominating the training target” with “training directly on within-genus contrasts still did not produce transferable prediction.” The retraining result rules out this particular rescue, but does not identify the original failure's cause; reduced support or predictor variation can also limit the retrained fit.

These are scientific-interpretation refinements; no rerun of code 04 is needed. The report appropriately keeps all-data phenotype discovery separate from animal-held-out and block-held-out internal evaluation, and does not call the resulting chemical association a molecular mechanism.


## 主进程已落实的报告修订

主图链接已改为实际生成路径；补明32动物/63株/8块与16动物/33株/4块的不同支持集合。relative162明确保留水平项。ASI后续trial的预测失败已限定post；直接属内训练失败只表示仍未得到泛化证据，不识别原模型失败的原因。各项修改不改变模型或关键数值。
