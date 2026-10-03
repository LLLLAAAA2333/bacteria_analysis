# Interpretation clarifications after numerical review

The fixed scientific analysis is unchanged. `PROTOCOL.md` and `model/protocol_snapshot.md` preserve the original plan and SHA-256. These notes document interpretation clarifications made after reviewing numerical outputs, not new prespecified decisions.

- Rank-third allocation uses chemical-score rank only, conditional on the selected full-cohort axis. That axis itself was selected using the primary neural response. The thirds therefore are not outcome-independent evidence from the entire discovery procedure, and are not independent validation. Captions and reports make this distinction explicit.
- The additional raw ADF and raw ASH separate-component slopes are descriptive normalization checks using the same selected full-cohort chemical axis. They do not involve a new target/state search or constitute separately validated discoveries. The original raw ADF−ASH contrast remains the prespecified representation sensitivity.
- Pooled prediction improvement is reduction in squared error relative to training-fold-mean predictions, not percent RMSE reduction or total neural variance explained.
- Whole-profile prediction uses the same primary-selected chemical score and 13 separately trained intercept/slope pairs; predicted vectors are not renormalized. Its score is not interchangeable with that of the prior PC1-to-vector model.
- Within-date-set demeaning is a limited description, not a complete date/batch adjustment. Fixed-axis single-deletion direction stability is not a claim that the entire discovery process is insensitive to all possible exclusions.

No alternative target, extra chemical candidate family, individual-compound search, nonlinear model, new cutoff, or additional strain exclusion was introduced.
