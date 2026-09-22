# Restore the complete neural HMDS workflow

This backup preserves the notebook and helper module immediately before replacing the cached 2D result with a current-input fit.

Current notebook: cell 24 bootstrap -> cell 26 fresh 2D fit/check/save/plot -> cell 28 fresh 3D fit/check/save -> cell 30 current 3D plot.
No historical model is loaded by default. Original raw data and fitted-result directories were not modified.
Eight synthetic stimuli were used to execute the actual fitting/plotting cells with smaller computation budgets and pickle.load explicitly blocked. The complete real dataset was not run. Checks also covered rejection of missing current results and incomplete optimization.

Validation record: ../../tmp/jupyter-notebook/full_neural_workflow_20260922_144503/validation.json
