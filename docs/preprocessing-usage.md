# Preprocessing Usage

Run preprocessing from the repository root with:

```powershell
pixi run python scripts/run_preprocessing.py --input H:\Process_temporary\WJH\bacteria_analysis\data\data.parquet --output-root H:\Process_temporary\WJH\bacteria_analysis\data\processed
```

Generated artifacts:

- `H:\Process_temporary\WJH\bacteria_analysis\data\processed\clean\neuron_segments_clean.parquet`
- `H:\Process_temporary\WJH\bacteria_analysis\data\processed\trial_level\trial_metadata.parquet`
- `H:\Process_temporary\WJH\bacteria_analysis\data\processed\trial_level\trial_wide_baseline_centered.parquet`
- `H:\Process_temporary\WJH\bacteria_analysis\data\processed\trial_level\trial_tensor_baseline_centered.npz`
- `H:\Process_temporary\WJH\bacteria_analysis\data\processed\qc\preprocessing_report.json`
- `H:\Process_temporary\WJH\bacteria_analysis\data\processed\qc\preprocessing_report.md`

`trial_id` is the canonical trial key:

`trial_id = "{date}__{worm_key}__{segment_index}"`

Missing values in the wide and tensor outputs mean the measurement was not observed, not that it was zero-filled:

- unobserved neurons stay `NaN` across all their timepoints
- partially observed traces keep `NaN` at the missing timepoints
- fully-NaN traces are removed before writing outputs
