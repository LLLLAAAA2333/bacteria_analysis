# %% Read saved results in an existing notebook; no fitting or file writes.
from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

report = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_adf_ash_chemical_20261003')
model_summary = json.loads((report / 'model/summary.json').read_text())
display(model_summary)

# %% All figures preserve the fixed primary target and the full 29-strain cohort.
# Consult diagnostics/figures/CAPTIONS.md for units and interpretation limits.
for figure in sorted((report / 'diagnostics/figures').glob('*.png')):
    display(Image(filename=str(figure)))

# %% Inspect available numeric tables without refitting/selecting a new model.
table_paths = sorted((report / 'model/tables').glob('*.csv'))
table_paths += sorted((report / 'diagnostics/tables').glob('*.csv'))
display(pd.DataFrame({'saved_table': [str(path.relative_to(report)) for path in table_paths]}))
# To read one table, use: table = pd.read_csv(report / 'model/tables/<name>.csv')
