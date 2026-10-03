# %% Shared coverage only. Analysis and ordering are independent across modalities.
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

ROOT = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis')
OUT = ROOT / 'reports/exploration_genus_patterns_independent_20261003'
display(pd.read_csv(OUT / 'shared_genus_coverage.csv'))

# %% Step 3: chemical-only discovery and ordering. This loads a saved figure.
display(Image(filename=str(OUT / 'chemical/figures/01_chemical_module_centers.png')))

# %% Step 4: neural-only profiles and ordering. This loads a saved figure.
display(Image(filename=str(OUT / 'neural/figures/01_neural_genus_centered_profiles.png')))

# %% Supporting chemical individual-strain view; no neural ordering is used.
display(Image(filename=str(OUT / 'chemical/figures/02_chemical_modules_all_strains.png')))

# %% Supporting neural individual-strain view; no chemical ordering is used.
display(Image(filename=str(OUT / 'neural/figures/02_neural_all_strains_centered_profiles.png')))

# %% Whole-neural-combination leave-one-strain-out alignment, descriptive only.
display(Image(filename=str(OUT / 'neural/figures/03_neural_leave_one_strain_out.png')))

# Detailed methods, tables and optional analysis replay are in each branch README.
# This file does not rerun clustering, refit neural templates or test correspondence.
