# Draw the 12-group phylogenetic partition; no embedding fitting or bootstrap.
from pathlib import Path
from datetime import datetime
import importlib
import json
import sys
import matplotlib.pyplot as plt
from IPython.display import display

color_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phylogenetic_colors
import plot_phylogenetic_colors
importlib.reload(phylogenetic_colors)
importlib.reload(plot_phylogenetic_colors)
from phylogenetic_colors import load_phylogenetic_colors, mpl_colors, plotly_colors, mpl_colorbar_kwargs

# The shared default is PHYLOGENETIC_N_GROUPS = 12 in phylogenetic_colors.py.
# All embeddings use this same partition of the FULL tree, aligned by AID.
phylo_colors = load_phylogenetic_colors(color_root / 'data/16S.aln.trim.fa.treefile')
phylo_group_figure, phylo_full_tree_figure, phylo_group_members = (
    plot_phylogenetic_colors.plot_phylogenetic_groups(phylo_colors)
)
phylo_group_output = color_root / 'output/jupyter-notebook' / (
    '16s_phylogenetic_groups_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
)
phylo_group_output.mkdir(parents=True, exist_ok=False)
phylo_group_figure.savefig(phylo_group_output / 'group_overview.png', dpi=180)
phylo_group_figure.savefig(phylo_group_output / 'group_overview.svg')
phylo_full_tree_figure.savefig(phylo_group_output / 'full_tree_all_aids.png', dpi=140)
phylo_full_tree_figure.savefig(phylo_group_output / 'full_tree_all_aids.svg')
phylo_group_members.to_csv(phylo_group_output / 'aid_to_group.csv', encoding='utf-8-sig')
phylo_colors['group_table'].to_csv(phylo_group_output / 'group_summary.csv', encoding='utf-8-sig')
phylo_colors['cut_table'].to_csv(phylo_group_output / 'cut_branches.csv', index=False)
(phylo_group_output / 'color_parameters.json').write_text(
    json.dumps(phylogenetic_colors.color_provenance(phylo_colors), indent=2), encoding='utf-8'
)
for filename in ('phylogenetic_colors.py', 'plot_phylogenetic_colors.py'):
    (phylo_group_output / filename.replace('.py', '_used.py')).write_bytes(
        (color_root / 'notebook' / filename).read_bytes()
    )
display(phylo_colors['group_table'])
plt.show()
print(phylo_colors['note'])
print('Group IDs are categorical; connected groups are not asserted taxonomic or monophyletic units.')
print('Cut lengths and input node labels are saved for inspection; groups are not support-filtered.')
print('Saved:', phylo_group_output)
