"""Focused saved-result arithmetic, source, labels, and overwrite checks."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd


def verify(result_dir):
    result=Path(result_dir).resolve();root=result.parents[2];tables=result/'tables'
    data=pd.read_csv(tables/'ordered_strains_and_thirds.csv',dtype={'dates':str}).set_index('strain')
    source=pd.read_csv(root/'reports/exploration_bacteroides_adf_ash_chemical_20261003/model/tables/selected_full_state_scores.csv',index_col='strain',dtype={'dates':str})
    assert data.index.tolist()==source.reset_index().sort_values(['chemical_state_score','strain']).strain.tolist()
    assert data.chemical_third.tolist()==['LOW']*10+['MID']*10+['HIGH']*9
    pd.testing.assert_frame_equal(data[source.columns],source.loc[data.index])
    target_rows=pd.concat([pd.read_csv(tables/'fixed_axis_associations.csv'),
                           pd.read_csv(tables/'unnormalized_adf_ash_descriptions.csv')])
    design=np.column_stack([np.ones(29),data.chemical_state_score])
    errors=[]
    for _,row in target_rows.iterrows():
        fit=np.linalg.lstsq(design,data[row.response].to_numpy(),rcond=None)[0]
        errors.extend(abs(fit-np.array([row.intercept,row.slope])))
    assert max(errors)<1e-12
    a=target_rows.set_index('response')
    assert abs(a.loc['unit_ADF','slope']-a.loc['unit_ASH','slope']-a.loc['primary_unit_adf_minus_ash','slope'])<1e-12
    assert abs(a.loc['raw_ADF','slope']-a.loc['raw_ASH','slope']-a.loc['raw_adf_minus_ash','slope'])<1e-12
    manifest=json.loads((result/'source_manifest.json').read_text())
    assert all(hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256'] for entry in manifest.values())
    before={str(p.relative_to(result)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [result/'summary.json',*sorted(tables.glob('*.csv'))]}
    for path in (result/'code').glob('*.py'):ast.parse(path.read_text())
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    module=load('guard_diagnostics',result/'code/fixed_axis_diagnostics.py')
    plot=load('guard_plot',result/'code/plot_fixed_axis.py')
    checks={}
    for label,fn in [('analysis_guard',lambda:module.run_diagnostics(root)),
                     ('raw_component_guard',lambda:module.add_unnormalized_coordinate_descriptions(result)),
                     ('plot_guard',lambda:plot.plot_saved(result))]:
        try:fn()
        except FileExistsError:checks[label]=True
        else:raise AssertionError(label)
    saved=module.load_saved(result);assert len(saved['ordered_strains'])==29
    after={str(p.relative_to(result)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [result/'summary.json',*sorted(tables.glob('*.csv'))]}
    assert after==before
    return {'status':'PASS','n_strains':29,'n_descriptive_responses':len(target_rows),
            'least_squares_coefficient_max_abs_error':float(max(errors)),
            'source_hashes_match':True,'labels_and_original_scores_match':True,
            'exact_10_10_9_chemical_order_verified':True,'ADF_minus_ASH_slope_identities':True,
            'all_python_syntax_valid':True,'read_only_load_valid':True,
            'scientific_outputs_unchanged':True,**checks}


if __name__=='__main__':
    result=Path(__file__).resolve().parents[1]
    checks=verify(result)
    (result/'verification/numeric_source_and_guard_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
