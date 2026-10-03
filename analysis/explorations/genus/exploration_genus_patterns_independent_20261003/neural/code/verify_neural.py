"""Independent numeric checks from original CSVs, without importing analysis code."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, label, errors):
    actual=np.asarray(actual,dtype=float);expected=np.asarray(expected,dtype=float)
    err=float(np.max(np.abs(actual-expected))) if actual.size else 0.0
    if not np.allclose(actual,expected,rtol=1e-10,atol=1e-10,equal_nan=True):
        raise AssertionError(f'{label}: {err}')
    errors[label]=max(errors.get(label,0),err)


def vector_cos(a,b):
    return np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b))


def verify(repo_root):
    root=Path(repo_root)
    out=root/'reports/exploration_genus_patterns_independent_20261003/neural'
    src=root/'reports/exploration_chemical_pattern_direct_report_20261003/tables'
    manifest=json.loads((out/'source_manifest.json').read_text())
    for item in manifest.values():
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
    raw=read_rows(src/'neural_unit_coefficients.csv');cells=list(raw[0])[1:]
    vectors={r['strain']:np.array([float(r[c]) for c in cells]) for r in raw}
    pregate={r['strain']:np.array([float(r[c]) for c in cells]) for r in read_rows(src/'neural_pre_gate_unit_coefficients.csv')}
    groups={}
    for r in read_rows(src/'sample_context.csv'):
        groups.setdefault(r['genus'],[]).append(r['strain'])
    groups={g:sorted(ss) for g,ss in groups.items() if len(ss)>=2}
    names=sorted(groups);assert len(names)==13 and sum(map(len,groups.values()))==90
    members=read_rows(out/'tables/main_membership.csv')
    assert {(r['strain'],r['genus']) for r in members}=={(s,g) for g in names for s in groups[g]}
    assert len(read_rows(out/'tables/singleton_coverage_only.csv'))==16
    means={g:np.array([vectors[s] for s in groups[g]]).mean(axis=0) for g in names}
    ref=np.array([means[g] for g in names]).mean(axis=0)
    deltas={g:means[g]-ref for g in names}
    pre_means={g:np.array([pregate[s] for s in groups[g]]).mean(axis=0) for g in names}
    pre_ref=np.array([pre_means[g] for g in names]).mean(axis=0)
    pre_deltas={g:pre_means[g]-pre_ref for g in names}
    errors={}
    close([np.linalg.norm(v) for v in vectors.values()],1,'input unit norms',errors)
    close([float(r['reference']) for r in read_rows(out/'tables/equal_genus_reference.csv')],ref,'reference',errors)
    for filename,expected in [('genus_mean_unit_profiles',means),('genus_centered_profiles',deltas),('pre_gate_genus_mean_unit_profiles',pre_means),('pre_gate_genus_centered_profiles',pre_deltas)]:
        for r in read_rows(out/'tables'/f'{filename}.csv'):
            close([r[c] for c in cells],expected[r['genus']],filename,errors)
    close(np.mean(list(deltas.values()),axis=0),0,'equal-genus zero centering',errors)
    for r in read_rows(out/'tables/genus_neuron_summary.csv'):
        g=r['genus'];j=cells.index(r['neuron']);a=np.array([vectors[s][j] for s in groups[g]])
        close(r['centered_mean'],a.mean()-ref[j],'cell centered mean',errors)
        close(r['same_sign_count'],np.sum(np.sign(a-ref[j])==np.sign(deltas[g][j])),'cell same-sign count',errors)
        close(r['same_sign_fraction'],np.mean(np.sign(a-ref[j])==np.sign(deltas[g][j])),'cell same-sign fraction',errors)
        close(r['sd_unit'],np.std(a,ddof=1),'cell sd',errors)
        for q in [.1,.25,.75,.9]:
            close(r[f'q{int(q*100)}_unit'],np.quantile(a,q),'cell quantiles',errors)
    heldout_by_genus={g:[] for g in names}
    for r in read_rows(out/'tables/leave_one_strain_out.csv'):
        g=r['genus'];s=r['strain'];remaining=[vectors[t] for t in groups[g] if t!=s]
        loo_means={**means,g:np.mean(remaining,axis=0)}
        loo_ref=np.mean([loo_means[t] for t in names],axis=0)
        loo_delta=loo_means[g]-loo_ref
        heldout=vectors[s]-loo_ref
        alignment=vector_cos(heldout,loo_delta)
        heldout_by_genus[g].append(alignment)
        close(r['center_direction_cosine'],vector_cos(deltas[g],loo_delta),'LOSO center cosine',errors)
        close(r['heldout_direction_cosine'],alignment,'LOSO heldout cosine',errors)
        close(r['center_l2_change'],np.linalg.norm(deltas[g]-loo_delta),'LOSO center change',errors)
    for r in read_rows(out/'tables/genus_summary.csv'):
        g=r['genus'];a=np.array([vectors[s] for s in groups[g]])
        dispersion=np.sqrt(np.mean(np.sum((a-means[g])**2,axis=1)))
        close(r['mean_profile_norm'],np.linalg.norm(means[g]),'mean concentration',errors)
        close(r['within_genus_rms_dispersion'],dispersion,'within dispersion',errors)
        close(float(r['mean_profile_norm'])**2+dispersion**2,1,'unit-vector dispersion identity',errors)
        close(r['centered_profile_norm'],np.linalg.norm(deltas[g]),'offset norm',errors)
        close(r['heldout_positive_count'],np.sum(np.array(heldout_by_genus[g])>0),'positive heldout count',errors)
    for r in read_rows(out/'tables/pre_gate_sensitivity_summary.csv'):
        g=r['genus']
        close(r['centered_profile_cosine'],vector_cos(deltas[g],pre_deltas[g]),'pre-gate cosine',errors)
        close(r['centered_profile_l2_change'],np.linalg.norm(deltas[g]-pre_deltas[g]),'pre-gate change',errors)
    pairs=read_rows(out/'tables/all_genus_pair_contrasts.csv');assert len(pairs)==78
    for r in pairs:
        expected=means[r['genus_a']]-means[r['genus_b']]
        close([r[c] for c in cells],expected,'all 78 pair contrasts',errors)
        close(r['center_distance'],np.linalg.norm(expected),'all 78 center distances',errors)
    params=json.loads((out/'parameters.json').read_text())
    assert set(params['genus_order'])==set(names) and params['neuron_order']==cells
    assert set(params['strain_order'])=={s for ss in groups.values() for s in ss}
    result={'status':'PASS','method':'independent csv + numpy calculations; no import of analysis functions',
            'input_hashes_match':True,'n_genera':13,'n_strains':90,'n_neurons':13,
            'n_holdouts':90,'n_genus_pair_contrasts':78,'max_absolute_error_by_check':errors,
            'maximum_absolute_error':max(errors.values())}
    (out/'verification/numerical_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    verify(Path(__file__).resolve().parents[4])
