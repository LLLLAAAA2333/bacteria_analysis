"""Audit chemical report values and reconstruct legacy reference normalization.

Rows of exported profiles are strain IDs, not culture or LC-MS replicates.
Names are report annotations without supplied MSI identification levels.
Missing report values remain missing. The +1 transform is a numerical convention
in reported ng/mL units, not a detection limit or a worm exposure concentration.
"""
from pathlib import Path
import json
import unicodedata

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
T = OUT/'tables'


def norm(s):
    return unicodedata.normalize('NFKC',str(s)).strip()


def main():
    source = pd.read_excel(ROOT/'data/metabolism_raw_data.xlsx',sheet_name='all')
    source['name'] = source.name.map(norm)
    source = source.set_index('name')
    fc = pd.read_excel(ROOT/'data/matrix.xlsx',index_col=0)
    fc.columns = fc.columns.map(norm)
    assert source.index.is_unique and fc.columns.is_unique
    assert set(source.index) == set(fc.columns)
    taxonomy = pd.read_excel(ROOT/'data/GM300_bacteria_species_summary.xlsx',sheet_name='Axxx_species_mapping').rename(columns={'AID':'sample_id'})
    ids = sorted(pd.read_csv(T/'animal_metrics.csv').sample_id.unique())
    assert set(ids) <= set(fc.index) & set(source.columns) & set(taxonomy.sample_id)
    assert taxonomy.sample_id.is_unique
    taxonomy.set_index('sample_id').loc[ids].to_csv(T/'taxonomy.csv')
    raw = source[ids].T
    raw.index.name = 'sample_id'
    raw.to_csv(T/'chemical_raw.csv')
    log = np.log2(raw+1)
    log.to_csv(T/'chemical_log.csv')
    np.log2(fc.loc[ids,raw.columns]).to_csv(T/'chemical_legacy_logfc.csv',index_label='sample_id')
    qcols = [c for c in source if c.startswith('QC-')]
    qc = source[qcols].std(axis=1,ddof=1)/source[qcols].mean(axis=1)
    assert np.allclose(qc,source.QCRSD,atol=1e-12)
    metadata = source[['Mass','RT','column','ChineseName','KEGG','HMDB','SuperClass','Class','SubClass','DirectParent','QCRSD']].copy()
    metadata['qc_n_detected'] = source[qcols].notna().sum(axis=1)
    metadata['paired_detection_fraction'] = raw.notna().mean()
    # Applied without neural outcome information. Missing observations are not zeros.
    metadata['primary_eligible'] = metadata.QCRSD.le(.30) & metadata.paired_detection_fraction.ge(.90)
    metadata['complete_eligible'] = metadata.QCRSD.le(.30) & metadata.paired_detection_fraction.eq(1)
    metadata.to_csv(T/'chemical_feature_metadata.csv')
    refs = {'A050':['A050'],'ref12':[f'A{i:03d}' for i in [51,52,53,54,55,56,57,88,89,90,91,92]],
            'A316_A317':['A316','A317'],'A225':['A225'],'A250':['A250'],'A306':['A306'],'A318':['A318']}
    v = source[fc.index].T[fc.columns]
    implied = (v.fillna(0)+1)/fc
    distances = pd.DataFrame({name:np.abs(np.log(implied/(source[cols].fillna(0).mean(axis=1)+1))).max(axis=1)
                              for name,cols in refs.items()})
    group = pd.DataFrame({'reference_group':distances.idxmin(axis=1),'max_log_reconstruction_error':distances.min(axis=1)})
    group.index.name='sample_id'
    assert group.max_log_reconstruction_error.max() < 1e-10
    group.to_csv(T/'chemical_reference_groups_all.csv')
    group.loc[ids].to_csv(T/'chemical_reference_groups.csv')
    for sheet in ['A_vs_B','C_vs_D']:
        other = pd.read_excel(ROOT/'data/metabolism_raw_data.xlsx',sheet_name=sheet)
        other['name'] = other['Name'].map(norm)
        other = other.set_index('name').reindex(source.index)
        assert other.unit.eq('ng/mL').all()
        cols = [c for c in source if c.startswith('A') and c[1:].isdigit()]
        assert np.allclose(other[cols],source[cols],equal_nan=True)
    audit = dict(n_neural_strains=len(ids),n_report_features=len(source),n_qc_injections=len(qcols),
                 report_unit='ng/mL (verified in A_vs_B and C_vs_D, absent from all header)',
                 primary_features=int(metadata.primary_eligible.sum()),complete_features=int(metadata.complete_eligible.sum()),
                 qc_rsd_max_reconstruction_error=float(np.max(np.abs(qc-source.QCRSD))),
                 fc_max_log_reconstruction_error=float(group.max_log_reconstruction_error.max()),
                 reference_definitions=refs,all_reference_counts=group.reference_group.value_counts().to_dict(),
                 paired_reference_counts=group.loc[ids].reference_group.value_counts().to_dict(),
                 unavailable=['matched culture/exposure batches','LC-MS biological replicate identity','LC-MS run order',
                              'reference sample biological identities','blank metadata','standards/calibration','LOD/LOQ',
                              'MSI identification level','pH/osmolality of neural exposure'],
                 interpretation='Strain-linked reference chemical reports only; no verified sample-level exposure pairing.')
    (OUT/'logs/chemical_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
