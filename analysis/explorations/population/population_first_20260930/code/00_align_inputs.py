"""Align this round to notebook 03's current 13 x 5 bins and 380 log2FC inputs.

No notebook is run wholesale or edited. The two reviewed preparation functions
are extracted from notebook cells 8 and 11, and their raw-derived neural output
is checked against the reused animal curves. Missing neurons remain NaN.
Chemical numeric validity is not detection/QC validity: the FC spreadsheet
already contains upstream zero filling and a +1 convention, checked separately.
Run: .pixi/envs/default/bin/python reports/population_first_20260930/code/00_align_inputs.py
"""
from pathlib import Path
import ast
import hashlib
import importlib.metadata
import json
import platform
import sys
import unicodedata

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
OLD = ROOT / 'reports/exploration_20260929/tables'
TABLES, LOGS = OUT / 'tables', OUT / 'logs'
NEURONS = ['ASK', 'ADL', 'ASI', 'AWA', 'AWB', 'ASG', 'ADF', 'ASH', 'ASJ',
           'ASEL', 'ASER', 'AWCON', 'AWCOFF']
KEY = ['sample_id', 'date', 'worm_key']
SEED = 20260930


def literal_assignment(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def norm(value):
    return unicodedata.normalize('NFKC', str(value)).strip()


def file_record(path):
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    for directory in [TABLES, LOGS]:
        directory.mkdir(parents=True, exist_ok=True)
    notebook_path = ROOT / 'notebook/03_chemical_neuron_bacteria.ipynb'
    notebook = json.loads(notebook_path.read_text())
    neural_source = ''.join(notebook['cells'][8]['source'])
    chemical_source = ''.join(notebook['cells'][11]['source'])
    assert literal_assignment(neural_source, 'N_BINS') == 5
    assert literal_assignment(neural_source, 'DATE_WEIGHTING') == 'equal'
    assert list(literal_assignment(neural_source, 'NEURON_CLASSES')) == NEURONS
    assert literal_assignment(chemical_source, 'RELIABLE_METABOLITES') is None

    curves = pd.read_parquet(OLD / 'animal_curves.parquet')
    curve_keys = curves.index.to_frame(index=False).astype(str)
    curves.index = pd.MultiIndex.from_frame(curve_keys)
    ids = pd.Index(sorted(curve_keys.sample_id.unique()), name='sample_id')
    features = []
    binned_columns = {}
    for start in range(0, 25, 5):
        binned_columns[f'{start:02d}_{start+5:02d}'] = curves[[str(t) for t in range(start, start+5)]].mean(axis=1)
    binned = pd.DataFrame(binned_columns)
    wide = binned.unstack('neuron_class').swaplevel(axis=1)
    desired = pd.MultiIndex.from_product([NEURONS, binned.columns], names=['neuron_class', 'time_bin'])
    wide = wide.reindex(columns=desired).sort_index()

    # Execute only the already reviewed declarations/preparation function, not
    # any notebook analysis or cell outputs; direct raw comparison uses all data.
    prepare_prefix = neural_source.split('\nneural_vectors = load_neural_vectors')[0]
    scope = {'__name__': '__alignment_preparation__'}
    exec(compile(prepare_prefix, str(notebook_path) + ':cell8:preparation', 'exec'), scope)
    direct = scope['load_neural_vectors'](ROOT / 'data/106bac.parquet')
    direct.columns = desired
    direct = direct.reindex(index=wide.index, columns=desired)
    assert wide.shape == (607, 65)
    assert np.array_equal(np.isfinite(wide), np.isfinite(direct))
    neural_max_error = float(np.nanmax(np.abs(wide.to_numpy() - direct.to_numpy())))
    assert neural_max_error < 1e-12
    for neuron in NEURONS:
        for bin_number, start in enumerate(range(0, 25, 5)):
            feature = f'{neuron}__{start:02d}_{start+5:02d}'
            features.append(dict(feature_id=feature, neuron_class=neuron, bin_number=bin_number,
                                 start_s=start, stop_s_exclusive=start+5,
                                 original_start_index=start+5, original_stop_index_exclusive=start+10,
                                 unit='delta_F_over_F0', phase='stimulus' if start < 10 else 'post_stimulus'))
    wide.columns = [f['feature_id'] for f in features]
    wide.to_parquet(TABLES / 'aligned_neural_animal_5bins.parquet')
    wide.to_csv(TABLES / 'aligned_neural_animal_5bins.csv')
    wide.groupby(level=['sample_id', 'date']).mean().groupby(level='sample_id').mean().to_csv(
        TABLES / 'aligned_neural_strain_5bins.csv')
    pd.DataFrame(features).to_csv(TABLES / 'aligned_neural_features.csv', index=False)
    support = wide.index.to_frame(index=False)
    support['animal_id'] = support.date + '__' + support.worm_key
    support['n_complete_neurons'] = np.isfinite(wide.to_numpy().reshape(-1, 13, 5)).all(axis=2).sum(axis=1)
    support.to_csv(TABLES / 'aligned_neural_support.csv', index=False)

    fc = pd.read_excel(ROOT / 'data/matrix.xlsx', sheet_name=0, index_col=0)
    fc.index = fc.index.astype(str).str.strip()
    fc.index.name = 'sample_id'
    fc.columns = fc.columns.astype(str).str.strip()
    assert fc.index.is_unique and fc.columns.is_unique
    assert fc.index.str.fullmatch(r'A\d{3}').all()
    assert ids.isin(fc.index).all()
    chemical_function = next(n for n in ast.parse(chemical_source).body
                             if isinstance(n, ast.FunctionDef) and n.name == 'chemical_rms_distance')
    chemical_scope = {'np': np, 'pd': pd, 'pdist': pdist, 'squareform': squareform}
    exec(compile(ast.Module(body=[chemical_function], type_ignores=[]), str(notebook_path) + ':cell11:function', 'exec'), chemical_scope)
    chemical_prepare = chemical_scope['chemical_rms_distance']
    paired, _, paired_audit = chemical_prepare(fc.loc[ids], None)
    full, _, full_audit = chemical_prepare(fc, None)
    assert paired.shape == (106, 380) and full.shape == (299, 380)
    assert paired.columns.equals(full.columns)
    assert np.allclose(paired, full.loc[ids], rtol=0, atol=0)
    paired.to_csv(TABLES / 'aligned_chemical_log2fc_paired.csv')
    full.to_csv(TABLES / 'aligned_chemical_log2fc_all.csv')
    paired_audit.add_prefix('paired_').join(full_audit.add_prefix('all_')).to_csv(
        TABLES / 'aligned_chemical_numerical_audit.csv')

    source = pd.read_excel(ROOT / 'data/metabolism_raw_data.xlsx', sheet_name='all')
    source['name'] = source.name.map(norm)
    source = source.set_index('name')
    assert source.index.is_unique
    lookup = pd.Series(paired.columns, index=paired.columns.map(norm))
    assert lookup.index.is_unique and lookup.index.isin(source.index).all()
    raw = source.loc[lookup.index, fc.index].T
    raw.columns = lookup.values
    raw.index.name = 'sample_id'
    raw.notna().to_parquet(TABLES / 'aligned_chemical_report_observed_all.parquet')
    raw.loc[ids].notna().to_parquet(TABLES / 'aligned_chemical_report_observed_paired.parquet')
    raw.to_csv(TABLES / 'aligned_chemical_report_values_all.csv')
    metadata_names = ['Mass', 'RT', 'column', 'ChineseName', 'KEGG', 'HMDB', 'SuperClass',
                      'Class', 'SubClass', 'DirectParent', 'QCRSD']
    metadata = source.loc[lookup.index, metadata_names].copy()
    metadata.index = paired.columns
    metadata.index.name = 'metabolite'
    qcols = [c for c in source.columns if str(c).startswith('QC-')]
    metadata['qc_n_observed'] = source.loc[lookup.index, qcols].notna().sum(axis=1).to_numpy()
    metadata['paired_report_observed_fraction'] = raw.loc[ids].notna().mean().reindex(metadata.index)
    metadata['all_report_observed_fraction'] = raw.notna().mean().reindex(metadata.index)
    metadata['qc_rsd_above_0_30'] = metadata.QCRSD.gt(.30)
    metadata['qc_rsd_missing'] = metadata.QCRSD.isna()
    metadata['previous_complete_162_eligible'] = metadata.QCRSD.le(.30) & metadata.paired_report_observed_fraction.eq(1)
    metadata['retained_current_analysis'] = True
    metadata['annotation_status'] = 'Report annotation; identification confidence levels not supplied'
    metadata.to_csv(TABLES / 'aligned_chemical_metadata.csv')

    references = {'A050': ['A050'], 'ref12': [f'A{i:03d}' for i in [51,52,53,54,55,56,57,88,89,90,91,92]],
                  'A316_A317': ['A316', 'A317'], 'A225': ['A225'], 'A250': ['A250'],
                  'A306': ['A306'], 'A318': ['A318']}
    errors = {}
    for reference, members in references.items():
        denominator = source.loc[lookup.index, members].fillna(0).mean(axis=1).to_numpy() + 1
        reconstructed = (raw.fillna(0).to_numpy() + 1) / denominator
        errors[reference] = np.abs(np.log(reconstructed / fc.to_numpy())).max(axis=1)
    errors = pd.DataFrame(errors, index=fc.index)
    reference_map = pd.DataFrame({'reference_group': errors.idxmin(axis=1),
                                  'max_log_reconstruction_error': errors.min(axis=1)})
    assert reference_map.max_log_reconstruction_error.max() < 1e-10
    reference_map.to_csv(TABLES / 'aligned_chemical_reference_groups_all.csv')
    reference_map.loc[ids].to_csv(TABLES / 'aligned_chemical_reference_groups_paired.csv')
    taxonomy = pd.read_excel(ROOT / 'data/GM300_bacteria_species_summary.xlsx', sheet_name='Axxx_species_mapping')
    taxonomy = taxonomy.rename(columns={'AID': 'sample_id'}).set_index('sample_id')
    assert taxonomy.index.is_unique
    taxonomy.reindex(fc.index).to_csv(TABLES / 'aligned_taxonomy_all.csv')
    taxonomy.loc[ids].to_csv(TABLES / 'aligned_taxonomy_paired.csv')

    inputs = [notebook_path, ROOT/'data/106bac.parquet', ROOT/'data/matrix.xlsx',
              ROOT/'data/metabolism_raw_data.xlsx', ROOT/'data/GM300_bacteria_species_summary.xlsx',
              OLD/'animal_curves.parquet', ROOT/'pixi.toml', ROOT/'pixi.lock']
    manifest = [file_record(p) for p in inputs]
    manifest_path = LOGS / 'alignment_inputs.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Inputs changed since first run'
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    audit = dict(neural_strains=106, neural_animal_strain_rows=len(wide),
                 animals=support.animal_id.nunique(), dates=support.date.nunique(),
                 neuron_classes=NEURONS, n_bins=5, relative_window_s=[0,25], stimulus_window_s=[0,10],
                 neural_features=65, neural_missing_cells=int(wide.isna().sum().sum()),
                 raw_neural_reconstruction_max_absolute_error=neural_max_error,
                 chemical_paired_shape=list(paired.shape), chemical_full_shape=list(full.shape),
                 chemical_transform='log2(matrix.xlsx value); no additional pseudocount, imputation, or QC filtering',
                 excluded_numerically_invalid_features=int((~paired_audit.included).sum()),
                 previous_162_were='chemical features, not bacterial strains',
                 previous_complete_eligible=int(metadata.previous_complete_162_eligible.sum()),
                 paired_raw_missing_values=int(raw.loc[ids].isna().sum().sum()),
                 all_raw_missing_values=int(raw.isna().sum().sum()),
                 n_qc_injections=len(qcols), qc_rsd_above_030=int(metadata.qc_rsd_above_0_30.sum()),
                 fc_reconstruction_max_log_error=float(reference_map.max_log_reconstruction_error.max()),
                 reference_groups_paired=reference_map.loc[ids].reference_group.value_counts().to_dict(),
                 notebook_metadata={'cell': 5, 'material': 'Bacterial spent medium',
                    'matching_scope': 'Same culture conditions; independent culture batches; strain-level reference chemical profiles',
                    'medium_reference': 'Notebook states FC is medium-relative; exact medium recipe/reference-ID biological labels unavailable'},
                 interpretation_limits=['Not LC-MS aliquots from the individual worm exposures',
                     'Numerically finite FC includes features whose source report values were missing',
                     'FC = (report.fillna(0)+1)/(mean(reference.fillna(0))+1) exactly reconstructs spreadsheet',
                     'Enrichment/depletion relative to assigned reference, not absolute concentration, production/consumption rate, or causal exposure dose',
                     'Metabolite names remain report annotations without supplied identification confidence'])
    (LOGS/'alignment_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    (LOGS/'alignment_environment.json').write_text(json.dumps(
        dict(python=sys.version, executable=sys.executable, platform=platform.platform(), seed=SEED,
             versions={name: importlib.metadata.version(name) for name in ['numpy','pandas','scipy','pyarrow','openpyxl']}), indent=2))
    saved = {}
    for cell in [8, 11]:
        saved[str(cell)] = [''.join(o.get('text', [])) for o in notebook['cells'][cell].get('outputs', []) if o.get('output_type') == 'stream']
    (LOGS/'alignment_notebook_saved_outputs.json').write_text(json.dumps(saved, indent=2, ensure_ascii=False))
    method = f'''# 本轮输入与 notebook 03 的对应\n\n- 神经侧采用 cell 8 当前设置：13 类神经元 × 5 个 5 s 窗口，即刺激开始后 [0,25) s；[0,10) s 为刺激期。607 个动物–菌株观测，49 个真实动物（date、worm_key 联合标识），106 个菌株，9 次采集。trial、神经元、时间窗不是独立动物。\n- 同一 trial 同一时点先平均左右侧，再在动物内平均 trials，最后每 5 s 平均。缺失神经元保持 NaN。菌株描述均值先动物内日期平均、再日期等权。本轮从已验证 animal_curves 复用计算，并直接调用 cell 8 的原始 parquet 准备函数核对全部 607×65 个位置；最大绝对差 {neural_max_error:.3g}。\n- 化学侧完全对应 cell 11：RELIABLE_METABOLITES=None，读 matrix.xlsx 第一张表；正且有限的固定特征集合上直接 log₂FC，无新增 +1、填补、截断或 QC 筛选。当前配对 106 菌株 × 380 特征，全库 299 × 380，数值无效排除 0。前轮的 162 指完整且 QC RSD≤0.30 的化学特征数，并非 162 株菌；本轮保留全部 380，与 notebook 对齐。\n- QC 和检测信息作为描述保留：配对原报告共有 {audit['paired_raw_missing_values']} 个缺失数值，但 FC 表全为正。数值有效不等于可靠检测。所有 299×380 FC 均可按 `(raw.fillna(0)+1)/(mean(reference.fillna(0))+1)` 重建，最大绝对 log 误差 {audit['fc_reconstruction_max_log_error']:.3g}。本轮不再施加第二个 +1。已有零填充和任意单位 +1 的含义仍须保留，尤其接近缺失/低水平的值。\n- 新纳入的 notebook cell 5 元数据明确：材料是 bacterial spent medium，化学谱是同培养条件、独立培养批次的菌株参考谱，FC 相对 medium。此前“培养条件完全未知”的表述应更新；仍不能称为神经刺激 aliquot 的实测分子浓度。具体培养基配方及参考 AID 生物学标签未找到，不作猜测。\n- 配对菌株分属四个数值参考组：{json.dumps(audit['reference_groups_paired'], ensure_ascii=False)}。组名仅代表重建表格用的参考ID组合，不是独立生物学解释。\n- 名称、Class/SubClass、Mass/RT、QC RSD、原报告检出状态均与 380 个 FC 特征一一对应；注释置信级别、LOD/LOQ 和神经暴露剂量仍缺失。\n\n## 可复用文件\n\n- `tables/aligned_neural_animal_5bins.parquet` / CSV：MultiIndex sample_id,date,worm_key；65 列如 ASK__00_05，单位 ΔF/F₀；缺失 NaN。\n- `tables/aligned_neural_features.csv`：每列对应神经元、相对窗口和原始索引。\n- `tables/aligned_neural_strain_5bins.csv`：仅作描述的日期等权均值。\n- `tables/aligned_chemical_log2fc_paired.csv`：sample_id 行索引；106×380，列序对应 notebook。\n- `tables/aligned_chemical_log2fc_all.csv`：299×380，全库用于化学结构描述时不得当作额外神经样本。\n- `tables/aligned_chemical_metadata.csv`、`aligned_chemical_report_observed_*`、`aligned_chemical_reference_groups_*`：描述质量、原报告非缺失状态和数值参考。\n- `logs/alignment_inputs.json`：原始文件、notebook、复用表 SHA256；`alignment_audit.json`：全部核对数值。\n\n已执行：全部配对/全库化学变换、全动物神经复算比较、参考变换重建、唯一ID和维度/缺失断言。未执行：整个 notebook、RSA重跑、化学原始谱峰重积分或新的身份验证。\n'''
    (LOGS/'alignment_methods.md').write_text(method)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
