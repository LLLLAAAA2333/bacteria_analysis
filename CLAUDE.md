# bacteria_analysis

C. elegans 细菌代谢物神经分析的探索性仓库。目的是探索不同的分析方法，而不是搭建可复用的分析流程。

## 环境

- 使用 `pixi` 管理 Python 3.11 环境，`pixi run python` 执行脚本
- `PYTHONPATH=src`，导入路径形如 `from bacteria_analysis.features.neural import ...`

## 代码组织

```
src/bacteria_analysis/
├── preprocessing.py           # 原始 trace → 分析就绪格式
├── constants.py               # 神经元顺序、时间点等常量
├── _data_loaders.py           # 读取 metabolite matrix、stimulus mapping
├── _vector_distance.py        # correlation / Euclidean 距离
├── _analysis_dataset_impl.py  # AnalysisDataset 加载 Parquet + matrix
├── features/
│   ├── neural.py              # neural feature matrix、stimulus prototypes
│   ├── chemical.py            # chemical feature matrix、QC 过滤
│   ├── taxonomy.py            # taxonomy 统计 (permutation, FDR)
│   └── anchor.py              # anchor stimulus 日期效应分析
└── analyses/rdm/
    ├── core.py                # RDM 对齐、上三角、相似度
    ├── builders.py            # 构建 neural/chemical RDM
    ├── stats.py               # permutation null、empirical p
    └── plots.py               # 绘图原语 (~1900 行)
```

## 工作方式

- **不做 test、不做 stage、不做 spec/plan。** 这是探索，不是软件工程。
- 分析脚本放在 `scripts/` 或 `exploratory/`，自由 import 需要的模块。
- 结果输出到 `results/<batch_name>/`。
- 数据在 `data/` 目录。

## 数据

| 文件 | 说明 |
|------|------|
| `data/86bac.parquet` | 86 细菌神经钙成像数据 |
| `data/GM300_bacteria_species_summary.xlsx` | 菌种元数据 |
| `data/matrix.xlsx` | 代谢物浓度矩阵 |
| `data/data_fc_missingto1_filtered.xlsx` | 过滤后的 fold-change 数据 |

## 常用模式

```python
# 加载数据
from bacteria_analysis._analysis_dataset_impl import AnalysisDataset
from bacteria_analysis._data_loaders import read_metabolite_matrix
ds = AnalysisDataset(neural_path="data/xxx.parquet", matrix_path="data/xxx.xlsx")

# 构建特征
from bacteria_analysis.features.neural import build_trial_feature_matrix, build_stimulus_prototypes
features = build_trial_feature_matrix(ds, view="response_window", merge_lr=True)
prototypes = build_stimulus_prototypes(features, aggregation="median")

# 构建 RDM
from bacteria_analysis.analyses.rdm.builders import build_neural_rdm, build_chemical_rdm
neural_rdm = build_neural_rdm(ds, view="response_window")
chemical_rdm = build_chemical_rdm(ds, qc_rsd_threshold=20)

# 统计
from bacteria_analysis.analyses.rdm.stats import label_shuffle_null, empirical_p_value
```

## 关键分析历史

- **76bac**: 76 细菌批次，主要 RSA 分析 (`scripts/plot_neural_chemical_rdm_foundation.py`)
- **86bac**: 86 细菌批次，shape PCA、HMDS、化学嵌入探索 (`exploratory/`)
