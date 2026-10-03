# Bacteroides 二维神经子空间：删样本与gate敏感性

**在本轮删样本检查中，前两PC张成的二维方向平面通常比单独PC1改变更少。** 这说明单PC1转动不必等于主要二维结构完全改变，但不能据此断言实验重复可靠。这里比较的是包含关系下的训练子集与full29参考；动物记录分半可靠性由另一分支独立评估，本目录未读取其结果，也未读取化学数据。

固定使用全部29株Bacteroides、全部13个神经坐标。各训练集重新估计均值后做PCA，不按神经元SD缩放，不再次单位化中心化后的菌株。K=2由本轮问题事先固定，没有根据结果选择维数。

![All strains in the top-two neural plane](figures/01_neural_pc1_pc2.png)

两面板点位置相同：左边按记录species，右边按完整记录date-set。保留全部菌株ID、16个species、8个date-set和6个taxonomy标记；`*`表示来源分类注释标记，不是显著性。两个坐标轴等比例显示，PC分数是单位神经profile坐标的投影，不是SD，也不包含整体响应强弱。日期只是来源记录标签，未假定为培养或测量批次。

## 主要数值

full29的PC1和PC2分别解释 **33.60%和31.24%** 的本菌属单位神经profile方差，合计 **64.84%**；仍有35.16%位于其他方向。这些比例是表示本身的方差分解，不是化学或预测模型解释度。

| 重新拟合方式 | 次数 | 二维最大主角度：最小 / 中位 / 最大 | 二维角度最大时的删除对象 | 单PC1角度中位 / 最大 |
|---|---:|---:|---|---:|
| 删一株 | 29 | 0.30° / 2.87° / 11.62° | A015 | 12.65° / 84.70° |
| 留出一个记录species | 16 | 1.54° / 4.73° / 11.70° | B. fluxus（A019、A020） | 18.18° / 76.70° |

例如，删去A048时，PC1的锐角变化为 **84.70°**，但二维平面的最大主角度为 **6.52°**；留出记录B. salyersiae（A010、A048）时，两者分别为 **76.70°和5.42°**。这与PC1/PC2在近似同一二维平面内重新定向相容。二维平面也并非完全不变：两种删除方式的最坏角度约为11.6–11.7°。未设置任何人为“稳定达标”阈值。

![All deletion sensitivities](figures/02_subspace_deletion_sensitivity.png)

左图展示全部45个最大主角度；黑线为中间50%范围及中位数，不是置信区间。标注为每种删除方式中二维角度最大者。右图展示同一45次拟合的单PC1角度与二维最大角度；虚线表示两种角度相同，标注为每种方式中PC1角度最大者。某些点重合，因为删除一个仅一株的species与删除该菌株是相同子集。

## 为什么同时检查PC2与PC3的间隔

full29样本特征值为λ1=0.062706、λ2=0.058311、λ3=0.026010。λ1−λ2=0.004395，而λ2−λ3=0.032301，λ2/λ3=**2.242**。前两轴彼此接近，而PC2与PC3间的间隔更大，这有助于解释单轴与二维平面敏感性的差别；它不证明存在两个独立生物过程。

删除后的训练top2累计EVR范围为：删株 **62.85–69.34%**，留记录species **61.86–69.34%**。对应λ2/λ3范围分别为 **1.938–2.605** 和 **1.734–2.605**。全部13个特征值、EVR、λ2−λ3、相对gap及比值均保留，没有只报告最有利删除对象。

## gate与pre-gate的同队列比较

对相同29株pre-gate单位profile，用其自身均值重新拟合PCA，与gated full29二维参考比较，两个主角度为 **5.23°、7.70°**；投影矩阵距离除以√2为 **0.1621**。单PC1角度为16.23°（绝对loading cosine=0.9602）。pre-gate前两PC合计解释63.17%方差。

这是固定输入表示的敏感性比较：没有搜索gate阈值、重拟合模板或剔除菌株，也不是独立重复实验。

## 指标、来源和边界

对两个13×2正交loading矩阵U、V，用UᵀV的两个奇异值计算主角度θ1≤θ2。它们比较两个方向平面的相对取向；最大角度θ2反映两个平面最不一致的方向。单PC1使用绝对内积后取锐角，因此不受PC符号翻转影响。

另存 `||UUᵀ−VVᵀ||F / √2 = sqrt(sin²θ1 + sin²θ2)`，取值范围0至√2。它不是角度，也不是0–1比例。主角度与该距离都不随平面内的基向量旋转、交换或符号改变而改变。

训练均值每折重新估计，所有投影使用该折的训练均值；主角度本身只比较loading方向平面，不计入均值位置移动。均值和逐株投影均另存，未把方向敏感性包装成位置或测量重复性。

仅使用前轮 `reports/exploration_bacteroides_local_model_20261003/neural/tables/` 中的 `neural_unit_profiles.csv`、`strain_metadata.csv`、`pre_gate_unit_profiles.csv`。按唯一strain ID及原始13神经列顺序对齐，六个taxonomy flags保留。记录species定义删除组，未验证分类正确性；日期只用于展示。来源处理、共享模板、物种/记录日期结构及样本量均限制独立性。

## 文件和调用

| 文件 | 内容 |
|---|---|
| `tables/full_gated_scores_with_metadata.csv`、`full_gated_loadings.csv` | 全29分数/13维loadings及逐株元信息 |
| `tables/gated_unit_profiles_29x13.csv`、`pre_gate_unit_profiles_29x13.csv` | 两套完整输入表示 |
| `tables/deletion_subspace_metrics.csv` | 全45次删除的角度、投影距离、PC1及谱间隔 |
| `tables/deletion_metric_summary.csv`、`worst_deletion_cases.csv` | 完整汇总和按预定极值规则选出的对象 |
| `tables/pre_gate_subspace_comparison.csv` | 同29株gate/pre-gate比较 |
| `tables/all_fit_means.csv`、`all_fit_loadings.csv`、`all_fit_spectra.csv` | 47套完整参数：2次全样本及45次删除 |
| `tables/all_fit_strain_scores.csv` | 每套参数下的全部29株投影；明确training/omitted_projection |
| `fold_ids.json`、`summary.json` | 每次train/omitted ID、根报告可读取的全部主要统计 |
| `manifest.json`、`plot_parameters.json` | 来源/代码哈希、参数、确切绘图样式与标签偏移 |
| `verification.json`、`independent_angle_verification.csv` | 独立数值复算与逐次角度对照 |

`code/neural_subspace.py`提供可供既有Notebook调用的`fit_pca(values)`、`compare_subspaces(reference_loadings, other_loadings)`和`spectrum_metrics(fit)`。`code/run_subspace_analysis.py`内的`run_analysis(repo_root, out=None)`拒绝覆盖已有核心结果；主动重算必须指定新的out目录。`code/plot_subspace.py`内的`save_plots(output)`只读取已保存结果并输出图，不重新估计PCA。

使用既有`.pixi/envs/default/bin/python`，没有安装依赖或新建Notebook。独立验证由协方差矩阵特征分解重建全部47套PCA，并用SciPy `subspace_angles`、显式投影矩阵及正弦恒等式复核所有45次删除及pre-gate比较；还检查了符号/68°平面内旋转不变性、全部ID、全部投影、极值、来源哈希和防覆盖。结果 **PASS**，最大绝对差 **3.27×10⁻⁹**（角度数值误差）。两张PNG已视检，图01两处标签碰撞仅通过文字偏移修正，数据坐标未变。
