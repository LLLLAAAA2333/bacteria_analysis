# 本轮输入与 notebook 03 的对应

- 神经侧采用 cell 8 当前设置：13 类神经元 × 5 个 5 s 窗口，即刺激开始后 [0,25) s；[0,10) s 为刺激期。607 个动物–菌株观测，49 个真实动物（date、worm_key 联合标识），106 个菌株，9 次采集。trial、神经元、时间窗不是独立动物。
- 同一 trial 同一时点先平均左右侧，再在动物内平均 trials，最后每 5 s 平均。缺失神经元保持 NaN。菌株描述均值先动物内日期平均、再日期等权。本轮从已验证 animal_curves 复用计算，并直接调用 cell 8 的原始 parquet 准备函数核对全部 607×65 个位置；最大绝对差 8.88e-16。
- 化学侧完全对应 cell 11：RELIABLE_METABOLITES=None，读 matrix.xlsx 第一张表；正且有限的固定特征集合上直接 log₂FC，无新增 +1、填补、截断或 QC 筛选。当前配对 106 菌株 × 380 特征，全库 299 × 380，数值无效排除 0。前轮的 162 指完整且 QC RSD≤0.30 的化学特征数，并非 162 株菌；本轮保留全部 380，与 notebook 对齐。
- QC 和检测信息作为描述保留：配对原报告共有 3962 个缺失数值，但 FC 表全为正。数值有效不等于可靠检测。所有 299×380 FC 均可按 `(raw.fillna(0)+1)/(mean(reference.fillna(0))+1)` 重建，最大绝对 log 误差 9.99e-15。本轮不再施加第二个 +1。已有零填充和任意单位 +1 的含义仍须保留，尤其接近缺失/低水平的值。
- 新纳入的 notebook cell 5 元数据明确：材料是 bacterial spent medium，化学谱是同培养条件、独立培养批次的菌株参考谱，FC 相对 medium。此前“培养条件完全未知”的表述应更新；仍不能称为神经刺激 aliquot 的实测分子浓度。具体培养基配方及参考 AID 生物学标签未找到，不作猜测。
- 配对菌株分属四个数值参考组：{"ref12": 40, "A306": 26, "A050": 25, "A250": 15}。组名仅代表重建表格用的参考ID组合，不是独立生物学解释。
- 名称、Class/SubClass、Mass/RT、QC RSD、原报告检出状态均与 380 个 FC 特征一一对应；注释置信级别、LOD/LOQ 和神经暴露剂量仍缺失。

## 可复用文件

- `tables/aligned_neural_animal_5bins.parquet` / CSV：MultiIndex sample_id,date,worm_key；65 列如 ASK__00_05，单位 ΔF/F₀；缺失 NaN。
- `tables/aligned_neural_features.csv`：每列对应神经元、相对窗口和原始索引。
- `tables/aligned_neural_strain_5bins.csv`：仅作描述的日期等权均值。
- `tables/aligned_chemical_log2fc_paired.csv`：sample_id 行索引；106×380，列序对应 notebook。
- `tables/aligned_chemical_log2fc_all.csv`：299×380，全库用于化学结构描述时不得当作额外神经样本。
- `tables/aligned_chemical_metadata.csv`、`aligned_chemical_report_observed_*`、`aligned_chemical_reference_groups_*`：描述质量、原报告非缺失状态和数值参考。
- `logs/alignment_inputs.json`：原始文件、notebook、复用表 SHA256；`alignment_audit.json`：全部核对数值。

已执行：全部配对/全库化学变换、全动物神经复算比较、参考变换重建、唯一ID和维度/缺失断言。未执行：整个 notebook、RSA重跑、化学原始谱峰重积分或新的身份验证。
