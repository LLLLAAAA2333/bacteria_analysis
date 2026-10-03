# 独立属内模式分析的原始范围

以下约束提取自 2026-10-03 的 [scope.json](../../../../reports/genus/exploration_genus_patterns_independent_20261003/scope.json)，保留英文原文。原 JSON 连同当时的样本数量、输入哈希和核对状态保留在本地产物目录。

## 工作范围

Steps 3 and 4 independently and concurrently; no step 5 or cross-modal matching.

样本纳入规则：Genus has >=2 strains in the saved 106-strain context; retain all such genera.

## 模态分离

- Chemical: Only chemical values, chemical annotation metadata and taxonomy/context enter discovery, ordering and checks.
- Neural: Only neural values and taxonomy/context enter patterns, ordering and checks.
- Shared: Sample inclusion rule and taxonomy labels only; no transfer of feature choices, modules, cluster trees or ordering between modalities.

## 解释边界

These are analyses conducted separately on the same previously explored samples, not independent biological validation.
