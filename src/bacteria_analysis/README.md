# 分析辅助函数

这里存放从旧 `notebook/` 目录迁入的 24 个 Python 模块，供 [notebooks/](../../notebooks/) 调用，涵盖响应表示、化学与神经距离、HMDS、配色和展示。通过 `bacteria_analysis` 包导入，例如 `from bacteria_analysis import neural_hmds`；包内模块使用相对导入。

迁移只调整导入与文件定位，科学算法和参数保持不变。输入和输出仍由调用方显式传入，`phylogenetic_colors` 的默认树文件仍为本地 `data/16S.aln.trim.fa.treefile`。当前 poster 的共享函数继续位于 [poster/src/poster_analysis/](../../poster/src/poster_analysis/)。

在仓库根目录运行 `pixi run test` 可检查分析模块及 poster 的科学计算约定，不会执行整本 notebook。
