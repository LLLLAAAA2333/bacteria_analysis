# 分析 notebook

这里保留三本分析 notebook：`01_measurement_first_inspection.ipynb` 检查测量与输入，`02_reproducibility_inspection.ipynb` 检查响应的可重复性，`03_chemical_neuron_bacteria.ipynb` 探索化学与神经响应的关系。当前 poster 入口见 [poster/README.md](../poster/README.md)。

从仓库根目录运行 `pixi run lab`，选择项目环境的 Python kernel。Notebook 可从仓库根目录或本目录定位输入；原始数据仍放在本地 `data/`，生成结果保留各 notebook 原有的输出路径。

共享辅助函数位于 [src/bacteria_analysis/](../src/bacteria_analysis/)，历史备份位于 [archive/notebooks/](../archive/notebooks/)。目录迁移只调整路径与导入，没有重新执行分析。Git 保存代码和叙述；已有执行结果保存在本地 `reports/notebooks/analysis/`，历史指纹仍描述当时执行。手动运行后，用 `pixi run clean-notebooks` 保存完整输出副本并清理源码 notebook。
