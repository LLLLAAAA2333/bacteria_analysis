# 模板幅度近邻报告复现

本目录使用图4和图5的固定0–40 s全数据模板，重新表达已有147对化学可比菌株，并查看沿用图5标准的5对候选及A022/A023参照。结果说明见[REPORT.md](REPORT.md)，这是分析检查图，尚未定稿为poster第6图。

代码由现有Notebook调用即可，无需新建Notebook或运行整本Notebook。在仓库环境中运行以下代码会覆盖本目录导出表和检查图；不修改原始数据、旧报告、图4或图5。

```python
from pathlib import Path
import sys

root = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis')
sys.path.insert(0, str(root / 'notebook'))
from neighborhood_amplitudes import (
    analyze_amplitude_neighborhoods, select_main_pairs, summarize_main_patterns,
)
from neighborhood_amplitudes_display import plot_amplitude_neighborhoods

reports = root / 'reports'
out = reports / 'poster_neighborhood_amplitudes_20261001'
analyze_amplitude_neighborhoods(reports, out)
select_main_pairs(out, reports, 'figure5_benchmark')
summarize_main_patterns(out)
plot_amplitude_neighborhoods(out)
```

运行环境为`.pixi/envs/default/bin/python`，无界面作图设`MPLBACKEND=Agg`。主计算代码为`notebook/neighborhood_amplitudes.py`，绘图代码为`notebook/neighborhood_amplitudes_display.py`。若更改模板、化学标准或细胞面板，不能沿用本报告正文的数字和判断。

`tables/animal_amplitudes.csv`包含所有112个条件、7类细胞的动物投影，完整缺测仍为缺测。`tables/pair_animal_amplitudes.csv`保存共同动物的两株幅度及差值；`tables/pair_animal_curve_differences.csv`保存八窗差曲线及模板外残差。每单元的均值、SEM、跨动物能量、覆盖及删除范围位于`tables/pair_cell_summary.csv`。汇总菌株对时固定使用六类共同覆盖细胞，AWCON仍保留在七列检查图及原始输出中。

Focused checks:

```bash
.pixi/envs/default/bin/python -m unittest discover -s tests -p 'test_neighborhood_amplitudes.py' -v
```

测试覆盖投影单位、缺测与零、部分bin拒绝、配对SEM与负能量、残差分解、A/B顺序交换以及低覆盖。`verification.json`记录缓存数据核验；`analysis_parameters.json`记录输入来源、SHA256、范围、单位及探索统计定义。
