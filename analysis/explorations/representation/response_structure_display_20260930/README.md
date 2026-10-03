# 神经响应结构：展示图

仅重绘已有结果，未重新拟合模型、改变观测掩码或损失权重。图中的曲线和预测均来自真实缓存数据。结论仍限定于当前刺激协议、已有菌株×采集日期中的钙响应表型。

| 图 | 主窗口 0–40秒 | 0–25秒检查 |
|---|---|---|
| A 共享时间轮廓 | [PNG](figures/A_time_profiles_0-40s.png) · [PDF](figures/A_time_profiles_0-40s.pdf) | [PNG](figures/A_time_profiles_0-25s.png) · [PDF](figures/A_time_profiles_0-25s.pdf) |
| B 细胞响应组合 | [全部四页PDF](figures/B_response_combinations.pdf) · [第1页PNG](figures/B_response_combinations_page1.png) | 本图仅展示主窗口 |
| C 跨动物预测偏差 | [PNG](figures/C_model_comparison_0-40s.png) · [PDF](figures/C_model_comparison_0-40s.pdf) | [PNG](figures/C_model_comparison_0-25s.png) · [PDF](figures/C_model_comparison_0-25s.pdf) |
| C补充 逐动物预测增益 | [PNG](figures/C_gain_animals_0-40s.png) · [PDF](figures/C_gain_animals_0-40s.pdf) | [PNG](figures/C_gain_animals_0-25s.png) · [PDF](figures/C_gain_animals_0-25s.pdf) |

各图另存SVG；B每页独立保存PNG和SVG。完整数值见 `figure_data/`。

## A 图注与选择规则

每格对应一个细胞、菌株和采集日期。黑线为示例动物，灰线为其余动物；蓝虚线为共享时程预测，橙线为该条件的独立均值时程预测。两种预测均由黑线动物以外的训练动物估计，包括模板和系数；没有用测试动物重新调整幅度。浅灰背景为0–10秒刺激期。曲线保留原始ΔF/F₀，未额外中心化或按幅度归一；各面板纵轴范围不同。

这些条件在查阅既有结果后选择，分别用于展示主体形状、局部反例、动物间不一致和覆盖不足，属于说明性案例，不构成新的确认性检验。AWCON A300是强正向响应，A264是弱负向响应，两者幅度不能按面板高度直接比较。AWB A002和AWCOFF A024来自已有时程定位证据，重点为10–25秒。ASG A305显示其他动物的响应不能可靠预测示例动物。

有可评分预测的五个条件均按同一规则选动物：以该动物0–40秒M1预测MSE升序排列，动物ID破同分，取索引 `n//2`。不先筛选M2受益动物，不挑误差最小动物。全部动物仍在图中。ASI A237来自13个仅两动物且未评分的ASI条件：按动物均值曲线的RMS升序、菌株和日期破同分，取中间条件；两条实际曲线全部展示，不画预测。短窗口复用相同条件和动物，预测取独立拟合的0–25秒缓存结果。

选择记录：`figure_data/A_example_selection.csv`；逐动物观测与预测：`figure_data/A_response_and_prediction_values.csv`。

## B 图注

七类细胞的共享时程系数与对应模板放在左侧；其余六类保留原分窗响应。†表示单模板保留了部分可重复差异，但仍有额外时程证据，不表示该细胞的所有条件都适合压缩。原单位色标跨细胞、跨页一致；超出显示范围的值以点标出，原数值不截断。低覆盖用斜线，缺失用灰色。系数正负表示沿模板或翻转模板，不能直接解释为兴奋或抑制。

全部112个菌株×日期条件按编号、日期排序，不按响应排序或聚类。完整说明、显示范围和覆盖规则见 [B_caption.md](B_caption.md)。

## C 图注

图例依次对应B、M0、M1、M2。主图的“预测偏差”为原加权MSE的平方根，即RMSE，单位为ΔF/F₀；不是逐动物RMSE的平均。每个细胞使用独立横轴，同一个细胞在两个窗口使用相同横轴。灰竖虚线为所有菌株使用相同训练平均曲线的偏差。蓝线向左表示允许各细胞分别改变方向和幅度增加了预测能力，橙线向左表示进一步允许时程改变增加了预测能力；向右表示预测变差。整体增益M0使用所有细胞共用的非负倍数。

所有模型比较相同的测试记录。逐细胞汇总先在每个时间窗内对有效动物等权，再对时间窗、同菌株采集块和菌株依次等权；未对细胞响应缩放。图中n为可评分动物数/菌株×采集块条件数。正式增益仍为MSE差，保留于CSV和C补充图；RMSE差不作为原MSE增益的替代指标。

补充图灰点描述每只留出动物在其可用条件上的配对误差差值，菱形为原层级加权汇总，两者权重对象不同。所有负增益保留，不把时间窗或重叠验证折当独立重复。独立均值时程M2是训练均值参照，不是真实响应或噪声上限；无额外收益不能证明时程严格相同。

## Notebook 调用

```python
from response_structure_display import plot_display
plot_display(RS_OUTPUT, RS_ROOT / "reports/response_structure_display_20260930")
```

02 Notebook原有新增分析段的绘图单元已改为浏览这三张独立图。设置 `RS_DISPLAY_REDRAW=True` 仅从缓存重绘；不会触发原始数据处理或模型拟合。绘图入口会核对人工判断绑定的证据指纹。

`display_parameters.json` 记录输入证据和代码指纹；`verification.json` 记录本次核验。原科学判断和原完整资源图仍见 [原分析报告](../response_structure_20260930/README.md)。
