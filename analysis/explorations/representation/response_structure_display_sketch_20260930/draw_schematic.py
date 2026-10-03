"""One-page communication sketch; all arrays and comparison positions are invented.

No experimental data are read, no model is fitted, and no notebook is modified.
Run with the repository Python environment; exports PNG, SVG and PDF beside source.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np


OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
    "font.size": 11, "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#32424D", "text.color": "#23343E",
    "xtick.color": "#5C6971", "ytick.color": "#5C6971",
    "axes.edgecolor": "#ABB4BA", "pdf.fonttype": 42,
})
BLUE, ORANGE, GREY = "#207F9C", "#BC692B", "#89939A"
DARK, PALE = "#23343E", "#EEF2F4"
T = np.arange(2.5, 40, 5)
H = np.array([[.24, 1, .72, .30, .12, .04, .01, 0],
              [.10, -.40, -.30, .10, .56, .40, .13, .03]])
H = H / np.sqrt(np.mean(H**2, axis=1, keepdims=True))
WEIGHTS = np.array([[.60, .25], [.28, -.28], [-.35, .45], [.50, np.nan]])
NOISE = np.array([[.02, -.05, .06, -.02, .01, .02, -.01, .02],
                  [-.03, .04, -.02, .02, -.03, .01, .03, -.02],
                  [.04, -.02, -.04, .03, .02, -.03, .01, .03],
                  [-.02, .02, .04, -.04, .03, .02, -.02, -.01]])

fig = plt.figure(figsize=(16, 13.8), facecolor="white")
fig.text(.05, .963, "菌株差异：时间形状改变，还是细胞响应组合改变？",
         fontsize=23, weight="bold")
fig.text(.05, .935, "展示结构草图  ·  全部曲线、菌株、细胞和比较位置均为模拟，不代表本批数据结论",
         fontsize=12, color=ORANGE)


def section(y, letter, title, subtitle):
    fig.text(.05, y, letter, fontsize=18, color=BLUE, weight="bold")
    fig.text(.082, y, title, fontsize=17, weight="bold")
    fig.text(.95, y+.002, subtitle, fontsize=10.5, color="#66757E", ha="right")


def line_axis(rect, title):
    ax = fig.add_axes(rect)
    ax.axvspan(0, 10, color=PALE, zorder=-3)
    ax.axhline(0, lw=.7, color="#C3CBD0", zorder=-2)
    ax.set(xlim=(0, 40), ylim=(-1.12, 1.58), xticks=[0, 10, 25, 40],
           yticks=[-1, 0, 1], xlabel="刺激后时间（秒）")
    ax.set_title(title, loc="left", fontsize=13, pad=11, weight="bold")
    ax.tick_params(labelsize=10, length=3)
    return ax


section(.893, "A", "先看曲线：哪些变化能用同一个形状概括？", "浅灰背景：刺激期 0–10 秒")
xs = [.083, .390, .697]
axes = [line_axis([x, .704, .251, .149], title) for x, title in zip(xs, [
    "细胞Ⅰ  ·  缩放或翻转后形状相近", "细胞Ⅱ  ·  某阶段出现额外偏离", "细胞Ⅲ  ·  信号弱，暂时难判断"])]
axes[0].set_ylabel("钙响应  ΔF/F₀", fontsize=11)
for i, a in enumerate(WEIGHTS[:3, 0]):
    pred = a*H[0]
    axes[0].plot(T, pred+NOISE[i], "o-", color=GREY, lw=1.4, ms=3, zorder=2)
    axes[0].plot(T, pred, "--", color=BLUE, lw=2, zorder=3)
    axes[0].text(11.6, pred[2]+(.08 if i!=1 else -.03), f"S{i+1}", fontsize=10, color=DARK)

pred = WEIGHTS[0, 1]*H[1]
extra = np.array([0, 0, .02, .25, .48, .57, .42, .26])
for noise in NOISE[:3]:
    axes[1].plot(T, pred+extra+noise, "o-", color=GREY, alpha=.7, lw=1.2, ms=2.6)
axes[1].plot(T, pred, "--", color=BLUE, lw=2)
axes[1].annotate("后期偏离", xy=(30, .65), xytext=(20, 1.30),
                 fontsize=11, color=ORANGE,
                 arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.2})
for noise in NOISE:
    axes[2].plot(T, 2.4*noise, "o-", color=GREY, lw=1.2, ms=2.5, alpha=.65)
axes[2].plot(T, .012*H[0], "--", color=BLUE, lw=2)
for x, text in zip(xs, ["同一细胞，不同示例菌株条件", "同一示例条件，多个动物", "同一示例条件，多个动物"]):
    fig.text(x, .661, text, fontsize=10.5, color="#67757E")
fig.legend(handles=[Line2D([], [], color=GREY, marker="o", ms=3, lw=1.4, label="模拟观测"),
                    Line2D([], [], color=BLUE, ls="--", lw=2, label="共享形状预测（正式图由其他动物估计）")],
           loc="center", bbox_to_anchor=(.53, .637), frameon=False, ncol=2, fontsize=10.5)

section(.594, "B", "再看组合：各细胞的权重怎样随菌株改变？", "每行 = 一个菌株 × 采集块；示例均来自同一块")

# Templates and coefficient matrix share column positions. Cell II is partial,
# explicitly retaining its observed time courses next to the compressed display.
for j, x in enumerate([.167, .282]):
    ax = fig.add_axes([x, .516, .091, .037])
    ax.axhline(0, lw=.6, color="#BDC6CC")
    ax.axvspan(0, 10, color=PALE)
    ax.plot(T, H[j], color=BLUE, lw=1.8)
    ax.set(xlim=(0, 40), ylim=(-1.5, 2.5), xticks=[], yticks=[])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.set_title(["细胞Ⅰ", "细胞Ⅱ · 部分概括"][j], fontsize=11, pad=3)

div = LinearSegmentedColormap.from_list("signed_template", ["#655DA7", "#FAFAF8", "#218E8A"])
div.set_bad("#D3D8DC")
ax = fig.add_axes([.155, .368, .235, .137])
im = ax.imshow(WEIGHTS, cmap=div, vmin=-.65, vmax=.65, aspect="auto")
ax.set(xticks=[], yticks=range(4), yticklabels=["S1", "S2", "S3", "S4"])
ax.tick_params(length=0, pad=9)
for (r, c), v in np.ndenumerate(WEIGHTS):
    ax.text(c, r, "缺失" if np.isnan(v) else f"{v:+.2f}", ha="center", va="center",
            fontsize=12, color="white" if np.isfinite(v) and abs(v)>.4 else DARK)
for s in ax.spines.values(): s.set_visible(False)
fig.text(.067, .431, "菌株", fontsize=11)
ca = fig.add_axes([.166, .343, .217, .010])
cb = fig.colorbar(im, cax=ca, orientation="horizontal", ticks=[-.6, 0, .6])
cb.ax.tick_params(labelsize=9, length=2)
cb.outline.set_visible(False)
fig.text(.274, .311, "有符号权重（模板 RMS = 1）", fontsize=10, ha="center")

raw_ii = np.array([a*H[1] if np.isfinite(a) else np.full(8, np.nan) for a in WEIGHTS[:, 1]])
raw_ii[0] += extra
raw_iii = 2.4*NOISE
rawmap = LinearSegmentedColormap.from_list("raw_calcium", ["#477BAD", "#FCFBF8", "#C17843"])
rawmap.set_bad("#D3D8DC")
for x, data, title in [(.485, raw_ii, "细胞Ⅱ · 保留未概括的时程"),
                       (.738, raw_iii, "细胞Ⅲ · 暂保留原分窗响应")]:
    ax = fig.add_axes([x, .368, .203, .137])
    im_raw = ax.imshow(data, aspect="auto", cmap=rawmap, vmin=-1, vmax=1,
                       extent=[0, 40, 3.5, -.5], interpolation="none")
    ax.set(xticks=[0, 10, 25, 40], yticks=range(4), yticklabels=["S1", "S2", "S3", "S4"])
    ax.set_title(title, fontsize=11, pad=10)
    ax.tick_params(length=2, labelsize=10)
    ax.set_xlabel("刺激后时间（秒）", fontsize=10)
    for s in ax.spines.values(): s.set_visible(False)
ca = fig.add_axes([.595, .317, .230, .009])
cb = fig.colorbar(im_raw, cax=ca, orientation="horizontal", ticks=[-1, 0, 1])
cb.ax.tick_params(length=2, labelsize=9)
cb.outline.set_visible(False)
fig.text(.71, .285, "原分窗钙响应  ΔF/F₀", fontsize=10, ha="center")
fig.text(.05, .274, "权重正负表示沿模板或翻转模板，不能直接读作兴奋／抑制。灰格为缺失。", fontsize=10.5, color="#67757E")

section(.233, "C", "最后验证：这些概括能预测另一只动物吗？", "先整动物留出 → 仅训练动物估计全部参数 → 预测留出动物")

labels = ["各菌株使用同一预测", "所有细胞一起变强弱", "各细胞分别改变权重", "再允许时间形状改变"]
colors = ["#8B959D", "#786B9D", BLUE, ORANGE]
markers = ["s", "D", "o", "^"]
# Qualitative layout positions only. These are not MSE, RMSE, or fitted results.
positions = [[.83, .72, .28, .30], [.84, .73, .51, .22], [.25, .27, .29, .33]]
ctitles = ["例Ⅰ：组合改变有用", "例Ⅱ：额外时程也有用", "例Ⅲ：仍然无法判断"]
for j, x in enumerate([.248, .491, .734]):
    ax = fig.add_axes([x, .092, .207, .103])
    ax.set(xlim=(0, 1), ylim=(3.6, -.6), yticks=range(4), xticks=[0, 1],
           xticklabels=["偏差较小", "偏差较大"])
    ax.set_yticklabels(labels if j==0 else [""]*4, fontsize=11)
    ax.tick_params(length=0, pad=8, labelsize=10)
    ax.get_xticklabels()[0].set_ha("left")
    ax.get_xticklabels()[1].set_ha("right")
    ax.set_title(ctitles[j], fontsize=12, pad=11, loc="left", weight="bold")
    ax.spines[["left", "bottom"]].set_visible(False)
    for i in range(4):
        ax.axhline(i, color=PALE, lw=1, zorder=0)
        ax.scatter(positions[j][i], i, s=75, color=colors[i], marker=markers[i], zorder=3)
    for a, b in [(1, 2), (2, 3)]:
        ax.annotate("", xy=(positions[j][b], b), xytext=(positions[j][a], a),
                    arrowprops={"arrowstyle": "->", "color": colors[b], "lw": 1.4,
                                "shrinkA": 8, "shrinkB": 8})
fig.text(.05, .046, "预测偏差位置仅为示意，无数值含义；正式结果保留原响应单位、逐动物差异和覆盖信息。", fontsize=10.5, color="#67757E")
fig.text(.05, .022, "读图顺序：看形状 → 看响应组合 → 看跨动物是否有用。完整时程是参照，不代表真实响应或预测上限。", fontsize=10.5, color="#67757E")

for suffix in ["png", "svg", "pdf"]:
    fig.savefig(OUT / f"response_structure_display_schematic.{suffix}", dpi=170, facecolor="white")
plt.close(fig)
print(OUT / "response_structure_display_schematic.png")
