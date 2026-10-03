# 测试

在仓库根目录运行 `pixi run test`，等价于 `python -m unittest discover -s tests -t . -v`。

- `analysis/`：响应数据准备、响应模型、neighborhood 幅度和 HMDS 数值行为。
- `poster/`：preparation、距离计算、局部化学轴及图中汇总的计算约定。

两组使用小型数据或构造样本，不执行整本 notebook、不重新拟合完整数据。`tests/__init__.py` 统一加入两个源码目录，测试文件使用正常包导入。
