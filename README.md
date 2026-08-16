# Photonic_Inverse_Adjoint_Opt：基于伴随法的光子超表面器件逆向设计

本项目是一个面向多端口光子超表面器件的逆向设计研究原型。它通过 Ansys Lumerical FDTD 完成正向/伴随电磁仿真，以归一化材料密度为设计变量，结合平滑二值投影、Adam 优化和可选的 KLayout DRC（Design Rule Check）修复，迭代得到高/低折射率材料分布。

可以提供 
3D\2D 强度\电场优化
数据集分类逐样本训练(相位、幅值编码)、基场叠加训练优化
最小线宽/间距DRC约束优化

## 1. 项目能做什么

| 任务 | 入口 | 当前目标与实现 | 每轮主要仿真成本 |
| --- | --- | --- | ---: |
| 3D 强度优化 | `opt_example/opt_intensity_3d.py` | 4 输入、3 输出；当前仅激励输入 1，目标为输出 2；3D TE、强度平方误差 | 1 次正向 + 1 次伴随 |
| Iris 逐样本训练 | `opt_example/opt_iris_training.py` | 4 个归一化特征编码为输入相位，3 个输出对应 one-hot 类别；2D、按 12 个样本组成固定批次 | 12 次正向 + 12 次伴随 |
| Iris 基场叠加训练 | `opt_example/opt_iris_training_basesim.py` | 分别求 4 个输入正向基场和 3 个输出伴随基场，再对完整训练集做线性叠加 | 4 次正向基仿真 + 3 次伴随基仿真 |
| Analog + DRC | `opt_example/opt_analog_drc.py` | 1 个有效输入、3 个输出，目标为输出 3；2D TM、空间滤波、二值化和 KLayout 修复 | 灰度阶段 1 正向 + 1 伴随；DRC 阶段 1 正向 + 1 次 KLayout |


## 2. 目录结构

```text
main_v0p7/
├── configs/                  # 四个任务的静态配置
│   ├── analog_drc.py
│   ├── intensity_3d.py
│   ├── iris_training.py
│   └── iris_training_basesim.py
├── dataset/
│   ├── iris_training_dataset.npz
│   └── iris_test_dataset.npz
├── drc/                      # KLayout DRC
│   ├── drc_45nm.lydrc
│   ├── drc_130nm.lydrc
│   └── drc_180nm.lydrc
├── lsf/                      # Lumerical 建模与 GDS 导出脚本
│   ├── obj_1_3d_coherent_fom.lsf
│   ├── obj_1_m.lsf
│   ├── obj_2_m.lsf
│   ├── obj_3_m.lsf
│   └── gds_contour.lsf
├── opt_example/              # 可直接执行的优化入口
├── opt_results/              # 自动生成的运行结果
├── src/                      # 核心实现
└── README.md
```

核心模块的职责如下。

| 模块 | 主要职责 |
| --- | --- |
| `src/environment.py` | 集中管理项目路径、Lumerical/KLayout 路径，并创建递增编号的结果目录 |
| `src/config.py` | 定义静态配置、派生网格、Lumerical 对象状态、优化器状态和历史记录 |
| `src/setting.py` | 读取 LSF、建立模型、创建逐像素结构或 import 结构，并刷新设计区域 |
| `src/simulation.py` | 2D/3D 正向、伴随以及基场仿真；从监视器取回复电场 |
| `src/fom.py` | 强度误差、交叉熵式损失、伴随源系数和基场叠加梯度 |
| `src/optimization.py` | 网格插值、场乘积梯度、滤波链式求导和 Adam |
| `src/quant.py` | 平滑/硬二值投影、投影导数和 `beta` 延续策略 |
| `src/filter.py` | 均值或“锥形”卷积核以及梯度用翻转核 |
| `src/drc.py` | GDS 处理、KLayout 调用、XML 报告解析和启发式像素修复 |
| `src/plot.py` | FOM 文本日志和 FOM/`beta` 双纵轴曲线 |

## 3. 环境与依赖

### 3.1 软件要求

代码中能直接确认的依赖包括：

- Python。仓库没有声明正式支持范围；现有 `.pyc` 文件来自 CPython 3.9，因此建议优先使用 Python 3.9 复现，其他版本需要自行验证。
- `numpy`、`scipy`、`matplotlib`。
- Ansys Lumerical FDTD 及其 Python API `lumapi`。
- Analog DRC 任务还同时需要：
  - Python 包 `klayout`，代码通过 `import klayout.db as db` 读写 GDS；
  - KLayout 桌面程序/命令行程序，用来执行 `.lydrc` 规则。
- 可用的 Lumerical 许可证；GPU 任务还要求本机的 Lumerical GPU 运行环境可用。

`lumapi` 通常随 Lumerical 安装，不应把一个来源不明的同名包当作替代品。可参考：

- [Ansys 官方：Lumerical Python API 安装与入门](https://optics.ansys.com/hc/en-us/articles/39744901602707-Installation-and-Getting-Started-Python-API)
- [KLayout 官方下载页](https://www.klayout.de/build.html)
- [KLayout 官方 Python 模块说明](https://www.klayout.org/klayout-pypi/)

本版本没有 `requirements.txt`、`pyproject.toml` 或锁文件。以下命令只是按源码导入推导出的基础环境，不代表经过验证的版本锁：

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib klayout
```

### 3.2 配置外部程序路径

`src/environment.py` 内置了开发机默认值：

```python
LUMERICAL_API_PATH = r"D:\Program Files\Lumerical\v241\api\python"
KLAYOUT_PATH = r"D:\Program Files\Klayout\klayout_app.exe"
```

不要为了换机器直接改源码；启动 Python 前用环境变量覆盖：

```powershell
$env:LUMERICAL_API_PATH = 'C:\Program Files\Lumerical\v241\api\python'
$env:KLAYOUT_PATH = 'C:\Program Files\KLayout\klayout_app.exe'
```

如果使用 Ansys Automated Installer，`lumapi` 也可能位于 `C:\Program Files\Ansys Inc\<版本>\Lumerical\api\python`。以本机实际的 `lumapi.py` 为准。

所有示例入口都会先调用 `env.configure()`，再间接导入 `src.setting` 中的 `lumapi`。若在交互式脚本中直接使用核心模块，也必须保持这个顺序：

```python
import src.environment as env

env.configure()

import lumapi
import src.setting
```

### 3.3 启动前自检

在项目根目录执行：

```powershell
python -c "import numpy, scipy, matplotlib; print('scientific stack OK')"
python -c "import src.environment as env; env.configure(); import lumapi; print(lumapi.__file__)"
python -c "import klayout.db as db; print('KLayout Python API OK')"
```

第三条仅 Analog DRC 必需。它只能验证 Python API；仍需单独确认 `KLAYOUT_PATH` 指向可执行文件。

> [!NOTE]
> `FDTDConfig.fdtd_gui` 当前被直接传给 `lumapi.FDTD(hide=fdtd.fdtd_gui)`。所以按现有代码，`fdtd_gui=True` 实际表示隐藏窗口，字段名与真实语义相反。

## 4. 快速开始

### 4.1 先做低成本冒烟测试

默认 `max_epoch=1000`，而一次 FDTD 就可能耗时很长。第一次运行前，先到对应的 `configs/*.py` 中把轮数改为 1～3，并根据机器能力把 `device` 改为 `"CPU"`：

```python
optimizer=OptimizerConfig(max_epoch=2)
```

冒烟测试的最低验收标准是：

1. 能创建 `opt_results/<output_dir>_1/`；
2. Lumerical 能执行对应 LSF，且可以找到配置中声明的源和监视器；
3. 至少完成一轮正向/伴随仿真；
4. `fom.txt` 与 `figure_iteration_1.png` 被写出；
5. 日志中没有 `nan`、`inf`、数组广播或 `getresult` 错误。

### 4.2 运行四个示例

从项目根目录运行：

```powershell
python .\opt_example\opt_intensity_3d.py
python .\opt_example\opt_iris_training.py
python .\opt_example\opt_iris_training_basesim.py
python .\opt_example\opt_analog_drc.py
```

入口会依据自身文件位置把项目根加入 `sys.path`，但仍建议从根目录启动。`src/simulation.py` 中多处使用相对的 `obj.filename` 保存模型，从其他工作目录启动可能产生位置不明确的 `.fsp`/日志文件。

## 5. 配置系统

`src/config.py` 用 dataclass 把静态配置与运行时状态分开。每个 `configs/*.py` 暴露一个 `make_config()`，入口不需要依赖全局可变参数。

### 5.1 `RegionConfig`

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `size_x` | `5e-6` | 设计区 x 尺寸，单位 m |
| `size_y` | `6e-6` | 设计区 y 尺寸，单位 m |
| `size_z` | `220e-9` | 器件厚度，单位 m |
| `pixel_size` | `100e-9` | 设计变量像素尺寸 |
| `z_spacing` | `110e-9` | z 方向离散间距 |
| `sim_spacing` | `20e-9` | 初始导入/插值网格间距 |
| `dz_ratio` | `None` | 非空时使用 Analog 网格：`dz=dx/dz_ratio`，且单元权重是 `dx*dy` |

普通配置得到：

- 设计变量形状 `(y, x) = (60, 50)`；
- z 方向 3 个采样点；
- 初始 x/y 仿真坐标点数分别为 251/301；
- `V_cell = dx * dy * dz`。

Analog 配置把 `size_x` 改为 4 µm、`pixel_size` 改为 20 nm、`dz_ratio` 改为 5，得到：

- 设计变量形状 `(301, 201)`；
- `dx=dy=20 nm`，`dz=4 nm`；
- 二维单元权重 `V_cell=dx*dy`。

当前实现用 `int(size/spacing)` 派生点数，没有检查能否整除。修改网格时应主动校验尺寸、点数和 LSF 监视器坐标是否一致。

### 5.2 `FDTDConfig`

| 字段 | 默认值 | 去向 |
| --- | ---: | --- |
| `fdtd_dimension` | `"3D"` | 覆盖 LSF 中 `FDTD` 对象的 `dimension` |
| `fdtd_gui` | `True` | 当前直接作为 `lumapi.FDTD(hide=...)` 的值 |
| `device` | `"GPU"` | 传给 `model.run("FDTD", device)` |
| `express_mode` | `1` | 写入 `FDTD` 对象的 `express mode` |

这些字段只覆盖少数 FDTD 属性。仿真边界、网格精度、时间、监视器范围、偏振和很多几何参数仍硬编码在 LSF 中。

### 5.3 `OptimizerConfig`

| 字段 | 默认值 |
| --- | ---: |
| `max_epoch` | `1000` |
| `learning_rate` | `0.1` |
| `beta1` / `beta2` | `0.9` / `0.999` |
| `params_min` / `params_max` | `0` / `1` |

这里的 `beta1`/`beta2` 是 Adam 动量参数，不要与二值投影的 `state.beta` 混淆。

### 5.4 `QuantConfig` 与 `MaterialConfig`

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `eta` | `0.5` | 二值投影阈值 |
| `beta0` | `1` | 初始投影锐度 |
| `beta_max` | `500` | 普通任务的投影锐度上限；Analog 配置覆盖为 200 |
| `test_length` | `10` | 收敛判断窗口 |
| `threshold` | `0.1` | 最近窗口内总 FOM 极差阈值 |
| `eps_min` | `1.44²` | 低折射率材料介电常数 |
| `eps_max` | `3.47²` | 高折射率材料介电常数 |

材料在 Python 与 LSF 中都有硬编码。若要替换材料，必须同时检查 `MaterialConfig`、LSF 波导/背景折射率和初始像素折射率。

### 5.5 `ObjectSpec`

`ObjectSpec` 是 Python 与 LSF 的契约：

- `filename`：`lsf/` 下的脚本名；
- `fom_name`：功率/FOM 监视器名；
- `mode_exp_monitor_name`：模式展开监视器名；
- `forward_source_name`、`forward_source_phase`、`forward_source_amp`：前向源名、相位和幅度；
- `adjoint_source_name`：伴随源名；
- `target_fom`：各输出目标；
- `weight`：伴随源系数的附加权重；
- `wavelength`：伴随源中心波长。

以下数量通常必须一致：

```text
len(fom_name)
  == len(mode_exp_monitor_name)
  == len(adjoint_source_name)
  == len(target_fom)
  == len(weight)

len(forward_source_name)
  == len(forward_source_phase)
  == len(forward_source_amp)
```

当前没有自动校验，名称或长度不一致会在 Lumerical `select/getresult` 或 Python 下标访问时才失败。

虽然 `TaskConfig.objects` 是列表，四个入口都只构造 `cfg.objects[0]`；本版本并未真正实现多对象联合优化。

### 5.6 `DatasetConfig` 与 `DRCConfig`

`DatasetConfig` 默认读取 `dataset/iris_training_dataset.npz`，批大小为 12。

`DRCConfig` 的默认值为：

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `gds_name` | `"gds_result"` | GDS 文件名前缀 |
| `drc_script` | `"drc_45nm.lydrc"` | KLayout 规则 |
| `drc_threshold` | `0.05` | DRC 几何判断阈值；按当前坐标换算，单位为 µm，即 50 nm |
| `param_delta` | `0.2` | 归一化设计变量的启发式修正步长 |
| `filter_R0` / `filter_R_min` / `filter_R_max` | `3 / 3 / 8` | 以设计像素为单位的滤波半径配置 |

## 6. 核心数据模型

### 6.1 `Region`

`Region` 从 `RegionConfig` 派生设计坐标、仿真坐标、网格步长和单元体积/面积。Python 侧统一把设计矩阵组织为 `(y, x)`。

正向仿真后，普通 2D/3D 路径会用 `opt_fields` 监视器返回的 x/y 坐标覆盖 `region.sim_x_pos` 和 `region.sim_y_pos`。因此 `sim_spacing` 只是初始化时的假定网格，真实梯度插值以监视器坐标为准。

### 6.2 `Object`

`Object` 将静态的 `ObjectSpec` 与以下运行时数据放在一起：

- `model`：Lumerical 会话；
- `fom`：每个输出的当前损失；
- `factor`：构造伴随源的复系数；
- `E_for` / `E_adj`：正向/伴随设计区复电场；
- `E_fom`：基场路径中各输入到各输出监视器的场；
- `E_fom_desire_mode`：基场路径的目标模式场。

按当前 `np.rot90` 和 `np.stack` 逻辑，代码期望：

- 2D 场形状接近 `(y_sim, x_sim, 3)`；
- 3D 场形状接近 `(y_sim, x_sim, z_sim, 3)`；
- 最后一维为 `Ex/Ey/Ez`。

源码没有 shape 断言，网格或监视器设置变化时应首先检查这些数组。

### 6.3 `OptimState` 与 `History`

`OptimState` 保存：

- `params`：归一化设计变量 \(\rho\)；
- `eps_opt`：投影后的介电常数；
- `index_opt`：\(\sqrt{\epsilon}\)；
- `m`、`v`：Adam 一、二阶矩；
- `iteration`、`beta`；
- Analog 专用的 `filter_R`、`params_all`、`params_conv` 和滤波核。

`History` 保存每轮各输出 FOM、总 FOM、`beta` 以及 Analog 的未收敛计数。

## 7. 算法原理与代码对应

### 7.1 设计变量和二值投影

归一化参数满足：

\[
\rho \in [0,1]
\]

平滑投影为：

\[
\epsilon(\rho,\beta)=\epsilon_{\min}
+\frac{\tanh(\beta\eta)+\tanh(\beta(\rho-\eta))}
{\tanh(\beta\eta)+\tanh(\beta(1-\eta))}
(\epsilon_{\max}-\epsilon_{\min})
\]

然后：

\[
n=\sqrt{\epsilon}
\]

`src/quant.py::quant_1bit()` 实现上述投影。当 `beta > beta_max` 时，代码改用 `rho >= eta` 的硬阈值。`dq_drho()` 给出平滑阶段的解析导数。

普通收敛判断取最近 `test_length` 轮总 FOM：

\[
\max(L)-\min(L)<\text{threshold}
\]

满足条件时执行 `beta *= 1.2`。Analog 路径会先尝试增加滤波半径，再提高 `beta`。

### 7.2 强度目标

模式展开结果构造复传输系数：

\[
c_i=a_i\sqrt{\operatorname{Re}(N_i)}
\]

归一化传输强度：

\[
T_i=\operatorname{Re}\left(\frac{c_ic_i^*}{P_\text{source}}\right)
\]

每个输出的损失：

\[
L_i=\left|y_i-T_i\right|^2
\]

`src/fom.py::get_fom_intensity()` 同时根据传输系数、角频率、源功率、`V_cell` 和真空介电常数生成 `obj.factor`，供伴随仿真设置源相位和幅度。

### 7.3 Iris 交叉熵式目标

代码把各输出传输归一化为：

\[
p_i=\frac{T_i}{\sum_jT_j}
\]

并记录：

\[
L_i=-y_i\log(p_i)
\]

one-hot 标签只有目标输出项非零，总 FOM 是各项之和。实现位于 `get_fom_ce()` 和 `get_fom_ce_gradient_parallel()`。


### 7.4 伴随梯度

代码中的 2D 介电常数梯度为：

\[
g_\epsilon=-2\operatorname{Re}
\left(\sum_c E_{\mathrm{for},c}E_{\mathrm{adj},c}\right)
\]

3D 路径在场分量求和后再沿 z 方向取均值。随后使用 `RegularGridInterpolator` 把仿真网格线性插值到设计网格。

### 7.5 Adam 更新

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

\[
\rho_t=\operatorname{clip}
\left(
\rho_{t-1}-\alpha
\frac{\hat m_t}{\sqrt{\hat v_t}+10^{-8}},
\rho_{\min},\rho_{\max}
\right)
\]

`src/optimization.py::adam_opt()` 实现最小化方向的更新和 `[0,1]` 裁剪。

## 8. 四条执行链详解

### 8.1 3D 强度优化

配置：`configs/intensity_3d.py`。

当前参数：

- LSF：`obj_1_3d_coherent_fom.lsf`；
- 4 个前向源、3 个输出/伴随源；
- 前向幅度 `[1,0,0,0]`，即只激励输入 1；
- 目标 `[0,1,0]`，即最小化与“输出 2 强度为 1，其余为 0”的偏差；
- 1550 nm、3D、GPU、express mode；
- 设计区由 60×50 个独立矩形像素组成。

单轮顺序：

1. `refresh_design_region_3d_rect()` 逐像素写入当前折射率；
2. `make_forward_sim_3d()` 开启前向源并取得 `opt_fields`；
3. `get_fom_intensity()` 计算三路损失和伴随系数；
4. 写 `fom.txt` 并检查最近窗口的总损失；
5. `make_adjoint_sim_3d()` 同时设置三路伴随源；
6. 场乘积、z 平均、插值得到 `grad_eps`；
7. 乘投影导数，Adam 更新 `params`；
8. 更新 `eps_opt/index_opt` 并绘图；
9. `beta` 达上限后再记录一轮并退出。

逐像素结构易理解，但创建和更新数千个 Lumerical 对象的成本较高。

### 8.2 Iris 逐样本训练

配置：`configs/iris_training.py`。

训练集 120 个样本按保存顺序切成 10 个固定批次，每批 12 个。第 `i` 轮选择：

```python
batch_index = (i - 1) % 10
```

对每个样本：

```text
样本前 4 项 × 360°
    → 四个前向源的 phase（度）

样本后 3 项
    → 三个输出的 one-hot target_fom
```

随后刷新相同的设计结构、运行一次正向和一次伴随 FDTD，并累计损失和梯度。

> [!WARNING]
> 当前配置的 `forward_source_amp=[1,0,0,0]`，所以第 2～4 路源虽然修改了相位，幅度仍为 0；同时代码计算了批平均 `grad_all`，最终 Adam 却使用最后一个样本的 `grad_eps`。当前入口不能被视为正确的四特征 mini-batch 训练实现。

### 8.3 Iris 基场叠加训练

配置：`configs/iris_training_basesim.py`。

每轮先执行：

1. 依次仅开启 4 个输入源，每个幅度为 1、相位为 0，保存设计区正向基场和三个输出监视器场；
2. 依次仅开启 3 个伴随源，每个幅度为 1、相位为 0，保存伴随基场。

对特征 \(x_k\) 构造：

\[
a_k=e^{i2\pi x_k}
\]

利用 Maxwell 方程在线性材料中的场叠加：

\[
E_\mathrm{for}=\sum_k a_kE_\mathrm{for}^{(k)}
\]

\[
E_\mathrm{adj}=\sum_j f_jE_\mathrm{adj}^{(j)}
\]

然后在 Python 中遍历全部 120 个训练样本，累计并平均 FOM 与梯度。该入口忽略 `DatasetConfig.batch_size`，每轮固定处理完整训练集。

相比逐样本版，FDTD 次数从名义上的 240 次/完整数据集降为 7 次/轮，但会保存多组大型复场数组，并依赖线性叠加、模式重叠和伴随归一化全部正确。建议先在小网格上比较“直接仿真梯度、基场梯度、有限差分梯度”。

### 8.4 Analog + DRC

配置：`configs/analog_drc.py`。

该路径使用 `obj_1_m.lsf`：

- 2D TM；
- 4 µm × 6 µm 设计区；
- 20 nm 设计步长；
- 一个有效前向源、三个输出/伴随源；
- 目标 `[0,0,1]`；
- CPU、`beta_max=200`；
- 默认使用 45 nm DRC 规则。

初始化时：

1. 将 LSF 中的 `filter_r = 3;` 替换为 `filter_R_max`；
2. 覆盖 `fields_mesh` 的 dx/dy/dz；
3. 创建 import 折射率结构；
4. 从 `opt_index` 读取带滤波边界的介电常数，构造 `params_all`。

灰度阶段尝试执行：

```text
raw params
→ 空间滤波 params_conv
→ 平滑投影 epsilon
→ index
→ 正向/伴随 FDTD
→ 卷积链式梯度
→ Adam
```

但当前正向结构实际由 raw `state.params` 投影得到，反向链却按 `state.params_conv` 计算；详见优化路线中的 P0-3。

`beta` 达到上限后：

1. `gds_contour.lsf` 从 `opt_index` 的 `index_x=2.7` 等值线导出 layer 1 GDS；
2. `src/drc.py` 用 KLayout Python API 处理 GDS；
3. 调用 `klayout_app.exe -r <rule> -i <gds>`；
4. 解析 XML 报告；
5. 仅针对最小线宽和同层最小间距两类违例修改附近像素；
6. 两类违例都消失时保存模型并退出。

当前三份规则的实际赋值为：

| 规则文件 | 最小宽度 | 最小间距 |
| --- | ---: | ---: |
| `drc_45nm.lydrc` | 0.045 µm | 0.055 µm |
| `drc_130nm.lydrc` | 0.13 µm | 0.16 µm |
| `drc_180nm.lydrc` | 0.18 µm | 0.22 µm |


## 9. 数据集格式

仓库内 NPZ 的只读统计如下。

| 文件 | 样本数 | 单样本 | 标签计数 |
| --- | ---: | --- | --- |
| `iris_training_dataset.npz` | 120 | shape `(7,)`、`float64` | 38 / 41 / 41 |
| `iris_test_dataset.npz` | 30 | shape `(7,)`、`float64` | 12 / 9 / 9 |

键为 `arr_0`、`arr_1`……，单样本的代码约定是：

```text
[phase_feature_0,
 phase_feature_1,
 phase_feature_2,
 phase_feature_3,
 class_0,
 class_1,
 class_2]
```

- 前 4 项均在 `[0,1]` 内，表示一个相位周期的比例；
- 逐样本入口用 `feature * 360°`；
- 基场入口用 `exp(i * 2π * feature)`；
- 后 3 项是 one-hot 标签。

仓库未说明原始 Iris 四个特征如何预处理，也未说明三个输出端与具体花卉类别的名称映射，不能仅凭当前代码推断。

`iris_test_dataset.npz` 当前没有被任何入口读取，因此本版本没有准确率、混淆矩阵或独立测试集评估流程。

逐样本训练依赖 `data.files` 的保存顺序：

- 不打乱样本；
- 数据量小于批大小时会发生取模除零；
- 数据量不能整除批大小时，尾部样本会被忽略。

## 10. LSF 模型契约

### 10.1 文件用途

| 文件 | 当前用途 |
| --- | --- |
| `obj_1_3d_coherent_fom.lsf` | 4 输入、3 输出、1550 nm、TE、5×6 µm；供 3D 强度和两个 Iris 入口使用 |
| `obj_1_m.lsf` | 3×3 波导几何中的输入 1、三个输出，1550 nm、TM、4×6 µm；供 Analog 使用 |
| `obj_2_m.lsf` | 1560 nm 辅助模板；当前无 Python 配置引用 |
| `obj_3_m.lsf` | 1570 nm 辅助模板；当前无 Python 配置引用 |
| `gds_contour.lsf` | 从 `opt_index` 提取等值线并导出 GDS |

### 10.2 必须存在的对象名

如果新建 LSF，至少要按所选 Python 路径提供这些对象：

| 名称 | 用途 |
| --- | --- |
| `FDTD` | 被 Python 覆盖 dimension/express mode 并启动 |
| `fields_mesh` | Analog 路径覆盖网格 |
| `opt_fields` | 返回设计区 `Ex/Ey/Ez` 以及 x/y 坐标 |
| `import` 或 `design_pixel` | 两种设计区表示 |
| `opt_index` | Analog 初始化和 GDS 轮廓提取 |
| `fom_1...N` | 输出场/功率监视器 |
| `fom_exp_1...N` | 模式展开监视器；结果名需符合 `"expansion for " + monitor_name` |
| `forward_source_1...N` | 正向模式源 |
| `adjoint_source_1...N` | 反向模式源 |

基场路径还会从模式展开监视器读取 `"mode profiles"["E1"]`。

### 10.3 Python 和 LSF 的双重配置风险

目前这些信息不是单一事实源：

- 设计区尺寸；
- 波导与背景折射率；
- 输入/输出数量；
- 波长与偏振；
- FDTD 边界和网格；
- 监视器位置与名称；
- Analog 滤波边界。

例如修改 `RegionConfig.size_x` 并不会自动修改 LSF 中的 `opt_size_x`。二者漂移时，最常见结果是场数组 shape 错误、插值错位，或结构与监视器不在同一物理区域。

## 11. 输出、日志与恢复

### 11.1 输出目录

每次启动都会扫描 `opt_results/` 并创建第一个不存在的编号：

```text
opt_results/
└── sim_100nm_N/
    ├── fom.txt
    ├── figure_iteration_1.png
    ├── figure_iteration_2.png
    ├── ...
    ├── grey_complete.*               # 普通量化阶段保存，后缀由 Lumerical 决定
    ├── dfm.txt                       # 仅 Analog DRC
    ├── interation<N>.*               # DRC 通过模型；源码当前拼写如此
    └── sim_file/
        ├── <模型>.fsp
        ├── <模型>_p0.log
        ├── gds_result_<N>.gds        # 仅 DRC 阶段
        └── gds_result_<N>_drc_result
```

不同任务只会生成其中一部分文件。当前编号创建方式不是并发安全的，不要同时用相同 `output_dir` 启动多个进程。

### 11.2 `fom.txt`

普通和训练路径每轮写一行：

```text
Iteration <i>\t<fom_1,fom_2,...,total>\t<beta>
```

`figure_iteration_<i>.png` 用对数坐标绘制完整 FOM 历史和 `beta`。默认 1000 轮会产生 1000 张重复包含完整历史的图片。

Analog 的 `dfm.txt` 格式为：

```text
Iteration <i>\t<HM.1 最小线宽违例数>\t<HM.2 最小间距违例数>
```

### 11.3 断点恢复

`TaskConfig.resume` 已预留 NPZ 加载接口，文件必须包含：

```text
eps_save
params
m
v
all_iteration
beta
```

数组 shape 应与当前 `(y_points, x_points)` 一致。

## 12. 如何新增一个任务

建议按以下顺序扩展：

1. 复制最接近的 `configs/*.py`，只在 `make_config()` 内声明新参数；
2. 复制并修改 LSF，确认 `FDTD`、设计区、源、监视器和模式展开对象名与 `ObjectSpec` 完全一致；
3. 决定设计区表示：
   - 小型、教学用 3D：`initialize_model_3d_rect()`；
   - 连续折射率导入：`initialize_model_import()`；
   - 带滤波边界的 2D：`initialize_model_import_analog()`；
4. 选择正向/伴随函数和 FOM，确保场维度匹配；
5. 在小网格上对一个或少量像素做中心有限差分：

\[
g_k^\mathrm{FD}
=
\frac{L(\rho_k+h)-L(\rho_k-h)}{2h}
\]

并比较伴随梯度的符号、方向相关系数和尺度；

6. 再开启 `beta` 延续、批训练或 DRC；
7. 保存完整配置快照、依赖版本、数据集摘要和随机/批次顺序。

在当前入口结构下，最简单的做法是复制一个 `opt_example/*.py`。

## 13. 常见故障

### `ModuleNotFoundError: lumapi`

- 确认 `LUMERICAL_API_PATH` 指向包含 `lumapi.py` 的目录；
- 环境变量必须在启动 Python 前设置；
- 不要在调用 `env.configure()` 前直接导入 `src.setting`。

### Lumerical 许可证、GPU 或运行设备错误

- 确认 FDTD 许可证可用；
- 先用 `device="CPU"` 和 `express_mode=0` 缩小问题范围；
- 检查所用 Lumerical 版本是否支持当前 `model.run("FDTD", device)` 调用。

### `ModuleNotFoundError: klayout`

安装的是 KLayout 桌面程序并不等于 Python 环境里存在 `klayout.db`。Analog 路径需要两者。

### `subprocess.CalledProcessError` 或 DRC 报告不存在

- 检查 `KLAYOUT_PATH`；
- 检查 GDS 是否已生成；
- 检查规则文件和 layer `(1,0)`；
- 检查 KLayout 的工作目录以及 `<gds_base>_drc_result` 的实际位置；
- 当前 subprocess 没有捕获标准输出到项目日志，也没有超时。

### Lumerical `select/getresult` 找不到对象

逐一对照配置与 LSF 中的：

- `FDTD`、`fields_mesh`；
- `opt_fields`、`opt_index`、`import`/`design_pixel`；
- `fom_*`、`fom_exp_*`；
- `forward_source_*`、`adjoint_source_*`。

### 插值、广播或 shape 错误

- 检查 Python 设计区尺寸与 LSF `opt_size_*`；
- 打印 `E_for.shape`、`E_adj.shape`、`region.sim_*_pos.shape`；
- 检查 resume 数组是否为 `(y,x)`；
- 检查样本是否严格为 4 个特征 + 3 个标签。

### 初始化或刷新永久等待

`initial_rect()` 和 `refresh_design_region_3d_rect()` 用无超时的 `while True` 等待 Lumerical 创建像素/`flag`。若 LSF 执行失败，Python 不会自行退出。此时应检查 Lumerical 脚本控制台和对象树，并为轮询增加超时。

### 磁盘快速增长

每轮都会保存完整历史图，仿真自身也可能生成 `.fsp` 和 `_p0.log`。调试时降低轮数，正式实现中增加 `plot_interval`、`checkpoint_interval` 和保留策略。



