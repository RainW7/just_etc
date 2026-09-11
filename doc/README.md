# JUST 曝光时间计算器 (Exposure Time Calculator, JUST ETC v1)

本目录为 **上海交通大学 4.4 米光谱望远镜 (JUST, Jiaotong University Spectroscopic Telescope) 曝光时间计算器 (ETC)** 的标准封装包与核心代码库，适用于本地科学分析、服务器部署及交大 Gravity 超算集群批处理任务。

---

## 1. 目录结构与文件速查

```text
/home/wenrun/ETC_py_v1/
├── just_etc_api.py               # [核心接口] 高层 Python API (推荐入口类与函数)
├── ETC_py_optimized.py           # [核心引擎] Numba JIT 矢量化光路与信噪比物理计算引擎
├── modeldata.py                  # [物理数据库] 大气透射查找表、UVES 夜天光线与 OH 气辉线库
├── spec.dat                      # [硬件配置] JUST 望远镜光学、光纤及三分光臂参数表
├── JUST_ETC_User_Guide.md        # [技术手册] 详细物理模型、数学公式与光学参数白皮书
│
├── cal_exp_time.py               # [命令行工具] 快速计算 SNR 或逆向求解曝光时间
├── generate_just_limit.py        # [极限星等] 连续谱/发射线极限星等与曝光时间关系生成工具
├── simulate_mock_spectrum.py     # [模拟谱生成] 结合物理噪声模型的 JUST 模拟观测谱生成工具
├── dwarf_ssp_model.py            # [星族模型] 基于恒星质量与恒星形成率的矮星系物理光谱模型
├── calculate_star_snr.py         # [恒星测试] 点源恒星多波段 SNR 快速评估工具
├── run_etc_preset.py             # [批处理预设] 非交互式预设参数批处理运行脚本
│
├── process_bgs_10k_redrock.py    # [万条光谱并行] 10,000 条光谱多进程并行仿真与 Redrock FITS 生成脚本
├── submit_bgs10k_etc.qsub        # [集群调度] 上海交大 Gravity 集群 PBS (qsub) 任务提交脚本
│
├── templates/                    # [FITS SED 模板库] 覆盖各种星系与恒星的高分辨率光谱
│   ├── galaxy/                   # 椭圆星系 (elliptical)、螺旋星系 (spiral, sa, sbc, scd) 等
│   ├── starburst/                # 星暴星系模板 (sb1 ~ sb6, Kinney et al.)
│   ├── bc03/                     # Bruzual & Charlot (2003) 单星族演化合成模板
│   └── IRgalaxy/                 # 红外星系与星暴并合模板 (M82, Arp220, Mrk231, NGC6240 等)
│
├── convert_input_fits_to_redrock.py  # FITS 输入转 DESI/Redrock 标准格式多波段数据工具
├── generate_just_specdat_fits.py     # 将 spec.dat 效率曲线转为标准 Throughput FITS 工具
├── generate_redrock_just_fits.py     # 生成单条/小批量模拟观测 Redrock Fits 数据
│
├── plot_demo.py                  # [演示脚本] 基础 SNR 与光谱可视化绘图
├── plot_demo_extended.py         # [演示脚本] 多观测条件 (视宁度/星等) 对比绘图
├── plot_fiber_comparison.py      # [分析脚本] 不同天体尺寸下的光纤透过率对比
├── plot_fiber_injection.py       # [分析脚本] 光纤几何注入效率与偏心误差分析
├── plot_fiber_snr_comparison.py  # [分析脚本] 光纤孔径与天体形态对 SNR 影响评估
├── photon_loss_analysis.py       # [分析脚本] 光子损耗级联瀑布图 (Photon Loss Cascade)
├── photon_loss_seeing.py         # [分析脚本] 大气视宁度对各波段光子损耗影响分析
├── run_survey_simulations.py     # [巡天模拟] 端到端巡天效率综合模拟脚本
│
├── dwarf/                        # [矮星系专题] 极低表面亮度矮星系可观测性与运动学分析
│   └── simulate_dwarf_observability.py
├── limit_mag/                    # [极限星等数据] 极限星等计算产出图表与数据表
├── compare/                      # [对比分析] JUST ETC 与 justspecsimu / PFS 参数对比报告
│
├── requirements.txt              # Python 环境依赖清单 (含 Numba JIT 加速)
├── README.md                     # 本使用手册 (中文版)
└── README_EN.md                  # 本使用手册 (英文版)
```

---

## 2. 环境配置与安装

推荐在 Conda 独立环境中运行：

```bash
# 1. 激活 Conda 环境
conda activate base

# 2. 安装核心依赖包
pip install -r requirements.txt
```

核心依赖包说明：
- `numpy >= 1.20.0`：基础数组与矩阵运算
- `astropy >= 4.0`：FITS 文件读写、测光系统与天文常量
- `scipy >= 1.5.0`：高斯滤波、数值插值与特殊函数积分 (Bessel)
- `matplotlib >= 3.3.0`：出版级科学图表绘制
- `numba >= 0.53.0`：**JIT 编译加速库**（针对底层光路追踪循环与数值积分进行机器码级加速）

*(可选依赖：如需在服务器上直接对接 Redrock 进行红移测量，需安装 `redrock` 与 `desispec`)*

---

## 3. Python API 详细使用指南 (`just_etc_api.py`)

在自定义 Python 脚本或 Jupyter Notebook 中，`just_etc_api.py` 提供了面向用户的最简便接口。

### 3.1 实例化与观测环境配置

```python
from just_etc_api import JUSTExposureTimeCalculator, load_template, normalize_to_mag

# 1. 实例化计算器 (精度模式可选: 'fast', 'balanced', 或 'accurate')
# 'fast' 适用于海量光谱仿真或初步估算；'accurate' 适用于单谱最终出图
etc = JUSTExposureTimeCalculator(calc_mode='fast')

# 2. 自定义设置当前观测条件 (支持任意关键词参数)
etc.set_obs_conditions(
    seeing_fwhm_800=0.8,    # 800 nm 处视宁度 FWHM (角秒, 默认 0.8")
    zenith_angle=45.0,      # 天体天顶角 (度, 对应大气质量 X=1.414)
    ebv=0.03,               # 银河系前相色余 E(B-V)
    decenter=0.03,          # 光纤中心对准误差 (角秒, 默认 0.03")
    r_eff=0.5,              # 扩展源半光半径 (角秒, 点源/恒星设为 0.0)
    lunar_za=135.0          # 月球天顶角 (>90° 为月亮在地平线下，即暗夜 Dark Time)
)
```

### 3.2 光谱模板加载与光度定标

```python
# 1. 从 templates 目录加载内置 FITS 光谱模板 (支持相对路径或完整路径)
wave_aa, flux_flam = load_template('galaxy/elliptical_001.fits')

# 2. 将模板光谱归一化到指定测光波段与 AB 星等 (例如 r = 21.5 mag)
# 支持波段: 'u', 'g', 'r', 'i', 'z', 'V', 'B'
flux_norm, scale_factor = normalize_to_mag(
    wave_aa=wave_aa,
    flux_flam=flux_flam,
    target_mag=21.5,
    band='r'
)
print(f"光谱定标完成，缩放系数: {scale_factor:.4e}")
```

### 3.3 计算三分光臂信噪比 (`compute_snr`)

```python
# 计算 4 次 900 秒曝光 (总计 3600 秒) 的全波段 SNR
results = etc.compute_snr(
    wave_aa=wave_aa,
    flux_flam=flux_norm,
    t_exp=900.0,
    n_exp=4
)

# 打印各分光臂输出结果
arm_names = {0: 'Blue (365-560 nm)', 1: 'Green/Red (540-745 nm)', 2: 'Z/NIR (720-925 nm)'}
for arm in results:
    ia = arm['arm']
    print(f"[{arm_names[ia]}] Mean SNR: {arm['snr_mean']:.2f}, Median SNR: {arm['snr_median']:.2f}")
    # 获取逐像素波长与信噪比数组:
    # wave_nm = arm['wave_nm'], snr_array = arm['snr']
```

### 3.4 模拟含噪声的实际观测光谱 (`simulate_mock_observation`)

```python
# 模拟经过 JUST 望远镜与仪器噪声影响后的实际输出光谱
mock_obs = etc.simulate_mock_observation(
    wave_aa=wave_aa,
    flux_flam=flux_norm,
    t_exp=900.0,
    n_exp=4,
    seed=42  # 随机数种子，保证结果可复现
)

for arm in mock_obs:
    print(f"Arm {arm['arm']}:")
    print(f"  - 真实无噪谱: arm['flux_intrinsic']")
    print(f"  - 观测含噪谱: arm['flux_mock']")
    print(f"  - 噪声电子数: arm['noise_e']")
```

### 3.5 逆向求解所需曝光时间 (`solve_exposure_time`)

```python
# 求解在 600 nm 处达到 SNR = 8.0 所需的曝光时长 (划分为 4 次曝光)
solution = etc.solve_exposure_time(
    wave_aa=wave_aa,
    flux_flam=flux_norm,
    target_snr=8.0,
    ref_wave_nm=600.0,
    n_exp=4,
    t_exp_init=600.0
)

print(f"建议单次曝光时间: {solution['t_exp']:.1f} 秒")
print(f"总计观测时间    : {solution['t_total']/3600:.2f} 小时")
print(f"精确达到的 SNR  : {solution['snr_achieved']:.3f} (收敛迭代步数: {solution['n_iter']})")
```

---

## 4. 命令行工具使用说明 (CLI Tools)

### 4.1 `cal_exp_time.py`：交互式与命令行计算工具

支持全套 `argparse` 命令行参数：

```bash
cd /home/wenrun/ETC_py_v1

# 示例 1: 计算点源恒星 (r = 21.5 mag) 在 1800 秒曝光下的 SNR
python cal_exp_time.py --mag 21.5 --band r --texp 1800 --target point

# 示例 2: 反向求解扩展源星系 (r = 22.0 mag, Reff = 0.6") 达到 SNR = 5 所需曝光时间
python cal_exp_time.py --target-snr 5.0 --mag 22.0 --band r --target extended --reff 0.6

# 示例 3: 指定自定义模板与较差视宁度条件 (Seeing = 1.2")
python cal_exp_time.py --template galaxy/sbc_cww_001.fits --mag 20.5 --band g --seeing 1.2 --texp 900 --nexp 2

# 示例 4: 直接运行默认演示
python cal_exp_time.py
```

### 4.2 `generate_just_limit.py`：极限星等与 SNR 曲线计算

批量模拟不同曝光时间下的连续谱及发射线极限星等曲线，生成数据表与对应图像：

```bash
python generate_just_limit.py
# 输出文件保存至 limit_mag/ 目录
```

### 4.3 `simulate_mock_spectrum.py`：基于物理星族模型的模拟谱生成

结合星系总恒星质量 $M_*$、比恒星形成率 sSFR 及 Kennicutt (1998) 电离线模型合成光谱并模拟观测：

```bash
python simulate_mock_spectrum.py
# 输出图像保存至 output/mock_spectra/
```

---

## 5. 科学测试与结果演示脚本

| 脚本名称 | 主要功能与科学意义 | 运行命令 | 产出图像 / 文件 |
| :--- | :--- | :--- | :--- |
| `plot_demo.py` | 绘制基础星系模板光谱、各分光臂透过率及 SNR 曲线 | `python plot_demo.py` | `output/etc_demo_plot.png` |
| `plot_demo_extended.py` | 对比不同星等 ($r=18\sim 23$) 与视宁度下的 SNR 演化 | `python plot_demo_extended.py` | `output/etc_reff_impact.png` 等 |
| `photon_loss_analysis.py` | 分析端到端光子能量在各环节损耗占比 | `python photon_loss_analysis.py` | `output/photon_loss_cascade.png` |
| `photon_loss_seeing.py` | 评估视宁度恶化对蓝/红/紫三臂光纤耦合损失的影响 | `python photon_loss_seeing.py` | `output/photon_loss_seeing_cascade.png` |
| `plot_fiber_comparison.py`| 对比点源与不同尺度扩展源 ($0.3'', 0.8'', 1.2''$) 的集光效率 | `python plot_fiber_comparison.py` | `output/fiber_loss_comparison.png` |
| `dwarf/simulate_dwarf_observability.py`| 针对低表面亮度矮星系 (dIrr, BCD, dE) 的 SNR 与运动学模拟 | `python dwarf/simulate_dwarf_observability.py` | `dwarf/output/*.png` |

---

## 6. 交大 Gravity 集群 10,000 条光谱批量仿真与 PBS (`qsub`) 任务提交

针对 `/home/yzgu/work/work-just/justspecsimu/milestone2026/spectrum_library/BGS_z01/input-spectra.fits` 存储的 10,000 条 BGS 光谱库：

### 6.1 方式 A：使用 PBS 提交集群批处理作业 (推荐)

集群作业脚本 `submit_bgs10k_etc.qsub` 已配置好 16 核多进程并行与环境自适应激活：

```bash
cd /home/wenrun/ETC_py_v1

# 1. 提交作业至集群队列
qsub submit_bgs10k_etc.qsub

# 2. 查看当前作业运行状态
qstat -u $USER

# 3. 实时查看运行日志
tail -f bgs10k.log
```

### 6.2 方式 B：在计算节点交互式直接运行

```bash
python process_bgs_10k_redrock.py \
    --input /home/yzgu/work/work-just/justspecsimu/milestone2026/spectrum_library/BGS_z01/input-spectra.fits \
    --output /home/wenrun/output/just_redrock_bgs_10k_obs.fits \
    --t_exp 900 \
    --n_exp 4 \
    --seeing 0.8 \
    --nproc 16
```

### 6.3 输出 FITS 格式说明与 Redrock 测量对接
生成的 `just_redrock_bgs_10k_obs.fits` 完全符合 DESI / Redrock 标准格式：
- `B_WAVELENGTH`, `B_FLUX`, `B_IVAR`, `B_RESOLUTION`
- `R_WAVELENGTH`, `R_FLUX`, `R_IVAR`, `R_RESOLUTION`
- `Z_WAVELENGTH`, `Z_FLUX`, `Z_IVAR`, `Z_RESOLUTION`
- `FIBERMAP`, `SCORES`

可直接传入 `rrdesi` 测量红移：
```bash
rrdesi --zbest /home/wenrun/output/redrock_zbest.fits /home/wenrun/output/just_redrock_bgs_10k_obs.fits
```

