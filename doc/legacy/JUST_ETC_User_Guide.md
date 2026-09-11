> Historical pre-packaging document. Paths and examples may be obsolete.
> Use the repository README.rst and doc/usage.md for current instructions.

# JUST 曝光时间计算器 (Exposure Time Calculator, ETC) 技术手册与用户指南

> **版本**：v1.0 (Python Vectorized & JIT Engine)  
> **适用项目**：JUST (Jiaotong University Spectroscopic Telescope, 上海交通大学 4.4 米光谱望远镜)  
> **维护环境**：`base` (Python >= 3.8, NumPy, SciPy, Astropy, Matplotlib, Numba)  
> **主要目标**：详细阐述 JUST ETC 的物理建模体系、光学与仪器参数、数值计算算法（光纤注入效率、大气与夜空模型、CCD 噪声方程、逆向曝光求解器）及使用规范。

---

## 1. 望远镜与光谱仪硬件参数详解

### 1.1 望远镜光学系统 (Telescope Optics)
JUST 望远镜是一台专为大规模高通量光谱巡天设计的 4.4 米光学红外望远镜：
- **主镜有效通光口径 ($D_\text{tel}$)**：$4.4\text{ m}$
- **副镜中央遮挡直径 ($D_\text{obs}$)**：$1.8\text{ m}$（中央线性遮挡比 $\epsilon = 1.8 / 4.4 \approx 0.41$）
- **有效几何集光面积 ($A_\text{geom}$)**：$A_\text{geom} = \frac{\pi}{4}(4.4^2 - 1.8^2) \approx 12.66\text{ m}^2$
- **主焦点有效焦距 ($EFL$)**：$26.5\text{ m}$，对应主焦点工作焦比为 $f/5.5$
- **视场角半径 ($\theta_\text{FoV}$)**：$0.6^\circ$（总全视场直径 $1.2^\circ$）
- **焦平面物理比例尺 (Plate Scale)**：
  $$PS = \frac{180 \times 3600}{\pi \times 26500} \approx 7.784\text{ arcsec/mm} \quad (\approx 128.47\text{ }\mu\text{m/arcsec})$$

### 1.2 多目标光纤定位系统 (Fiber Positioner System)
- **光纤物理纤芯直径 ($d_\text{core}$)**：$175\text{ }\mu\text{m}$（焦点处光纤有效半径 $r_\text{fiber} = 87.5\text{ }\mu\text{m}$）
- **天球投影光纤角直径**：$\theta_\text{fiber} = 175\text{ }\mu\text{m} \times 0.007784\text{ arcsec/}\mu\text{m} \approx 1.362''$（角半径 $r_0 \approx 0.681''$）
- **焦平面像斑质量 ($SPOT$)**：从视场中心至边缘 5 个等距采样点 ($0.0^\circ, 0.15^\circ, 0.3^\circ, 0.45^\circ, 0.6^\circ$) 的光学 RMS 像斑大小分别为 `19.1, 20.9, 24.4, 27.8, 31.3` $\mu\text{m}$。
- **光学渐晕因子 ($VIGNET$)**：从视场中心到边缘 5 点对应的渐晕透过率分别为 `0.85, 0.84, 0.83, 0.81, 0.79`。

### 1.3 三分光臂光谱仪系统 (Three-Arm Spectrograph)
JUST 摄谱仪采用双向色向分光镜（Dichroics）将光束分离为三个独立的分光臂（Arm 0 / Arm 1 / Arm 2），配置如下：

| 分光臂 | 波长覆盖范围 ($\lambda$) | 探测器像素数 | 色散率 ($\Delta\lambda_\text{pix}$) | 光谱分辨率 ($R = \lambda/\Delta\lambda$) | 谱线 Trace 宽度 | 读出噪声 ($\sigma_\text{read}$) | 暗电流 ($I_\text{dark}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm 0 (Blue)** | $365.0 \sim 560.0\text{ nm}$ | 4096 像素 | $\approx 0.050\text{ nm/pix}$ | $R \approx 2000 \sim 3000$ | 7 像素 | $4.0\,e^-\text{ RMS}$ | $0.010\,e^-/\text{pix/s}$ |
| **Arm 1 (Green/Red)** | $540.0 \sim 745.0\text{ nm}$ | 4096 像素 | $\approx 0.050\text{ nm/pix}$ | $R \approx 2500 \sim 3500$ | 7 像素 | $4.0\,e^-\text{ RMS}$ | $0.005\,e^-/\text{pix/s}$ |
| **Arm 2 (Z/NIR)** | $720.0 \sim 925.0\text{ nm}$ | 4096 像素 | $\approx 0.050\text{ nm/pix}$ | $R \approx 3000 \sim 4000$ | 7 像素 | $4.0\,e^-\text{ RMS}$ | $0.002\,e^-/\text{pix/s}$ |

---

## 2. 物理算法与数学建模体系

JUST ETC 的核心目标是模拟光子从外层空间穿越星际介质、地球大气、望远镜光学系统、光纤耦合及光谱仪分光探测的完整物理过程。

```
[天体固有辐射 F_nu]
       │
       ▼  (银河系消光 A_lambda)
[星际衰减光谱]
       │
       ▼  (大气消光 k(lambda) + 水汽/臭氧吸收)
[地面接收光谱] ──┐
       │         │
       ▼         ▼ (夜空发射线 UVES/OH + 连续天光 + 月光散射)
[光纤几何截取 Geo(r_eff, Seeing)] ◄── [夜空背景辐射]
       │
       ▼  (望远镜集光 A_eff + 光谱仪效率 Thrput)
[CCD 光电子数 S_source, B_sky]
       │
       ▼  (泊松噪声 + 暗电流 + 读出噪声 + 天空扣除系统误差 + 杂散光)
[信噪比 SNR / 逆向曝光时间求解器]
```

### 2.1 银河系前相尘埃消光 (Galactic Extinction)
采用 Cardelli, Clayton & Mathis (CCM 1989) / Fitzpatrick (1999) 尘埃消光模型。对于给定色余 $E(B-V)$ 与目标波长 $\lambda$，消光调制系数为：
$$\eta_\text{ext}(\lambda) = 10^{-0.4 \cdot A_\lambda} = 10^{-0.4 \cdot R_V \cdot E(B-V) \cdot \left[ a(x) + b(x)/R_V \right]}$$
其中 $x = 1/\lambda\text{ }[\mu\text{m}^{-1}]$，$R_V = 3.1$。

### 2.2 大气传输与消光模型 (Atmospheric Transmission)
地面接收通量受两部分大气效应调制：
1. **连续谱消光 (Continuum Extinction)**：
   $$\eta_\text{atm, cont}(\lambda) = 10^{-0.4 \cdot k(\lambda) \cdot X}$$
   其中 $X = \sec(\text{ZA})$ 为大气质量（Air Mass），$k(\lambda)$ 包含瑞利散射（$\propto \lambda^{-4}$）、气溶胶米氏散射（Aerosol $\propto \lambda^{-1}$）与臭氧查普伊带吸收。
2. **高分辨率大气分子吸收线 (Molecular Absorption Lines)**：
   加载 `modeldata.py` 中的 Kitt Peak 高分辨率大气吸收查找表 `ATMTRANS_KP`（$500 \sim 1500\text{ nm}$，步长 $0.25\text{ \AA}$）与 Mauna Kea $3\text{mm}$ 可沉降水汽（PWV）吸收查找表 `MKTRANS_3MM`。

### 2.3 光纤几何耦合效率模型 ($\eta_\text{geo}$)
光纤几何集光效率是 ETC 最关键的非线性模块之一，取决于**天体本征形态（Point / Sérsic Profile）**、**视宁度点扩散函数（Seeing PSF）**、**像斑离焦（Spot Blur）**与**瞄准中心偏（Decenter）**：

#### A. 视宁度与波长依赖性
视宁度随波长呈现 Kolmogorov 湍流标度规律：
$$\text{FWHM}(\lambda) = \text{FWHM}_{800} \cdot \left( \frac{\lambda}{800\text{ nm}} \right)^{-0.2} \cdot X^{0.6}$$

#### B. 光纤截获积分（Hankel 变换与 Bessel 积分）
对于半光半径为 $r_\text{eff}$ 的天体面亮度分布 $I(\theta)$ 与系统总 PSF $P(\theta)$：
$$\eta_\text{geo} = \int_{\text{Fiber}} [I \otimes P](\mathbf{r}) \, d^2\mathbf{r} = 2\pi r_0 \int_0^\infty \tilde{I}(u) \tilde{P}(u) J_1(2\pi u r_0) J_0(2\pi u \Delta r) \, du$$
- $r_0$：光纤角半径 ($0.681''$)；
- $\Delta r$：光纤中心瞄准偏移误差 (`decenter`, 默认 $0.03''$)；
- $J_0, J_1$：第 0 阶与第 1 阶 Bessel 函数；
- $\tilde{I}(u)$：点源时 $\tilde{I}(u) \equiv 1$；指数圆盘星系 ($n=1$) 时 $\tilde{I}(u) = [1 + (2\pi u r_\text{scale})^2]^{-3/2}$；de Vaucouleurs 椭圆星系 ($n=4$) 采用多项式变换求解。

### 2.4 夜空背景辐射模型 (Sky Emission)
夜空背景 $B_\text{sky}$ 包含三大物理来源：
1. **夜天光连续谱 (Sky Continuum)**：在暗夜条件下，$600\text{ nm}$ 处典型表面亮度为 $V \approx 21.55\text{ mag/arcsec}^2$。
2. **高分辨率夜天光发射线 (Sky Emission Lines)**：
   - 紫外-可见波段：加载 ESO UVES 数据库的 **2,816 条**精细夜天光谱线（波长 $314 \sim 1042\text{ nm}$，强度精确至 $10^{-12}\text{ erg/s/cm}^2/\text{arcsec}^2$）；
   - 近红外波段：加载 **698 条**强高阶 OH 激发气辉发射线表 (`OHDATA`)。
3. **Krisciunas & Schaefer (1991) 物理月光散射模型**：
   根据月相系数 $f_\text{moon}$、月球天顶角 $\text{ZA}_\text{moon}$ 与目标-月球夹角 $\theta_\text{moon}$，动态叠加瑞利散射与气溶胶散射形成的漫射月光背景。

### 2.5 探测器电子计数与完整噪声模型 (CCD Noise Model)
在单次曝光时间 $t_\text{exp}$（秒）、共 $N_\text{exp}$ 次曝光下，每个波长像素上的电子数与噪声方差计算公式如下：

#### A. 目标光电子信号数 ($S$)
$$S = F_\nu \cdot \eta_\text{atm}(\lambda) \cdot \eta_\text{ext}(\lambda) \cdot \eta_\text{geo}(\lambda) \cdot \eta_\text{trace} \cdot A_\text{eff}(\lambda) \cdot t_\text{exp} \cdot \left(\frac{c}{\lambda^2}\right) \Delta\lambda_\text{pix} \cdot \frac{\lambda}{hc}$$

#### B. 噪声方差总和 ($\sigma^2$)
$$\sigma^2 = \sigma_\text{Poisson, source}^2 + \sigma_\text{Poisson, sky}^2 + \sigma_\text{sys, sky}^2 + \sigma_\text{stray}^2 + \sigma_\text{dark}^2 + \sigma_\text{readout}^2$$

- **天体与天光泊松噪声**：$\sigma_\text{Poisson, source}^2 = S$，$\sigma_\text{Poisson, sky}^2 = B_\text{sky}$；
- **天空扣除系统残差**：$\sigma_\text{sys, sky}^2 = (\text{sysfrac} \cdot B_\text{sky})^2$（默认 $\text{sysfrac} = 0.01$，即 1% 天空残余下限）；
- **漫射杂散光噪声**：$\sigma_\text{stray}^2 = \text{diffuse\_stray} \cdot \langle B_\text{sky} \rangle \cdot t_\text{exp}$（默认 2% 杂散光比例）；
- **暗电流噪声**：$\sigma_\text{dark}^2 = I_\text{dark} \cdot N_\text{pix, trace} \cdot t_\text{exp}$；
- **CCD 读出噪声**：$\sigma_\text{readout}^2 = N_\text{pix, trace} \cdot \sigma_\text{read}^2$（单像素 $\sigma_\text{read} = 4\,e^-\text{ RMS}$，谱线抽取宽度 $N_\text{pix, trace} = 7$ 像素）。

#### C. 合并信噪比 (Total Combined SNR)
$$SNR_\text{total} = \sqrt{N_\text{exp}} \cdot \frac{S}{\sqrt{S + B_\text{sky} + (\text{sysfrac} \cdot B_\text{sky})^2 + \sigma_\text{stray}^2 + I_\text{dark} N_\text{trace} t_\text{exp} + N_\text{trace} \sigma_\text{read}^2}}$$

---

## 3. 核心算法设计与数值优化

### 3.1 曝光时间逆向求解算法 (Exposure Time Solver)
当用户指定目标信噪比 $SNR_\text{target}$ 反算所需曝光时间 $t_\text{exp}$ 时，由于噪声模型中包含了常数项（读出噪声）、线性项（泊松噪声/暗电流）与平方项（天空扣除残差），$SNR(t_\text{exp})$ 是关于 $t_\text{exp}$ 的非线性严格单调递增函数。

#### 求解器算法步骤（Illinois Hybrid Secant / Bisection）：
1. **物理初始步长估计**：根据初猜时间 $t_0 = 600\text{ s}$ 计算初始信噪比 $SNR_0$，利用天光极限标度关系 $t_1 \approx t_0 \cdot (SNR_\text{target} / SNR_0)^2$ 快速收敛至真实解附近；
2. **区间括号化 (Bracketing)**：确定上下界 $[t_\text{lo}, t_\text{hi}]$ 满足 $SNR(t_\text{lo}) \le SNR_\text{target} \le SNR(t_\text{hi})$；
3. **割线迭代与安全除零保护**：
   $$\Delta s = s_\text{hi} - s_\text{lo}$$
   $$t_\text{new} = \begin{cases} 0.5(t_\text{lo} + t_\text{hi}), & \text{if } |\Delta s| < 10^{-12} \\ t_\text{lo} + (SNR_\text{target} - s_\text{lo}) \frac{t_\text{hi} - t_\text{lo}}{\Delta s}, & \text{otherwise} \end{cases}$$
4. **收敛判据**：当 $|SNR(t_\text{new}) - SNR_\text{target}| / SNR_\text{target} < 0.01$（相对误差 $< 1\%$）时停止，通常在 **3 ~ 5 步** 内高精度收敛。

### 3.2 引擎性能与 Numba JIT 加速 (`ETC_py_optimized.py`)
为了在几毫秒内完成全谱段计算，底层的光线追踪循环、Bessel 函数数值积分及多波长大气表格插值均使用了 `@jit(nopython=True, fastmath=True)` 编译加速，消除了 Python 解释器的循环开销。

---

## 4. 参数配置速查表与掩码定义

### 4.1 观测条件字典 (`_DEFAULTS`)
可以通过 `etc.set_obs_conditions(**kwargs)` 进行配置：

| 参数键值 | 类型 | 默认值 | 物理含义与推荐取值 |
| :--- | :--- | :--- | :--- |
| `seeing_fwhm_800` | float | `0.8` | $800\text{ nm}$ 处大气视宁度 FWHM（角秒，优秀条件 $0.6''$，中等 $0.8''\sim 1.0''$，较差 $1.2''\sim 1.5''$） |
| `zenith_angle` | float | `45.0` | 天体观测天顶角（度，天顶为 $0^\circ$，典型观测为 $30^\circ\sim 45^\circ$） |
| `ebv` | float | `0.03` | 银河系前相色余 $E(B-V)$（高银纬区通常 $\approx 0.02\sim 0.05$） |
| `field_angle` | float | `0.675` | 焦平面天体位置角距离（度，视场中心为 $0^\circ$，视场边缘为 $0.6^\circ$） |
| `decenter` | float | `0.03` | 光纤中心对准误差（角秒，典型巡天为 $0.03''\sim 0.1''$） |
| `lunar_za` | float | `135.0` | 月球天顶角（$>90^\circ$ 表示月亮在地平线以下，即暗夜 Dark Time） |
| `lunar_angle` | float | `90.0` | 月球与目标天体的角距离（度） |
| `lunar_phase` | float | `0.25` | 月相（$0.0$ 为新月，$0.5$ 为满月，$1.0$ 为新月） |
| `sysfrac` | float | `0.01` | 天空背景扣除系统误差下限（1% rms） |
| `diffuse_stray` | float | `0.02` | 探测器杂散光比例（2%） |
| `r_eff` | float | `0.0` | 天体半光半径（角秒，点源/恒星设为 `0.0`，扩展星系设为实际尺寸） |
| `skytype_hex` | str | `'10003'` | 5 位 16 进制天空模型控制掩码（见下表） |

### 4.2 16 进制天空模型掩码编码 (`skytype_hex`)
```text
skytype = 0x L  A  C  M  S
             │  │  │  │  └── S: 天光连续谱模型 (0:关, 3:标准暗夜连续谱 21.55 mag/arcsec²)
             │  │  │  └───── M: 月光散射模型 (0:开启 Krisciunas & Schaefer 物理月光模型)
             │  │  └──────── C: 大气连续消光模型 (0:Mauna Kea/KPNO 分段消光曲线)
             │  └─────────── A: 大气线吸收模型 (0:Kitt Peak 高分辨率透射表, 1:附加 MK 3mm 水汽带)
             └────────────── L: 天光发射线模型 (0:关, 1:开启 UVES 2816条 + OH 698条夜天光谱线)
```

---

## 5. 与其它光谱流水线（`justspecsimu` / `PFS ETC`）的对比

| 参数 / 模型维度 | `justspecsimu` (DESI-based) | `JUST ETC` (PFS-based Python Vectorized) | 物理与实现差异说明 |
| :--- | :--- | :--- | :--- |
| **几何注入效率 ($\eta_\text{geo}$)** | 查表插值 (`DESI-0347_blur.ecsv` + `offset.ecsv`) | **解析卷积积分** (Hankel/Bessel 变换) | ETC 支持任意连续的 $r_\text{eff}$ 与 Seeing 实时积分，无网格插值截断误差。 |
| **夜天光建模** | `spec-sky.dat` 固定连续谱 | **UVES 2816 条线 + OH 气辉 + 月光散射** | ETC 在强 OH 发射线区及月光条件下的信噪比预测更为精细。 |
| **焦平面比例尺** | $128.473\text{ }\mu\text{m/arcsec}$ | $128.473\text{ }\mu\text{m/arcsec}$ ($EFL=26.5\text{ m}$) | 两者硬件参数完全一致对齐。 |
| **消光曲线** | KPNO 宽带消光表 | KPNO 高分辨率透射表 + CCM 尘埃模型 | 波长采样率更高（$0.25\text{ \AA}$），能分辨精细吸收带。 |
| **计算性能** | 基于 Python 静态流水线 | **Numba JIT 向量化 + 多进程并行** | 单谱计算时间 $< 2\text{ ms}$，支持万条光谱并行仿真。 |
