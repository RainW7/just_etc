# 参数对比数据说明

本目录为从 `py/justspec/data/datamodel/` 复制的参考副本，用于参数对比与查阅。
**请勿在模拟流程中直接改路径指向此处**；正式输入仍以 `datamodel` 下原文件为准。

来源根目录：`py/justspec/data/datamodel/`

---

## 文件一览

| 文件 | 格式 | 在模拟中的用途 |
|------|------|----------------|
| `spec-sky.dat` | ASCII 两列 | 暗夜天空面亮度谱（`atmosphere.sky`，condition=`dark`） |
| `ZenithExtinction-KPNO.dat` | ASCII 两列 | KPNO 天顶消光系数（`atmosphere.extinction`） |
| `DESI-0347_blur.ecsv` | Astropy ECSV | 光学 RMS blur（场角 × 波长），用于光纤损失 / PSF |
| `DESI-0347_offset.ecsv` | Astropy ECSV | 光学径向质心偏移（场角 × 波长） |
| `thru-b_y1measured.fits` | FITS 表 | b 相机 Y1 实测 throughput |
| `thru-r_y1measured.fits` | FITS 表 | r 相机 Y1 实测 throughput |
| `thru-z_y1measured.fits` | FITS 表 | z 相机 Y1 实测 throughput |

---

## 1. `spec-sky.dat` — 暗夜天空谱

**来源路径：** `spectra/spec-sky.dat`  
**配置引用：** `atmosphere.sky.table.paths.dark`

### 格式

- 纯文本 ASCII，`#` 开头为注释
- 两列空白分隔，约 65000 行数据
- 波长范围：3500.0 – 9999.9 Angstrom，步长 0.1 Angstrom

| 列 | 名称（文件头） | 单位（config） | 含义 |
|----|----------------|----------------|------|
| 0 | WAVELENGTH | Angstrom | 波长 |
| 1 | FLUX | `1e-17 erg / (Angstrom arcsec2 cm2 s)` | 天空面亮度 |

示例：

```
#   WAVELENGTH        FLUX
#------------- -----------
     3500.000      1.6266
```

### 用途

提供 DESI QuickSim 的 **dark** 天空发射谱。模拟时与 airmass、是否对天空做消光（`extinct_emission`）等一起，生成进入光纤的天空背景光子。同系列还有 `spec-sky-grey.dat`、`spec-sky-bright.dat`（本目录未复制）。

---

## 2. `ZenithExtinction-KPNO.dat` — 天顶消光

**来源路径：** `spectra/ZenithExtinction-KPNO.dat`  
**配置引用：** `atmosphere.extinction.table.path`

### 格式

- 纯文本 ASCII，`#` 开头为注释
- 两列空白分隔，约 65000 行数据
- 波长范围：3500.0 – 9999.9 Angstrom，步长 0.1 Angstrom

| 列 | 名称（文件头） | 单位 | 含义 |
|----|----------------|------|------|
| 0 | WAVELENGTH | Angstrom | 波长 |
| 1 | EXTINCTION | mag / airmass（天顶） | 消光系数 k(λ) |

示例：

```
#  WAVELENGTH    EXTINCTION
#------------ -------------
     3500.000     0.6000000
```

### 用途

KPNO 站点天顶消光曲线。透射近似为 `10^(-0.4 * k(λ) * airmass)`，用于源谱（及可选的天空发射）的大气衰减。

---

## 3. `DESI-0347_blur.ecsv` — 光学 RMS blur

**来源路径：** `throughput/DESI-0347_blur.ecsv`  
**配置引用：** `instrument.blur`  
**出处：** DESI-0347（光线追迹 + 色差/非色差合成），详见文档中 `DESI-0347_Throughput` 相关说明。

### 格式

- Astropy **ECSV 0.9**（YAML 头 + 空格分隔表）
- 第 1 列：`wavelength` [Angstrom]
- 其余列：不同场角下的 RMS spot size [micron]
- 场角：`r = 0.00, 0.15, …, 1.50, 1.60` deg（共 12 个）
- 波长采样：约 3550 – 9850 Angstrom（稀疏网格，约 15 点）

### 用途

描述理想光学下焦面光斑 RMS 大小随 **波长与场角** 的变化。在启用 blur 的配置中，与 seeing、光纤直径等一起进入光纤损失 / 有效 PSF 计算。

---

## 4. `DESI-0347_offset.ecsv` — 光学径向质心偏移

**来源路径：** `throughput/DESI-0347_offset.ecsv`  
**配置引用：** `instrument.offset`  
**出处：** DESI-0347 光线追迹（理想光学径向偏移；不含表中随机非色差项）。

### 格式

- 与 blur 相同的 ECSV 布局
- 第 1 列：`wavelength` [Angstrom]
- 其余列：各场角径向质心偏移 [micron]（可正可负）
- 场角与波长网格与 `DESI-0347_blur.ecsv` 对齐

### 用途

描述光斑质心相对光纤中心的径向偏移。与 `blur`、可选的 `static` 静态偏移图、`sigma1d` 径向抖动等组合，用于更真实的光纤对准 / 损失模型。

---

## 5. `thru-{b,r,z}_y1measured.fits` — Y1 实测吞吐量

**来源路径：** `throughput/thru-b_y1measured.fits` 等  
**配置用法：** 可替换默认的 `thru-b.fits` / `thru-r.fits` / `thru-z.fits`  
（默认 config 使用无设计吞吐；本文件为 **Year-1 实测** 版本，便于对比。）

### 格式

- FITS，主 HDU 为空；扩展 HDU：`EXTNAME = THROUGHPUT`（`BINTABLE`）
- 行数：63001
- 波长：约 3550.05 – 9850.05 Angstrom，步长 0.1 Angstrom
- 列（均为 float64）：

| 列名 | 单位 | 含义 |
|------|------|------|
| `wavelength` | Angstrom | 波长 |
| `throughput` | 无量纲 (0–1) | 端到端吞吐（设计/测量合成，含仪器链路） |
| `extinction` | — | 表内附带的消光相关列（与大气独立消光表并存时需注意是否重复使用） |
| `fiberinput` | 无量纲 (0–1) | 光纤入射效率相关因子 |

各相机吞吐峰值（本副本统计）：

| 文件 | throughput max |
|------|-----------------|
| `thru-b_y1measured.fits` | ~0.36 |
| `thru-r_y1measured.fits` | ~0.42 |
| `thru-z_y1measured.fits` | ~0.48 |

与默认 `thru-*.fits` 的差异简述：默认表行数更少、带 `WAVEMIN`/`WAVEMAX` 等头关键字；Y1 measured 为更密波长网格的实测曲线，三相机文件波长轴相同（全波段表，带外吞吐为 0）。

### 用途

将光子从望远镜到相机 CCD 的效率按波长计入模拟，决定每根光纤、每个相机的电子计数与噪声预算。做 **设计吞吐 vs Y1 实测** 对比时，改 config 中 `cameras.*.throughput.table.path` 指向对应文件即可。

---

## 读取示例

```python
from astropy.table import Table
from astropy.io import ascii

sky = ascii.read("spec-sky.dat", names=["wavelength", "flux"])
ext = ascii.read("ZenithExtinction-KPNO.dat", names=["wavelength", "extinction"])
blur = Table.read("DESI-0347_blur.ecsv")
offset = Table.read("DESI-0347_offset.ecsv")
thru_b = Table.read("thru-b_y1measured.fits", hdu="THROUGHPUT")
```

---

## 复制信息

- 复制日期：2026-07-23
- 操作：从 `datamodel` 原路径 `cp` 到本目录（内容一致，非软链接）
