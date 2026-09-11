# 中文入门

本仓库将原 `ETC_py_v1` 整理为可安装的 `just_etc` 包。已有的物理计算引擎、
模型数据和仪器参数沿用原版本；主要变化是目录结构、导入方式、资源定位和文档。

## 安装和运行

在用于科研计算的环境中执行：

```sh
git clone https://github.com/RainW7/just_etc.git
cd just_etc
python -m pip install .
just-etc --help
just-etc --mag 21.5 --texp 900 --nexp 4 --target point
```

`--texp` 是单次曝光时间，以上共 3600 s。输出是每像元信噪比。
`--target point` 只指定点源空间轮廓，默认 SED 仍是星暴星系模板；模拟恒星时应
通过 `--template` 提供相应恒星光谱。首次运行需要 Numba 编译，耗时较长。

Python 接口改为：

```python
from just_etc import JUSTExposureTimeCalculator, load_template, normalize_to_mag

wave, flux = load_template("galaxy/elliptical_001.fits")
flux, scale = normalize_to_mag(wave, flux, target_mag=21.5, band="r")
etc = JUSTExposureTimeCalculator(calc_mode="fast")
result = etc.compute_snr(wave, flux, t_exp=900, n_exp=4)
```

模板和默认 `spec.dat` 随安装包提供，不必切换回源码目录。波长输入为 Å，
流量密度为 erg/s/cm²/Å。曝光求解器的参考波长使用 nm。星等归一化采用近似
波段窗口，不等同于真实滤光片响应曲线下的合成测光。

## 开发和验证

开发安装：`python -m pip install -e '.[test]'`。
测试：`python -m pytest py/just_etc/test`。
科研示例位于 `examples/`，通常将结果写入运行时当前目录下的 `output/`。
大型外部星系库和历史模拟输出不随仓库发布。

原有 Python 3.6 科研环境保留兼容支持；新建环境建议使用当前 Python。
不要为了安装而无差别升级现有科学计算依赖。更多内容见 [使用说明](usage.md)
和 [Git 维护说明](development.md)。
