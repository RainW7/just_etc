"""
PFS Exposure Time Calculator - Python Version (Optimized)
Converted from C code (gsetc.c)
VERSION 5 - With NumPy vectorization and Numba JIT optimization
"""

import numpy as np
import math
import sys
import time
from numba import jit, prange
from functools import lru_cache

try:
    from . import modeldata as _modeldata  # type: ignore[import]
except ImportError:  # pragma: no cover - script-style execution
    try:
        import modeldata as _modeldata  # type: ignore[import]
    except ImportError:
        _modeldata = None


MODEL_DATA_AVAILABLE = _modeldata is not None

if MODEL_DATA_AVAILABLE:
    MODEL_GS_SKY_UVES_NLINES = int(_modeldata.GS_SKY_UVES_NLINES)
    MODEL_GS_SKY_UVES_LAMBDA = _modeldata.GS_SKY_UVES_LAMBDA
    MODEL_GS_SKY_UVES_INT = _modeldata.GS_SKY_UVES_INT
    MODEL_N_IR_OH_LINE = int(_modeldata.N_IR_OH_LINE)
    MODEL_OHDATA = _modeldata.OHDATA
    MODEL_ATMTRANS_KP = _modeldata.ATMTRANS_KP
    MODEL_ATMTRANS_KP_LAMBDA_MIN = getattr(_modeldata, "ATMTRANS_KP_LAMBDA_MIN", None)
    MODEL_ATMTRANS_KP_LAMBDA_STEP = getattr(_modeldata, "ATMTRANS_KP_LAMBDA_STEP", None)
    MODEL_MKTRANS_3MM = _modeldata.MKTRANS_3MM
    MODEL_MKTRANS_3MM_LAMBDA_MIN = getattr(_modeldata, "MKTRANS_3MM_LAMBDA_MIN", None)
    MODEL_MKTRANS_3MM_LAMBDA_STEP = getattr(_modeldata, "MKTRANS_3MM_LAMBDA_STEP", None)
else:
    MODEL_GS_SKY_UVES_NLINES = None
    MODEL_GS_SKY_UVES_LAMBDA = None
    MODEL_GS_SKY_UVES_INT = None
    MODEL_N_IR_OH_LINE = None
    MODEL_OHDATA = None
    MODEL_ATMTRANS_KP = None
    MODEL_ATMTRANS_KP_LAMBDA_MIN = None
    MODEL_ATMTRANS_KP_LAMBDA_STEP = None
    MODEL_MKTRANS_3MM = None
    MODEL_MKTRANS_3MM_LAMBDA_MIN = None
    MODEL_MKTRANS_3MM_LAMBDA_STEP = None

# Constants
ARCSEC_PER_URAD = 0.206264806247097
DEGREE = 0.017453292519943278
PHOTONS_PER_ERG_1NM = 5.03411747e8
RAT_HL_SL_EXP = 1.67834

# Ratio of half-light to scale radius for exponential profile galaxy
RAT_HL_SL_EXP = 1.67834

# Global Calculation Configuration (Default: Balanced)
CALC_CONFIG = {
    'geo_nu': 50,
    'geo_du': 0.024,
    'atm_step': 0.1,
    'cont_samp': 5,
    'snr_stride': 1
}

def set_calculation_mode(mode_name):
    """
    Set calculation precision/speed mode.
    mode_name: 'fast', 'balanced', 'accurate'
    """
    global CALC_CONFIG
    if mode_name == 'fast':
        CALC_CONFIG.update({'geo_nu': 20, 'geo_du': 0.06, 'atm_step': 0.5, 'cont_samp': 2, 'snr_stride': 10})
    elif mode_name == 'accurate':
        CALC_CONFIG.update({'geo_nu': 100, 'geo_du': 0.012, 'atm_step': 0.05, 'cont_samp': 10, 'snr_stride': 1})
    else: # balanced
        # Moderate mode: Stride 2 for 2x speedup with minimal loss? Or stay 1?
        # User said Moderate was 360s. Let's make Moderate stride=2 to help it too, or keep 1.
        # Let's keep Moderate strict for now, optimize Fast.
        CALC_CONFIG.update({'geo_nu': 50, 'geo_du': 0.024, 'atm_step': 0.1, 'cont_samp': 5, 'snr_stride': 1})
    print(f"ETC Calculation Mode: {mode_name.upper()} {CALC_CONFIG}")

# [OII] recovery histogram data
ZMIN_OII = 0.10
NZ_OII = 24
DZ_OII = 0.10

# Maximum legal number of arms
MAXARM = 5

# Maximum number of l-pixels per spectrograph arm
MAXPIX = 8192

# Maximum length of throughput table
MAXNTHR = 1024

# Spectrograph PSF length (must be even)
SP_PSF_LEN = 32

# Global storage for optional magnitude input tables
lambda_inmag = []
mag_inmag = []
lambda_inmag2 = []
mag_inmag2 = []
num_inmag = 0
num_inmag2 = 0


class SpectroAttrib:
    """Spectrograph attributes structure"""
    def __init__(self):
        self.D_outer = 0.0
        self.EFL = np.zeros(5, dtype=np.float64)
        self.fiber_ent_rad = 0.0
        self.centobs = 0.0
        self.rfov = 0.0
        self.rms_spot = np.zeros(5, dtype=np.float64)
        self.vignette = np.ones(5, dtype=np.float64)
        self.N_arms = 0
        self.MR = 0
        
        self.lmin = np.zeros(MAXARM, dtype=np.float64)
        self.lmax = np.zeros(MAXARM, dtype=np.float64)
        self.npix = np.zeros(MAXARM, dtype=np.int32)
        self.dl = np.zeros(MAXARM, dtype=np.float64)
        self.width = np.zeros(MAXARM, dtype=np.int32)
        self.fratio = np.zeros(MAXARM, dtype=np.float64)
        self.thick = np.zeros(MAXARM, dtype=np.float64)
        self.pix = np.zeros(MAXARM, dtype=np.float64)
        self.temperature = np.zeros(MAXARM, dtype=np.float64)
        self.rms_cam = np.zeros(MAXARM, dtype=np.float64)
        self.diam = np.zeros(MAXARM, dtype=np.float64)
        self.dark = np.zeros(MAXARM, dtype=np.float64)
        self.read = np.zeros(MAXARM, dtype=np.float64)
        self.sep = np.zeros(MAXARM, dtype=np.float64)
        self.nline = np.full(MAXARM, 1e12, dtype=np.float64)
        
        self.N_thr = 0
        self.istart = np.zeros(MAXARM + 1, dtype=np.int32)
        self.l = np.zeros(MAXNTHR, dtype=np.float64)
        self.T = np.zeros(MAXNTHR, dtype=np.float64)
        
        self.Dtype = np.zeros(MAXARM, dtype=np.int32)
        
        self.sysfrac = 0.0
        self.diffuse_stray = 0.0


class ObsAttrib:
    """Observing condition attributes structure"""
    def __init__(self):
        self.seeing_fwhm_800 = 0.0
        self.elevation = 0.0
        self.zenithangle = 0.0
        self.lunarphase = 0.0
        self.lunarangle = 0.0
        self.lunarZA = 0.0
        self.EBV = 0.0
        self.skytype = 0


def spectro_arm(spectro, ia):
    if getattr(spectro, "MR", 0):
        return 3 if ia == 1 else ia
    return ia


# ==================== OPTIMIZED SPECIAL FUNCTIONS ====================

# 预计算 Bessel 函数查找表
_LOOKUP_SIZE = 10000
_LOOKUP_MAX = 30.0
_J0_LOOKUP = np.zeros(_LOOKUP_SIZE)
_J1_LOOKUP = np.zeros(_LOOKUP_SIZE)

# _init_bessel_lookup removed - initialization happens after helper definitions

def _getJ0_original(x):
    """原始 J0 实现（用于构建查找表）"""
    x = abs(x)
    if x > 3:
        u = 3.0 / x
        f = (0.79788456 - 0.00000077*u - 0.00552740*u*u - 0.00009512*u**3 +
             u**4 * (0.00137234 - 0.00072805*u + 0.00014476*u*u))
        t = (x - 0.78539816 - 0.04166397*u - 0.00003954*u*u + 0.00262573*u**3 +
             u**4 * (-0.00054125 - 0.00029333*u + 0.00013558*u*u))
        return f / math.sqrt(x) * math.cos(t)
    
    f = 0.0
    for i in range(50):
        f += math.cos(x * math.cos((i + 0.5) * math.pi / 50.0))
    return 0.02 * f

def _getJ1_original(x):
    """原始 J1 实现（用于构建查找表）"""
    s = 1
    if x < 0:
        x = -x
        s = -1
    
    if x > 3:
        u = 3.0 / x
        f = (0.79788456 + 0.00000156*u + 0.01659667*u*u + 0.00017105*u**3 +
             u**4 * (-0.00249511 + 0.00113653*u - 0.00020033*u*u))
        t = (x - 2.35619449 + 0.12499612*u + 0.0000565*u*u - 0.00637879*u**3 +
             u**4 * (0.00074348 + 0.00079824*u - 0.00029166*u*u))
        return s * f / math.sqrt(x) * math.cos(t)
    
    f = 0.0
    for i in range(50):
        angle = (i + 0.5) * math.pi / 50.0
        f += math.cos(x * math.sin(angle) - angle)
    return 0.02 * s * f

# Initialize lookup tables immediately
for i in range(_LOOKUP_SIZE):
    x = i * _LOOKUP_MAX / (_LOOKUP_SIZE - 1)
    _J0_LOOKUP[i] = _getJ0_original(x)
    _J1_LOOKUP[i] = _getJ1_original(x)

@jit(nopython=True, cache=True)
def getJ0_fast(x):
    """使用查找表的快速 J0"""
    x_abs = abs(x)
    if x_abs > _LOOKUP_MAX:
        # 大值使用渐近展开
        u = 3.0 / x_abs
        f = (0.79788456 - 0.00000077*u - 0.00552740*u*u - 0.00009512*u**3 +
             u**4 * (0.00137234 - 0.00072805*u + 0.00014476*u*u))
        t = (x_abs - 0.78539816 - 0.04166397*u - 0.00003954*u*u + 0.00262573*u**3 +
             u**4 * (-0.00054125 - 0.00029333*u + 0.00013558*u*u))
        return f / np.sqrt(x_abs) * np.cos(t)
    
    # 使用线性插值查找表
    idx_f = x_abs * (_LOOKUP_SIZE - 1) / _LOOKUP_MAX
    idx = int(idx_f)
    if idx >= _LOOKUP_SIZE - 1:
        idx = _LOOKUP_SIZE - 2
    frac = idx_f - idx
    return _J0_LOOKUP[idx] * (1 - frac) + _J0_LOOKUP[idx + 1] * frac

@jit(nopython=True, cache=True)
def getJ1_fast(x):
    """使用查找表的快速 J1"""
    s = 1.0 if x >= 0 else -1.0
    x_abs = abs(x)
    
    if x_abs > _LOOKUP_MAX:
        u = 3.0 / x_abs
        f = (0.79788456 + 0.00000156*u + 0.01659667*u*u + 0.00017105*u**3 +
             u**4 * (-0.00249511 + 0.00113653*u - 0.00020033*u*u))
        t = (x_abs - 2.35619449 + 0.12499612*u + 0.0000565*u*u - 0.00637879*u**3 +
             u**4 * (0.00074348 + 0.00079824*u - 0.00029166*u*u))
        return s * f / np.sqrt(x_abs) * np.cos(t)
    
    idx_f = x_abs * (_LOOKUP_SIZE - 1) / _LOOKUP_MAX
    idx = int(idx_f)
    if idx >= _LOOKUP_SIZE - 1:
        idx = _LOOKUP_SIZE - 2
    frac = idx_f - idx
    return s * (_J1_LOOKUP[idx] * (1 - frac) + _J1_LOOKUP[idx + 1] * frac)

# 兼容接口
def getJ0(x):
    if _J0_LOOKUP is None:
        _init_bessel_lookup()
    return getJ0_fast(x)

def getJ1(x):
    if _J1_LOOKUP is None:
        _init_bessel_lookup()
    return getJ1_fast(x)

@jit(nopython=True, cache=True)
def geterf_fast(x):
    """优化的误差函数"""
    s = 1.0 if x >= 0 else -1.0
    x = abs(x)
    
    if x > 6:
        return s
    
    if x < 1.5:
        erfabsx = 0.0
        term = 2.0 / np.sqrt(np.pi) * x
        for j in range(25):
            erfabsx += term / (2*j + 1)
            term *= -x * x / (j + 1)
        return s * erfabsx
    
    erfcx = 0.0
    dy = 0.01
    for j in range(-3000, 300):
        y = j * dy
        u = x + np.exp(y) / x
        erfcx += np.exp(y - u*u)
    erfcx *= 2.0 / np.sqrt(np.pi) / x * dy
    erfabsx = 1 - erfcx
    
    return s * erfabsx

def geterf(x):
    return geterf_fast(x)


# ==================== CONVERSION FUNCTIONS ====================

@jit(nopython=True, cache=True)
def gs_n_air_fast(lambda_vac):
    """Index of refraction of air - Edlen 1953 formula"""
    return (1.000064328 + 0.0294981/(146 - 1e6/lambda_vac/lambda_vac) +
            0.0002554/(41 - 1e6/lambda_vac/lambda_vac))

def gs_n_air(lambda_vac):
    return gs_n_air_fast(lambda_vac)

@jit(nopython=True, cache=True)
def gs_vac2air(lambda_vac):
    """Vacuum to air conversion"""
    return lambda_vac / gs_n_air_fast(lambda_vac)

@jit(nopython=True, cache=True)
def gs_air2vac(lambda_air):
    """Air to vacuum conversion"""
    lambda_vac = lambda_air
    for i in range(10):
        lambda_vac += lambda_air - lambda_vac / gs_n_air_fast(lambda_vac)
    return lambda_vac


# ==================== MATERIAL PROPERTIES ====================

# Silicon 数据表预处理
_SI_TABLE = np.array([
    0.9317, 1.0031, 1.0811, 1.1490, 1.2253, 1.3295, 1.4848, 1.5861, 1.5833, 1.5690,
    1.5800, 1.6170, 1.6834, 1.7971, 2.0176, 2.4043, 2.9198, 3.5317, 4.3658, 4.8761,
    5.0036, 5.0203, 5.0100, 5.0096, 5.0234, 5.0551, 5.0983, 5.1548, 5.2220, 5.3086,
    5.4358, 5.6569, 6.0519, 6.5467, 6.8110, 6.7364, 6.4693, 6.1850, 5.9438, 5.7404,
    5.5700, 5.4254, 5.2960, 5.1868, 5.0890, 5.0024, 4.9235, 4.8522, 4.7873, 4.7289,
    4.6738, 4.6230, 4.5759, 4.5317, 4.4918, 4.4543, 4.4202, 4.3866, 4.3553, 4.3253,
    4.2975, 4.2716, 4.2460, 4.2224, 4.2000, 4.1787, 4.1586, 4.1378, 4.1197, 4.1017,
    4.0844, 4.0682, 4.0526, 4.0376, 4.0226, 4.0092, 3.9954, 3.9825, 3.9700, 3.9585,
    3.9473, 3.9367, 3.9262, 3.9156, 3.9058, 3.8955, 3.8865, 3.8776, 3.8684, 3.8594,
    3.8512, 3.8435, 3.8362, 3.8285, 3.8208, 3.8134, 3.8066, 3.8005, 3.7946, 3.7888,
    3.7831, 3.7774, 3.7712, 3.7660, 3.7617, 3.7566, 3.7514, 3.7474, 3.7430, 3.7379,
    3.7333, 3.7289, 3.7250, 3.7212, 3.7176, 3.7139, 3.7093, 3.7048, 3.7008, 3.6968,
    3.6925, 3.6881, 3.6848, 3.6815, 3.6778, 3.6742, 3.6719, 3.6703, 3.6686, 3.6670,
    3.6654, 3.6637, 3.6621, 3.6605, 3.6589, 3.6572, 3.6556, 3.6540, 3.6523, 3.6503,
    3.6482, 3.6460, 3.6439, 3.6418, 3.6397, 3.6375, 3.6354, 3.6333, 3.6311, 3.6290,
    3.6269, 3.6248, 3.6233, 3.6225, 3.6217, 3.6203, 3.6183, 3.6163, 3.6144, 3.6125,
    3.6106, 3.6091, 3.6076, 3.6061, 3.6046, 3.6031, 3.6015, 3.6000, 3.5984, 3.5968,
    3.5953, 3.5937, 3.5922, 3.5906, 3.5890, 3.5875, 3.5859, 3.5844, 3.5828, 3.5812,
    3.5797
], dtype=np.float64)

@jit(nopython=True, cache=True)
def gsOP_Si_abslength_fast(lambda_nm, T):
    """优化的硅吸收长度计算"""
    beta = 7.021e-4
    k = 8.617e-5
    gamma = 1108.0
    E_g0_0, E_g0_1 = 1.1557, 2.5
    E_gd0 = 3.2
    E_p_0, E_p_1 = 1.827e-2, 5.773e-2
    C_0, C_1 = 5.5, 4.0
    A_0, A_1 = 323.1, 7237.0
    A_d = 1.052e6
    
    lambda_m = lambda_nm / 1e9
    
    if lambda_m < 2e-7 or lambda_m > 1.1e-6:
        return 0.0  # Error case
    
    hnu = 1.239842e-6 / lambda_m
    E_gd = E_gd0 - beta * T * T / (T + gamma)
    E_g_0 = E_g0_0 - beta * T * T / (T + gamma)
    E_g_1 = E_g0_1 - beta * T * T / (T + gamma)
    
    alpha = 0.0
    if hnu > E_gd:
        alpha += A_d * np.sqrt(hnu - E_gd)
    
    # Unroll loops for better performance
    for E_g, A in [(E_g_0, A_0), (E_g_1, A_1)]:
        for E_p, C in [(E_p_0, C_0), (E_p_1, C_1)]:
            de = hnu - E_g + E_p
            if de > 0:
                alpha += C * A * de * de / (np.exp(E_p/k/T) - 1)
            de = hnu - E_g - E_p
            if de > 0:
                alpha += C * A * de * de / (1 - np.exp(-E_p/k/T))
    
    return 1e4 / alpha if alpha > 0 else 1e10

def gsOP_Si_abslength(lambda_nm, T):
    result = gsOP_Si_abslength_fast(lambda_nm, T)
    if result == 0.0:
        print(f"Error: Si data out of range")
        sys.exit(1)
    return result

@jit(nopython=True, cache=True)
def gsOP_Si_indexreal_fast(lambda_nm, si_table):
    """优化的硅折射率计算"""
    lambda_m = lambda_nm / 1e9
    
    if lambda_m < 2e-7 or lambda_m > 1.1e-6:
        return 0.0
    
    x = (lambda_m - 2e-7) / 5e-9
    xint = int(np.floor(x))
    if xint <= 0:
        xint = 1
    if xint >= 179:
        xint = 178
    xfrac = x - xint
    
    return (-xfrac*(xfrac-1)*(xfrac-2)/6.0 * si_table[xint-1] +
            (xfrac*xfrac-1)*(xfrac-2)/2.0 * si_table[xint] +
            -xfrac*(xfrac+1)*(xfrac-2)/2.0 * si_table[xint+1] +
            xfrac*(xfrac*xfrac-1)/6.0 * si_table[xint+2])

def gsOP_Si_indexreal(lambda_nm, T):
    result = gsOP_Si_indexreal_fast(lambda_nm, _SI_TABLE)
    if result == 0.0:
        print("Error: Si data out of range")
        sys.exit(1)
    return result


# ==================== FOREGROUND ABSORPTION ====================

# 预计算消光表
_EXTINCTION_TABLE = np.array([
    0.24174, 0.25504, 0.26708, 0.26392, 0.23976, 0.20540, 0.17650, 0.15484, 0.13199, 0.10599,
    0.08604, 0.07340, 0.06302, 0.05489, 0.05107, 0.05005, 0.05041, 0.05151, 0.05313, 0.05554,
    0.06009, 0.06392, 0.06130, 0.06117, 0.06239, 0.06417, 0.06629, 0.06860, 0.07123, 0.07400,
    0.07702, 0.08018, 0.08361, 0.08723, 0.09105, 0.09506, 0.09934, 0.10388, 0.10862, 0.11356,
    0.11876, 0.12416, 0.12989, 0.13581, 0.14207, 0.14858, 0.15543, 0.16261, 0.17156, 0.17781,
    0.18571, 0.19401, 0.20276, 0.21192, 0.22140, 0.23127, 0.24147, 0.25214, 0.26313, 0.27459,
    0.28650, 0.29875, 0.31152, 0.32469, 0.33831, 0.35240, 0.36708, 0.38216, 0.39776, 0.41389,
    0.43055, 0.44779, 0.46557, 0.48394, 0.50296, 0.52265, 0.54292, 0.56386, 0.58552, 0.60790,
    0.63094, 0.65477, 0.67939, 0.70441, 0.73074, 0.75839, 0.78670, 0.81567, 0.84529, 0.87689,
    0.90915, 0.94273, 0.97762, 1.01382, 1.05201, 1.09151, 1.13232, 1.17512, 1.21922, 1.26531,
    1.31336, 1.36340, 1.41475, 1.46741, 1.52205, 1.57867, 1.63594, 1.69585, 1.75708, 1.82028,
    1.88545, 1.95260, 2.02107, 2.09217, 2.16524, 2.24029, 2.31731, 2.39631, 2.47795, 2.56155,
    2.64714, 2.73469, 2.82423, 2.91573, 3.00987, 3.10533, 3.20342, 3.30349, 3.40487, 3.50823,
    3.61290, 3.71955, 3.82752, 3.93680, 4.04608, 4.15668, 4.26728, 4.37854, 4.49111, 4.60369,
    4.71692, 4.83081, 4.94470, 5.05991, 5.17643, 5.29361, 5.41145, 5.53061, 5.65109, 5.77354,
    5.89862, 6.02567, 6.15339, 6.26596, 6.38907, 6.52732, 6.68203, 6.87294, 7.09019, 7.33377,
    7.64319, 7.96577, 8.45293, 8.95326, 9.47334, 9.94075, 10.22383, 10.19750, 9.92100, 9.52600,
    9.11126, 8.75576, 8.47268, 8.26201, 8.11718, 8.01843, 7.96577, 7.95260, 7.96577, 8.02502,
    8.11718, 8.24226, 8.39368, 8.55826, 8.74259, 8.92034, 9.10467, 9.31534, 9.54575, 9.82883,
    10.11192, 10.41475, 10.72416, 11.05332, 11.40224, 11.77749, 12.17248, 12.61356, 13.06781, 13.52205,
    13.92363
], dtype=np.float64)

@jit(nopython=True, cache=True)
def gsGalactic_Alambda__EBV_fast(lambda_nm, extinction_table):
    """优化的银河系消光计算"""
    lambda_um = lambda_nm / 1e3
    
    if lambda_um < 0.1 or lambda_um > 10:
        return 0.0
    
    lset = 100.0 / np.log(10.0) * np.log(10.0 / lambda_um)
    il = int(np.floor(lset))
    if il < 0:
        il = 0
    if il > 199:
        il = 199
    lf = lset - il
    
    return extinction_table[il] + lf * (extinction_table[il+1] - extinction_table[il])

def gsGalactic_Alambda__EBV(lambda_nm):
    result = gsGalactic_Alambda__EBV_fast(lambda_nm, _EXTINCTION_TABLE)
    if result == 0.0 and (lambda_nm / 1e3 < 0.1 or lambda_nm / 1e3 > 10):
        print(f"Error: Wavelength lambda = {lambda_nm/1e3:.5e} microns out of range")
        sys.exit(1)
    return result

def gsAtmContOp(obs, lambda_nm, flags):
    """
    Continuum atmospheric opacity in magnitudes per airmass
    lambda in nm
    """
    k = 0.113; # V band opacity -- placeholder !!! 
    # Median extinction on Mauna Kea
    opacity_model = (obs.skytype >> 8) & 0xf
    
    if opacity_model == 0x0:
        k = 0.075
        if lambda_nm < 900:
            k = 0.097 - 0.022 * (lambda_nm - 800.0) / 100.0
        if lambda_nm < 800:
            k = 0.12 - 0.023 * (lambda_nm - 700.0) / 100.0
        if lambda_nm < 700:
            k = 0.13 - 0.01 * (lambda_nm - 650.0) / 50.0
        if lambda_nm < 650:
            k = 0.15 - 0.02 * (lambda_nm - 600.0) / 50.0
        if lambda_nm < 600:
            k = 0.16 - 0.01 * (lambda_nm - 550.0) / 50.0
        if lambda_nm < 550:
            k = 0.18 - 0.02 * (lambda_nm - 500.0) / 50.0
        if lambda_nm < 500:
            k = 0.27 - 0.09 * (lambda_nm - 450.0) / 50.0
        if lambda_nm < 450:
            k = 0.35 - 0.08 * (lambda_nm - 400.0) / 50.0
        if lambda_nm < 400:
            k = 0.52 - 0.17 * (lambda_nm - 380.0) / 20.0
        if lambda_nm < 380:
            k = 0.64 - 0.12 * (lambda_nm - 360.0) / 20.0
        if lambda_nm < 360:
            k = 0.88 - 0.24 * (lambda_nm - 340.0) / 20.0
        if lambda_nm < 340:
            k = 1.42 - 0.54 * (lambda_nm - 320.0) / 20.0
        if lambda_nm < 320:
            k = 2.38 - 0.96 * (lambda_nm - 310.0) / 10.0
        if lambda_nm < 310:
            k = 2.38
    else:
        print("Error: gsAtmContOp: illegal opacity model")
        sys.exit(1)
    
    return k

@jit(nopython=True, cache=True)
def gsAtmContOp_fast(skytype, lambda_nm):
    """
    Continuum atmospheric opacity in magnitudes per airmass
    lambda in nm
    """
    k = 0.113 # V band opacity -- placeholder !!! 
    # Median extinction on Mauna Kea
    opacity_model = (skytype >> 8) & 0xf
    
    if opacity_model == 0x0:
        k = 0.075
        if lambda_nm < 900:
            k = 0.097 - 0.022 * (lambda_nm - 800.0) / 100.0
        if lambda_nm < 800:
            k = 0.12 - 0.023 * (lambda_nm - 700.0) / 100.0
        if lambda_nm < 700:
            k = 0.13 - 0.01 * (lambda_nm - 650.0) / 50.0
        if lambda_nm < 650:
            k = 0.15 - 0.02 * (lambda_nm - 600.0) / 50.0
        if lambda_nm < 600:
            k = 0.16 - 0.01 * (lambda_nm - 550.0) / 50.0
        if lambda_nm < 550:
            k = 0.18 - 0.02 * (lambda_nm - 500.0) / 50.0
        if lambda_nm < 500:
            k = 0.27 - 0.09 * (lambda_nm - 450.0) / 50.0
        if lambda_nm < 450:
            k = 0.35 - 0.08 * (lambda_nm - 400.0) / 50.0
        if lambda_nm < 400:
            k = 0.52 - 0.17 * (lambda_nm - 380.0) / 20.0
        if lambda_nm < 380:
            k = 0.64 - 0.12 * (lambda_nm - 360.0) / 20.0
        if lambda_nm < 360:
            k = 0.88 - 0.24 * (lambda_nm - 340.0) / 20.0
        if lambda_nm < 340:
            k = 1.42 - 0.54 * (lambda_nm - 320.0) / 20.0
        if lambda_nm < 320:
            k = 2.38 - 0.96 * (lambda_nm - 310.0) / 10.0
        if lambda_nm < 310:
            k = 2.38
    
    return k

@jit(nopython=True, cache=True)
def interp_lookup_fast(array, lambda_min, lambda_step, lambda_nm):
    if len(array) < 2:
        return 1.0 
    
    x = (lambda_nm - lambda_min) / lambda_step
    xint = int(math.floor(x))
    xfrac = x - xint
    if xint < 0:
        xint = 0
        xfrac = 0.0
    elif xint >= len(array) - 1:
        xint = len(array) - 2
        xfrac = 1.0
    return array[xint] * (1.0 - xfrac) + array[xint + 1] * xfrac

# ==================== THROUGHPUT & RELATED QUANTITIES ====================

@jit(nopython=True, cache=True)
def gsGeometricThroughput_fast(lambda_nm, r_eff, decent, fieldang, 
                              D_outer, centobs, rfov, rms_spot, EFL, fiber_ent_rad, seeing_fwhm_800,
                              Nu, du):
    """
    Computes geometric throughput (encircled energy in fiber)
    Optimized with Numba
    """
    # Nu, du passed as args
    
    # Get EFL and spot size for this field angle
    i = int(np.floor(4 * fieldang / rfov))
    if i < 0:
        sigma = rms_spot[0]
        EFL_val = EFL[0]
    elif i >= 4:
        sigma = rms_spot[4]
        EFL_val = EFL[4]
    else:
        frac = 4 * fieldang / rfov - i
        sigma = rms_spot[i] + (rms_spot[i+1] - rms_spot[i]) * frac
        EFL_val = EFL[i] + (EFL[i+1] - EFL[i]) * frac
    
    sigma *= ARCSEC_PER_URAD / EFL_val
    
    # Fiber radius in arcsec
    R = fiber_ent_rad / EFL_val * ARCSEC_PER_URAD
    
    # Seeing MTF
    uscale = 0.465 / seeing_fwhm_800 * (lambda_nm / 800.0) ** 0.2
    
    # Galaxy scale length
    rs = r_eff / RAT_HL_SL_EXP
    
    # The integral
    EE = 0.0
    for iu in range(Nu):
        u = (iu + 0.5) * du
        k = 2.0 * np.pi * u
        
        # Telescope PSF with diffraction
        theta_D = 0.001 * lambda_nm / D_outer / (1.0 - centobs) * ARCSEC_PER_URAD
        
        Gtilde = (np.exp(-k*k * sigma*sigma / 2.0 - (u / uscale) ** (5.0/3.0)) *
                  getJ0_fast(k * decent) *
                  np.exp(-4.0 / np.pi * u * theta_D))
        
        # Galaxy profile
        ftilde = (1.0 + k * rs * k * rs) ** (-1.5)
        
        EE += R * 2.0 * np.pi * du * getJ1_fast(2.0 * np.pi * u * R) * ftilde * Gtilde
    return EE

def gsGeometricThroughput(spectro, obs, lambda_nm, r_eff, decent, fieldang, flags):
    return gsGeometricThroughput_fast(
        lambda_nm, r_eff, decent, fieldang,
        spectro.D_outer, spectro.centobs, spectro.rfov, 
        spectro.rms_spot, spectro.EFL, spectro.fiber_ent_rad,
        obs.seeing_fwhm_800,
        CALC_CONFIG['geo_nu'], CALC_CONFIG['geo_du']
    )

@jit(nopython=True, cache=True)
def gsAeff_fast(i_arm, lambda_nm, fieldang, 
               D_outer, centobs, rfov, vignette, istart, l_arr, T_arr):
    """
    Computes effective area of the system
    Optimized with Numba
    """
    # Geometric area
    Aeff = (np.pi / 4.0 * D_outer * D_outer *
            (1 - centobs * centobs))
    
    # Vignetting
    i = int(np.floor(4 * fieldang / rfov))
    if i < 0:
        Vig = vignette[0]
    elif i >= 4:
        Vig = vignette[4]
    else:
        frac = 4 * fieldang / rfov - i
        Vig = vignette[i] + (vignette[i+1] - vignette[i]) * frac
    Aeff *= Vig
    
    # Throughput
    imin = istart[i_arm]
    imax = istart[i_arm + 1]
    
    if lambda_nm <= l_arr[imin]:
        Thr = T_arr[imin]
    elif lambda_nm >= l_arr[imax - 1]:
        Thr = T_arr[imax - 1]
    else:
        ti = imin
        while lambda_nm > l_arr[ti + 1]:
            ti += 1
        fr = (lambda_nm - l_arr[ti]) / (l_arr[ti+1] - l_arr[ti])
        Thr = T_arr[ti] + (T_arr[ti+1] - T_arr[ti]) * fr
    
    Aeff *= Thr
    return Aeff

def gsAeff(spectro, obs, i_arm, lambda_nm, fieldang):
    return gsAeff_fast(
        i_arm, lambda_nm, fieldang,
        spectro.D_outer, spectro.centobs, spectro.rfov, spectro.vignette,
        spectro.istart, spectro.l, spectro.T
    )

@jit(nopython=True, cache=True)
def gsSpectroMTF_fast(i_arm, lambda_nm, u,
                     diam, pix, rms_cam, Dtype, thick, temperature, fratio, dl, nline,
                     si_table):
    """
    Computes 1D Fourier transform of spectrograph PSF
    Optimized with Numba
    """
    mtf = 1.0
    
    # Fiber size
    D_spot = diam[i_arm] / pix[i_arm]
    if abs(D_spot * u) > 1e-6:
        mtf *= 2.0 * getJ1_fast(np.pi * D_spot * u) / (np.pi * D_spot * u)
    
    # Pixelization
    if abs(u) > 1e-9:
        mtf *= np.sin(np.pi * u) / (np.pi * u)
    
    # Spot size
    sigma = rms_cam[i_arm] / pix[i_arm]
    mtf *= np.exp(-2.0 * np.pi * np.pi * sigma * sigma * u * u)
    
    # Defocus in detector - Si only
    if Dtype[i_arm] == 0:
        N = 3
        ddepth = thick[i_arm] / N
        numer = 0.0
        denom = 0.0
        mfp = gsOP_Si_abslength_fast(lambda_nm, temperature[i_arm])
        nSi = gsOP_Si_indexreal_fast(lambda_nm, si_table)
        
        if mfp < 1e4 * ddepth:
            d0 = mfp * (1 - (1 + ddepth/mfp) * np.exp(-ddepth/mfp)) / (1 - np.exp(-ddepth/mfp))
        else:
            d0 = 0.5 * ddepth
        
        for i in range(2 * N):
            depth = d0 + ddepth * i
            contrib = np.exp(-ddepth * i / mfp) * (0.3 if depth > thick[i_arm] else 1.0)
            rspot = depth / nSi / 2.0 / fratio[i_arm] / pix[i_arm]
            arg = 2 * np.pi * rspot * u
            denom += contrib
            
            # MTF approximation
            numer += contrib * (0.1666666667 * np.cos(0.866025404 * arg) + 0.5 * np.cos(0.5 * arg) + 0.3333333333)
        
        if denom > 0:
            mtf *= numer / denom
    
    # Scattering from grating
    mtf *= np.exp(-lambda_nm / dl[i_arm] / nline[i_arm] * abs(u))
    
    return mtf

@jit(nopython=True, cache=True)
def gsSpectroDist_fast(i_arm, lambda_nm, pos, sigma, N, fr,
                      diam, pix, rms_cam, Dtype, thick, temperature, fratio, dl, nline, si_table):
    """
    Computes fraction of radiation in each l-pixel
    Optimized with Numba
    """
    for ip in range(N):
        fr[ip] = 0.0
    
    Nu = 1000
    du = 0.005
    
    for iu in range(Nu):
        u = du * (iu + 0.5)
        mtf1d = (gsSpectroMTF_fast(i_arm, lambda_nm, u, diam, pix, rms_cam, Dtype, thick, temperature, fratio, dl, nline, si_table) *
                 np.exp(-2.0 * np.pi * np.pi * sigma * sigma * u * u))
        for ip in range(N):
            fr[ip] += 2.0 * du * np.cos(2.0 * np.pi * u * (pos - ip)) * mtf1d

def gsSpectroDist(spectro, obs, i_arm, lambda_nm, pos, sigma, N, fr):
    gsSpectroDist_fast(
        i_arm, lambda_nm, pos, sigma, N, fr,
        spectro.diam, spectro.pix, spectro.rms_cam, spectro.Dtype, spectro.thick,
        spectro.temperature, spectro.fratio, spectro.dl, spectro.nline, _SI_TABLE
    )

def gsFracTrace(spectro, obs, i_arm, lambda_nm, tr):
    """
    Fraction of radiation in spectral trace
    """
    N = spectro.width[i_arm]
    FR = np.zeros(N)
    total = 0.0
    
    for j in range(-tr, tr + 1):
        gsSpectroDist(spectro, obs, i_arm, lambda_nm,
                     0.5 * (N - 1) + j * spectro.sep[i_arm] / spectro.pix[i_arm],
                     0, N, FR)
        total += np.sum(FR)
    
    return total

def gsAtmTrans(obs, lambda_nm, flags, AtmTransKP=None, MKTrans_3mm=None):
    """
    Atmospheric transmission as function of wavelength and observing conditions
    """
    if AtmTransKP is None:
        AtmTransKP = MODEL_ATMTRANS_KP
    if MKTrans_3mm is None:
        MKTrans_3mm = MODEL_MKTRANS_3MM

    k = gsAtmContOp(obs, lambda_nm, flags)
    trans = 10.0 ** (-0.4 * k / math.cos(obs.zenithangle * DEGREE))

    line_model = (obs.skytype >> 12) & 0xf

    def interp_lookup(array, lambda_min, lambda_step):
        if array is None or len(array) < 2:
            return None
        if lambda_min is None or lambda_step is None:
            lambda_min = 500.0
            lambda_step = 0.025
        x = (lambda_nm - lambda_min) / lambda_step
        xint = int(math.floor(x))
        xfrac = x - xint
        if xint < 0:
            xint = 0
            xfrac = 0.0
        elif xint >= len(array) - 1:
            xint = len(array) - 2
            xfrac = 1.0
        return array[xint] * (1.0 - xfrac) + array[xint + 1] * xfrac

    if line_model == 0x0:
        lookup = interp_lookup(
            AtmTransKP,
            MODEL_ATMTRANS_KP_LAMBDA_MIN,
            MODEL_ATMTRANS_KP_LAMBDA_STEP,
        )
        if lookup is not None:
            trans *= lookup
        else:
            print("WARNING: AtmTransKP data not loaded")

    elif line_model == 0x1:
        if lambda_nm > (MODEL_MKTRANS_3MM_LAMBDA_MIN or 900.0):
            lookup = interp_lookup(
                MKTrans_3mm,
                MODEL_MKTRANS_3MM_LAMBDA_MIN,
                MODEL_MKTRANS_3MM_LAMBDA_STEP,
            )
            if lookup is not None:
                trans *= lookup
            else:
                print("WARNING: MKTrans_3mm data not loaded")
        else:
            lookup = interp_lookup(
                AtmTransKP,
                MODEL_ATMTRANS_KP_LAMBDA_MIN,
                MODEL_ATMTRANS_KP_LAMBDA_STEP,
            )
            if lookup is not None:
                trans *= lookup
            else:
                print("WARNING: AtmTransKP data not loaded")
    else:
        print("Error: Unrecognized atmospheric line absorption model.")
        sys.exit(1)

    return trans

# ==================== SIGNAL AND NOISE ====================

def gsGetNoise(spectro, obs, i_arm, fieldang, t_exp, flags,
               gsSKY_UVES_NLINES=None, gsSKY_UVES_LAMBDA=None, gsSKY_UVES_INT=None,
               N_IR_OH_LINE=None, OHDATA=None, AtmTransKP=None, MKTrans_3mm=None):
    """Construct noise vector and associated sky model for a spectrograph arm."""

    if gsSKY_UVES_NLINES is None:
        gsSKY_UVES_NLINES = MODEL_GS_SKY_UVES_NLINES
    if gsSKY_UVES_LAMBDA is None:
        gsSKY_UVES_LAMBDA = MODEL_GS_SKY_UVES_LAMBDA
    if gsSKY_UVES_INT is None:
        gsSKY_UVES_INT = MODEL_GS_SKY_UVES_INT
    if N_IR_OH_LINE is None:
        N_IR_OH_LINE = MODEL_N_IR_OH_LINE
    if OHDATA is None:
        OHDATA = MODEL_OHDATA
    if AtmTransKP is None:
        AtmTransKP = MODEL_ATMTRANS_KP
    if MKTrans_3mm is None:
        MKTrans_3mm = MODEL_MKTRANS_3MM

    if N_IR_OH_LINE is None and OHDATA is not None:
        N_IR_OH_LINE = OHDATA.size // 2

    Npix = spectro.npix[i_arm]
    lmin = spectro.lmin[i_arm]
    dl = spectro.dl[i_arm]

    Noise = np.zeros(Npix)
    SkyMod = np.zeros(Npix)

    print(f" //Arm{spectro_arm(spectro, i_arm)}//")

    sample_factor = 1.0
    if spectro.Dtype[i_arm] == 1:
        sample_factor = 1.2  # HGCDTE_SUTR equivalent

    airmass = 1.0 / math.sqrt(1.0 - 0.96 * math.sin(obs.zenithangle * DEGREE) ** 2)

    i_field = int(math.floor(4 * fieldang / spectro.rfov))
    if i_field < 0:
        EFL = spectro.EFL[0]
    elif i_field >= 4:
        EFL = spectro.EFL[4]
    else:
        frac = 4 * fieldang / spectro.rfov - i_field
        EFL = spectro.EFL[i_field] + (spectro.EFL[i_field + 1] - spectro.EFL[i_field]) * frac

    rad = spectro.fiber_ent_rad / EFL * ARCSEC_PER_URAD

    sky_line_model = (obs.skytype >> 16) & 0xF

    if sky_line_model == 0x0:
        # Explicitly request no line emission (test mode)
        print("  --> Sky line model disabled (code 0x0)")

    elif sky_line_model == 0x1:
        if spectro.Dtype[i_arm] == 1:
            sample_factor = 1.2

        if gsSKY_UVES_NLINES is not None and gsSKY_UVES_LAMBDA is not None and gsSKY_UVES_INT is not None:
            print("  --> Computing Sky Lines Contribution ...")
            for iline in range(gsSKY_UVES_NLINES):
                lambda_air = gsSKY_UVES_LAMBDA[iline]
                lambda_nm = gs_air2vac(lambda_air)
                pos = (lambda_nm - lmin) / dl
                if -(SP_PSF_LEN / 2) < pos < Npix + SP_PSF_LEN / 2 - 1:
                    count = (gsSKY_UVES_INT[iline] * lambda_nm * 1e-12 * PHOTONS_PER_ERG_1NM *
                             gsFracTrace(spectro, obs, i_arm, lambda_nm, 1) *
                             gsAeff(spectro, obs, i_arm, lambda_nm, fieldang) *
                             t_exp * math.pi * rad * rad)
                    count = max(count, 0.0)
                    count *= airmass / 1.1 * math.exp(-gsAtmContOp(obs, lambda_nm, flags) * airmass / 1.086)

                    iref = int(math.floor(pos - (SP_PSF_LEN / 2 - 0.5)))
                    iref = max(0, min(Npix - SP_PSF_LEN, iref))
                    FR = np.zeros(SP_PSF_LEN)
                    gsSpectroDist(spectro, obs, i_arm, lambda_nm, pos - iref, 0, SP_PSF_LEN, FR)
                    for j in range(SP_PSF_LEN):
                        Noise[iref + j] += count * FR[j] * sample_factor
        else:
            print("WARNING: UVES sky line data not loaded")

        if N_IR_OH_LINE is not None and OHDATA is not None:
            for iline in range(N_IR_OH_LINE):
                lambda_nm = OHDATA[2 * iline]
                pos = (lambda_nm - lmin) / dl
                if -(SP_PSF_LEN / 2) < pos < Npix + SP_PSF_LEN / 2 - 1:
                    count = (OHDATA[2 * iline + 1] * lambda_nm * 1e-12 * PHOTONS_PER_ERG_1NM *
                             gsFracTrace(spectro, obs, i_arm, lambda_nm, 1) *
                             gsAeff(spectro, obs, i_arm, lambda_nm, fieldang) *
                             t_exp * math.pi * rad * rad)
                    count = max(count, 0.0)
                    count *= (airmass *
                              math.exp(-gsAtmContOp(obs, lambda_nm, flags) * airmass / 1.086) *
                              math.exp((14.8 - 15.8) / 1.086))
                    iref = int(math.floor(pos - SP_PSF_LEN / 2 + 0.5))
                    iref = max(0, min(Npix - SP_PSF_LEN, iref))
                    FR = np.zeros(SP_PSF_LEN)
                    gsSpectroDist(spectro, obs, i_arm, lambda_nm, pos - iref, 0, SP_PSF_LEN, FR)
                    for j in range(SP_PSF_LEN):
                        Noise[iref + j] += count * FR[j] * sample_factor
        else:
            print("WARNING: OH line data not loaded")
    else:
        raise RuntimeError(f"illegal sky line model: {(obs.skytype >> 16) & 0xF:x}")

    print("  --> Computing Sky Continuum Contribution ...")

    stride = CALC_CONFIG.get('snr_stride', 1)
    
    pixels_to_calc = range(Npix)
    if stride > 1:
         pixels_to_calc = list(range(0, Npix, stride))
         if pixels_to_calc[-1] != Npix - 1: pixels_to_calc.append(Npix - 1)
    
    continuum_noise = np.zeros(Npix)
    
    # Sparse calculation
    res_indices = []
    res_vals = []
    
    for ipix in pixels_to_calc:
        lambda_nm = lmin + (ipix + 0.5) * dl
        FR = np.zeros(SP_PSF_LEN)
        gsSpectroDist(spectro, obs, i_arm, lambda_nm, SP_PSF_LEN / 2 - 0.5, 0, SP_PSF_LEN, FR)

        num = den = 0.0
        for j in range(5 * SP_PSF_LEN):
            offset = lambda_nm + (0.2 * j - SP_PSF_LEN / 2 + 0.5) * dl
            trans = gsAtmTrans(obs, offset, flags, AtmTransKP, MKTrans_3mm)
            num += FR[j // 5] * trans
            den += FR[j // 5]
        trans = num / den if den > 0 else 0.0

        sky_cont_model = obs.skytype & 0xF

        if sky_cont_model == 0x0:
            continuum = 0.0
        elif sky_cont_model in (0x1, 0x2):
            if lambda_nm < 375:
                continuum = 0.17
            elif lambda_nm < 483:
                continuum = 0.14
            elif lambda_nm < 580:
                continuum = 0.09
            elif lambda_nm < 674.5:
                continuum = 0.10
            elif lambda_nm < 858:
                continuum = 0.08
            else:
                continuum = 0.07
            continuum /= 1.1
            if lambda_nm > 1040 and sky_cont_model == 0x1:
                continuum = 0.4669 * (1000.0 / lambda_nm) * (2.0 * lambda_nm / 1000.0 - 0.5)
            continuum *= 1e-11 * PHOTONS_PER_ERG_1NM * lambda_nm
            continuum *= airmass * trans
        elif sky_cont_model == 0x3:
            mag = (21.55 + (lambda_nm - 600) * (5e-5 if lambda_nm > 600 else -6e-3) -
                   0.55 * math.exp(-0.005 * (lambda_nm - 594) ** 2) -
                   0.175 * (1.0 + math.tanh(375 - lambda_nm)) -
                   6.14656e9 / lambda_nm ** 4)
            continuum = 0.01089 * 10 ** (0.4 * (22.5 - mag)) * 1e6 / lambda_nm ** 2
            continuum *= 1e-11 * PHOTONS_PER_ERG_1NM * lambda_nm
            continuum *= airmass * trans
        elif sky_cont_model == 0x4:
            continuum = (0.035 * math.sqrt(1000 / lambda_nm) +
                         0.045 * math.exp(-0.005 * (lambda_nm - 594) ** 2))
            continuum *= 10 ** (0.4 * (0.05 + 6.14656e9 / lambda_nm ** 4))
            continuum *= 1e-11 * PHOTONS_PER_ERG_1NM * lambda_nm
            continuum *= airmass * trans
        elif sky_cont_model == 0x5:
            mag = ((24.316 if lambda_nm > 600 else 27.166) +
                   ( -5.199e-03 if lambda_nm > 600 else -1.419e-02) * lambda_nm +
                   ( 1.465e-06 if lambda_nm > 600 else 8.541e-06) * lambda_nm * lambda_nm -
                   0.55 * math.exp(-0.005 * (lambda_nm - 594) ** 2) -
                   6.14656e9 / lambda_nm ** 4)
            continuum = 0.01089 * 10 ** (0.4 * (22.5 - mag)) * 1e6 / lambda_nm ** 2
            continuum *= 1e-11 * PHOTONS_PER_ERG_1NM * lambda_nm
            continuum *= airmass * trans
        elif sky_cont_model == 0x6:
            mag = ((24.316 if lambda_nm > 600 else 27.166) +
                   ( -5.199e-03 if lambda_nm > 600 else -1.419e-02) * lambda_nm +
                   ( 1.465e-06 if lambda_nm > 600 else 8.541e-06) * lambda_nm * lambda_nm -
                   0.55 * math.exp(-0.005 * (lambda_nm - 594) ** 2) -
                   6.14656e9 / lambda_nm ** 4)
            continuum = 0.01089 * 10 ** (0.4 * (22.5 - mag)) * 1e6 / lambda_nm ** 2
            continuum *= 1e-11 * PHOTONS_PER_ERG_1NM * lambda_nm
            scale = 1.0 if lambda_nm > 800 else (-1.02278215e-03 * lambda_nm + 1.77400498)
            continuum *= scale
            continuum *= airmass * trans
        else:
            raise RuntimeError(f"illegal sky continuum model: {sky_cont_model:x}")

        if obs.lunarZA < 90:
            lunarphase = obs.lunarphase - math.floor(obs.lunarphase)
            k = gsAtmContOp(obs, lambda_nm, flags)
            moon_model = (obs.skytype >> 4) & 0xF
            if moon_model == 0x0:
                scale_RS = ((math.exp(2480.0 / 550.0) - 1) /
                            (math.exp(2480.0 / lambda_nm) - 1) *
                            (lambda_nm / 550.0) ** -7.0)
                scale_MS = ((math.exp(2480.0 / 550.0) - 1) /
                            (math.exp(2480.0 / lambda_nm) - 1) *
                            (lambda_nm / 550.0) ** -4.3)
                kV = gsAtmContOp(obs, 550.0, flags)
                alpha = 360 * abs(lunarphase - 0.5)
                Istar = 10 ** (-0.4 * (3.84 + 0.026 * alpha + 4e-9 * alpha ** 4))
                if alpha < 7:
                    Istar *= 1.35 - 0.05 * alpha
                f1 = 2.29e5 * (1.06 + math.cos(obs.lunarangle * DEGREE) ** 2)
                f2 = 10 ** (6.15 - obs.lunarangle / 40.0)
                Bmoon = ((f1 * scale_RS + f2 * scale_MS) * Istar *
                         10 ** (-0.4 * kV / math.sqrt(1 - 0.96 * math.sin(obs.lunarZA * DEGREE) ** 2)) *
                         (1 - 10 ** (-0.4 * kV / math.sqrt(1 - 0.96 * math.sin(obs.zenithangle * DEGREE) ** 2))))
                lunar_cont = Bmoon / 3.408e10 * 5.48e10 / 550.0
            else:
                raise RuntimeError(f"illegal Moonlight model: {moon_model:x}")
            continuum += lunar_cont

        count = (continuum * dl * gsAeff(spectro, obs, i_arm, lambda_nm, fieldang) *
                 t_exp * math.pi * rad * rad *
                 gsFracTrace(spectro, obs, i_arm, lambda_nm, 1))
        
        if stride > 1:
            res_vals.append(count * sample_factor)
        else:
            Noise[ipix] += count * sample_factor

    if stride > 1:
        # Interpolate counts
        full_counts = np.interp(np.arange(Npix), pixels_to_calc, res_vals)
        Noise += full_counts

    SkyMod[:] = Noise / sample_factor


    for ipix in range(Npix):
        sky_sysref = SkyMod[ipix]
        for j in range(max(0, ipix - 1), min(Npix, ipix + 2)):
            sky_sysref = max(sky_sysref, SkyMod[j])
        Noise[ipix] += spectro.sysfrac ** 2 * sky_sysref ** 2

    sky_sysref = (np.sum(SkyMod) * spectro.width[i_arm] * spectro.pix[i_arm] /
                  spectro.sep[i_arm] / Npix)
    Noise += spectro.diffuse_stray * sky_sysref * sample_factor

    var = ((spectro.dark[i_arm] * t_exp * sample_factor +
            spectro.read[i_arm] ** 2) * spectro.width[i_arm])
    Noise += var

    return Noise, SkyMod, sample_factor


def gsGetSignal(spectro, obs, i_arm, lambda_nm, F, sigma_v, r_eff, decent,
                fieldang, t_exp, flags, AtmTransKP=None, MKTrans_3mm=None):
    """
    Computes signal for a spectral feature
    """
    NP_WIN = 32
    
    Npix = spectro.npix[i_arm]
    lmin = spectro.lmin[i_arm]
    lmax = spectro.lmax[i_arm]
    dl = spectro.dl[i_arm]
    
    Signal = np.zeros(Npix)
    
    # Find feature location
    pos = (lambda_nm - lmin) / dl
    iref = int(math.floor(pos - NP_WIN / 2.0))
    
    if iref < -NP_WIN or iref >= Npix:
        return Signal
    
    if iref < 0:
        iref = 0
    if iref > Npix - NP_WIN:
        iref = Npix - NP_WIN
    
    # Atmospheric transmission
    trans = den = 0.0
    step = CALC_CONFIG['atm_step']
    for x_val in np.arange(-4, 4.01, step):
        trans += (gsAtmTrans(obs, lambda_nm * (1 + x_val * sigma_v / 299792.458),
                            flags, AtmTransKP, MKTrans_3mm) *
                 math.exp(-0.5 * x_val * x_val))
        den += math.exp(-0.5 * x_val * x_val)
    trans /= den
    
    # Count photons from object
    counts = (F * trans *
             10 ** (-0.4 * gsGalactic_Alambda__EBV(lambda_nm) * obs.EBV) *
             gsGeometricThroughput(spectro, obs, lambda_nm, r_eff, decent, fieldang, flags) *
             gsFracTrace(spectro, obs, i_arm, lambda_nm, 0) *
             PHOTONS_PER_ERG_1NM * lambda_nm * t_exp *
             gsAeff(spectro, obs, i_arm, lambda_nm, fieldang) * 1e4)
    
    # Distribute light over pixels
    FR = np.zeros(NP_WIN)
    gsSpectroDist(spectro, obs, i_arm, lambda_nm, pos - iref,
                 sigma_v / 299792.458 * lambda_nm / dl, NP_WIN, FR)
    
    for ipix in range(NP_WIN):
        Signal[ipix + iref] = FR[ipix] * counts
    
    return Signal


def gsGetSNR(spectro, obs, i_arm, lambda_nm, F, sigma_v, r_eff, decent,
            fieldang, Noise, t_exp, flags, snrType,
            AtmTransKP=None, MKTrans_3mm=None):
    """
    Generate SNR for a spectral line
    """
    Npix = spectro.npix[i_arm]
    Signal = gsGetSignal(spectro, obs, i_arm, lambda_nm, F, sigma_v, r_eff,
                        decent, fieldang, t_exp, flags, AtmTransKP, MKTrans_3mm)
    
    SNR = 0.0
    
    # 1D optimal
    if snrType == 0:
        for ipix in range(Npix):
            if Noise[ipix] > 0:
                SNR += Signal[ipix] ** 2 / Noise[ipix]
        SNR = math.sqrt(SNR)
    
    # Uniform matched filter
    elif snrType == 1:
        numer = denom = 0.0
        for ipix in range(Npix):
            numer += Signal[ipix] ** 2
            denom += Signal[ipix] ** 2 * Noise[ipix]
        if numer >= 0.001:
            SNR = numer / math.sqrt(denom)
    
    return SNR

@jit(nopython=True, cache=True)
def compute_oii_trans_fast(lambda_vals, obs_skytype, obs_zenithangle, flags,
                          AtmTransKP, AtmTransKP_min, AtmTransKP_step,
                          MKTrans_3mm, MKTrans_3mm_min, MKTrans_3mm_step):
    trans = 0.0
    den = 0.0
    for x_val in np.arange(-4, 4.01, 0.1):
        lambda_avg = lambda_vals[0] + (0.5 + 0.5 * x_val) * (lambda_vals[1] - lambda_vals[0])
        
        # Inline gsAtmTrans_fast
        k = gsAtmContOp_fast(obs_skytype, lambda_avg)
        t_val = 10.0 ** (-0.4 * k / math.cos(obs_zenithangle * DEGREE))
        
        line_model = (obs_skytype >> 12) & 0xf
        if line_model == 0x0:
            if AtmTransKP.size > 2: # Check if valid
                t_val *= interp_lookup_fast(AtmTransKP, AtmTransKP_min, AtmTransKP_step, lambda_avg)
        elif line_model == 0x1:
            if lambda_avg > (MKTrans_3mm_min if MKTrans_3mm_min > 0 else 900.0):
                if MKTrans_3mm.size > 2:
                    t_val *= interp_lookup_fast(MKTrans_3mm, MKTrans_3mm_min, MKTrans_3mm_step, lambda_avg)
            else:
                if AtmTransKP.size > 2:
                    t_val *= interp_lookup_fast(AtmTransKP, AtmTransKP_min, AtmTransKP_step, lambda_avg)
        
        weight = np.exp(-0.5 * x_val * x_val)
        trans += t_val * weight
        den += weight
        
    return trans / den

def gsGetSNR_OII(spectro, obs, i_arm, z, F, sigma_v, r_eff, src_cont, ROII,
                decent, fieldang, Noise, t_exp, flags, snrType,
                AtmTransKP=None, MKTrans_3mm=None):
    """
    Obtains SNR for [OII] doublet
    """
    lambda_vals = np.array([372.71 * (1 + z), 372.98 * (1 + z)])
    
    if ROII < 0.667:
        ROII = 0.667
    if ROII > 3.87:
        ROII = 3.87
    
    frac = [ROII / (1 + ROII), 1.0 / (1 + ROII)]
    
    # Build noise vector including continuum
    Npix = spectro.npix[i_arm]
    myNoise = Noise.copy()
    
    # Atmospheric transmission
    # Use JIT helper
    AtmTransKP_min = MODEL_ATMTRANS_KP_LAMBDA_MIN if MODEL_ATMTRANS_KP_LAMBDA_MIN is not None else 500.0
    AtmTransKP_step = MODEL_ATMTRANS_KP_LAMBDA_STEP if MODEL_ATMTRANS_KP_LAMBDA_STEP is not None else 0.025
    MKTrans_3mm_min = MODEL_MKTRANS_3MM_LAMBDA_MIN if MODEL_MKTRANS_3MM_LAMBDA_MIN is not None else 900.0
    MKTrans_3mm_step = MODEL_MKTRANS_3MM_LAMBDA_STEP if MODEL_MKTRANS_3MM_LAMBDA_STEP is not None else 0.0002
    
    dummy = np.array([1.0, 1.0])
    
    trans = compute_oii_trans_fast(
        lambda_vals, obs.skytype, obs.zenithangle, flags,
        AtmTransKP if AtmTransKP is not None else dummy, AtmTransKP_min, AtmTransKP_step,
        MKTrans_3mm if MKTrans_3mm is not None else dummy, MKTrans_3mm_min, MKTrans_3mm_step
    )
    
    # Object continuum counts per Hz
    ll = (lambda_vals[0] + lambda_vals[1]) / 2.0
    counts = (src_cont * trans *
             10 ** (-0.4 * gsGalactic_Alambda__EBV(ll) * obs.EBV) *
             gsGeometricThroughput(spectro, obs, ll, r_eff, decent, fieldang, flags) *
             gsFracTrace(spectro, obs, i_arm, ll, 0) *
             PHOTONS_PER_ERG_1NM * ll * t_exp *
             gsAeff(spectro, obs, i_arm, ll, fieldang) * 1e4)
    
    if spectro.Dtype[i_arm] == 1:
        counts *= 1.2
    
    # Convert per Hz to per l-pixel
    counts *= 2.99792458e17 * spectro.dl[i_arm] / (ll * ll)
    
    # Add to noise vector
    for ipix in range(Npix):
        myNoise[ipix] += counts
    
    SNR = 0.0
    indivSNR = [0.0, 0.0]
    
    # 1D optimal - brighter feature
    if snrType == 0:
        for i in range(2):
            indivSNR[i] = gsGetSNR(spectro, obs, i_arm, lambda_vals[i],
                                  frac[i] * F, sigma_v, r_eff, decent, fieldang,
                                  myNoise, t_exp, flags, 0, AtmTransKP, MKTrans_3mm)
        SNR = max(indivSNR)
    
    # Uniform matched filter - brighter feature
    elif snrType == 1:
        for i in range(2):
            indivSNR[i] = gsGetSNR(spectro, obs, i_arm, lambda_vals[i],
                                  frac[i] * F, sigma_v, r_eff, decent, fieldang,
                                  myNoise, t_exp, flags, 1, AtmTransKP, MKTrans_3mm)
        SNR = max(indivSNR)
    
    # Combined optimal of both lines
    elif snrType == 2:
        Signal0 = gsGetSignal(spectro, obs, i_arm, lambda_vals[0],
                            frac[0] * F, sigma_v, r_eff, decent, fieldang,
                            t_exp, flags, AtmTransKP, MKTrans_3mm)
        Signal1 = gsGetSignal(spectro, obs, i_arm, lambda_vals[1],
                            frac[1] * F, sigma_v, r_eff, decent, fieldang,
                            t_exp, flags, AtmTransKP, MKTrans_3mm)
        
        SNR = 0.0
        for ipix in range(Npix):
            if myNoise[ipix] > 0:
                SNR += (Signal0[ipix] + Signal1[ipix]) ** 2 / myNoise[ipix]
        SNR = math.sqrt(SNR)
    
    return SNR


def gsGetSNR_Single(spectro, obs, i_arm, mag, lambda_nm, F, sigma_v, r_eff,
                    decent, fieldang, Noise, t_exp, flags, snrType,
                    AtmTransKP=None, MKTrans_3mm=None):
    """
    Generates the signal/noise ratio for a single emission line given the noise vector Noise 
    taking into consideration the continuum effect.
    """
    Npix = spectro.npix[i_arm]
    
    # Handle magnitude interpolation from input file if mag == -99.9
    if mag == -99.9:
        if len(lambda_inmag2) == 0:
            # Fallback if no data loaded, though this shouldn't happen if inputs are correct
            mag = 22.5 
        else:
            # Simple linear interpolation
            mag = np.interp(lambda_nm, lambda_inmag2, mag_inmag2)

    # Atmospheric transmission
    trans = den = 0.0
    for x_val in np.arange(-4, 4.01, 0.2):
        trans += (gsAtmTrans(obs, lambda_nm, flags, AtmTransKP, MKTrans_3mm) *
                 math.exp(-0.5 * x_val * x_val))
        den += math.exp(-0.5 * x_val * x_val)
    trans /= den

    # Determine how many counts per Hz we get from the object continuum
    src_cont = 3.631e-20 * 10 ** (-0.4 * mag) # in erg/cm2/s
    
    counts = (src_cont * trans *
             10 ** (-0.4 * gsGalactic_Alambda__EBV(lambda_nm) * obs.EBV) *
             gsGeometricThroughput(spectro, obs, lambda_nm, r_eff, decent, fieldang, flags) *
             gsFracTrace(spectro, obs, i_arm, lambda_nm, 0) *
             PHOTONS_PER_ERG_1NM * lambda_nm * t_exp *
             gsAeff(spectro, obs, i_arm, lambda_nm, fieldang) * 1e4)

    if spectro.Dtype[i_arm] == 1:
        counts *= 1.2

    # Convert from per Hz --> l-per pixel
    counts *= 2.99792458e17 * spectro.dl[i_arm] / (lambda_nm * lambda_nm)

    # Allocate and get the signal vector
    Signal = gsGetSignal(spectro, obs, i_arm, lambda_nm, F, sigma_v, r_eff, decent, fieldang, t_exp, flags, AtmTransKP, MKTrans_3mm)

    SNR = 0.0
    
    # 1D optimal
    if snrType == 0:
        for ipix in range(Npix):
            if (counts + Noise[ipix]) > 0:
                SNR += Signal[ipix] * Signal[ipix] / (counts + Noise[ipix])
        SNR = math.sqrt(SNR)

    # uniform matched filter
    elif snrType == 1:
        numer = denom = 0.0
        for ipix in range(Npix):
            numer += Signal[ipix] * Signal[ipix]
            denom += Signal[ipix] * Signal[ipix] * (counts + Noise[ipix])
        
        if numer >= 0.001:
            SNR = numer / math.sqrt(denom)
        else:
            SNR = 0.0

    return SNR


def gsGetSNR_Continuum(spectro, obs, i_arm, mag, r_eff, decent, fieldang,
                       Noise, t_exp, flags, AtmTransKP=None, MKTrans_3mm=None):
    """
    Continuum S/N per pixel per exposure for given AB magnitude.

    Returns
    -------
    out_SNR_curve : ndarray, shape (Npix,)
        SNR per pixel per single exposure.
    out_signal_counts : ndarray, shape (Npix,)
        Source photon counts [e⁻] per pixel per single exposure.
    out_noise_variance : ndarray, shape (Npix,)
        Total noise variance [e⁻²] per pixel per single exposure,
        i.e. (sample_factor * signal_counts) + sky_noise_variance.
        This is the denominator used when computing SNR.
    """
    dl = spectro.dl[i_arm]
    Npix = spectro.npix[i_arm]

    out_SNR_curve      = np.zeros(Npix)
    out_signal_counts  = np.zeros(Npix)   # NEW: source photon counts per pixel
    out_noise_variance = np.zeros(Npix)   # NEW: total noise variance per pixel

    sample_factor = 1.2 if spectro.Dtype[i_arm] == 1 else 1.0

    stride = CALC_CONFIG.get('snr_stride', 1)

    # Define pixels to calculate
    if stride > 1:
        pixels_to_calc = list(range(0, Npix, stride))
        if pixels_to_calc[-1] != Npix - 1:
            pixels_to_calc.append(Npix - 1)
    else:
        pixels_to_calc = range(Npix)

    calc_snr  = {}  # ipix -> snr
    calc_sig  = {}  # ipix -> signal counts
    calc_nvar = {}  # ipix -> noise variance

    for ipix in pixels_to_calc:
        lambda_nm = spectro.lmin[i_arm] + spectro.dl[i_arm] * ipix

        # Atmospheric transmission (PSF-weighted average over the pixel)
        FR = np.zeros(SP_PSF_LEN)
        gsSpectroDist(spectro, obs, i_arm, lambda_nm, 7.5, 0, SP_PSF_LEN, FR)

        num = den = 0.0
        n_samp = CALC_CONFIG['cont_samp']
        # Cover the PSF footprint: range [-SP_PSF_LEN/2, +SP_PSF_LEN/2] in pixel units
        for j in range(n_samp * SP_PSF_LEN):
            offset = (j / n_samp) - (SP_PSF_LEN / 2.0) + (0.5 / n_samp)
            trans = gsAtmTrans(obs, lambda_nm + offset * dl,
                               flags, AtmTransKP, MKTrans_3mm)
            idx = j // n_samp
            if idx < SP_PSF_LEN:
                num += FR[idx] * trans
                den += FR[idx]
        trans = num / den if den > 0 else 0.0

        # Source continuum photon counts per pixel per exposure [e⁻]
        src_cont = 3.631e-20 * 10 ** (-0.4 * mag)  # F_nu [erg/cm²/s/Hz]
        counts = (src_cont * trans *
                  10 ** (-0.4 * gsGalactic_Alambda__EBV(lambda_nm) * obs.EBV) *
                  gsGeometricThroughput(spectro, obs, lambda_nm, r_eff, decent, fieldang, flags) *
                  gsFracTrace(spectro, obs, i_arm, lambda_nm, 0) *
                  PHOTONS_PER_ERG_1NM * lambda_nm * t_exp *
                  gsAeff(spectro, obs, i_arm, lambda_nm, fieldang) * 1e4)
        # Convert F_nu [per Hz] → F_lambda [per l-pixel]
        counts *= 2.99792458e17 * spectro.dl[i_arm] / (lambda_nm * lambda_nm)

        # Total noise variance = Poisson(signal) + sky/detector noise
        noise_var = sample_factor * counts + Noise[ipix]

        # SNR
        if noise_var > 0:
            snr_val = counts / math.sqrt(noise_var)
        else:
            snr_val = 0.0

        calc_snr[ipix]  = snr_val
        calc_sig[ipix]  = counts
        calc_nvar[ipix] = noise_var

    # Interpolate sparse results onto full pixel grid if stride > 1
    if stride > 1:
        indices = sorted(calc_snr.keys())
        out_SNR_curve      = np.interp(np.arange(Npix), indices, [calc_snr[k]  for k in indices])
        out_signal_counts  = np.interp(np.arange(Npix), indices, [calc_sig[k]  for k in indices])
        out_noise_variance = np.interp(np.arange(Npix), indices, [calc_nvar[k] for k in indices])
    else:
        for ipix in calc_snr:
            out_SNR_curve[ipix]      = calc_snr[ipix]
            out_signal_counts[ipix]  = calc_sig[ipix]
            out_noise_variance[ipix] = calc_nvar[ipix]

    return out_SNR_curve, out_signal_counts, out_noise_variance


# ==================== I/O FUNCTIONS ====================

def gsReadSpectrographConfig(FileName, spectro):
    """
    Read a spectrograph configuration file
    """
    
    # Set defaults
    spectro.D_outer = -1
    spectro.rms_spot = np.full(5, -1.0, dtype=np.float64)
    spectro.fiber_ent_rad = -1
    spectro.N_arms = 0
    spectro.vignette = np.ones(5, dtype=np.float64)
    spectro.Dtype = np.zeros(MAXARM, dtype=np.int32)
    
    try:
        with open(FileName, 'r') as fp:
            lines = fp.readlines()
    except IOError:
        print(f"Error: Can't read file: {FileName}")
        sys.exit(1)
    
    line_idx = 0
    while line_idx < len(lines):
        InfoLine = lines[line_idx].strip()
        line_idx += 1
        
        # Skip comments and empty lines
        if not InfoLine or InfoLine.startswith('#'):
            continue
        
        # OPTICS keyword
        if InfoLine.startswith("OPTICS"):
            parts = InfoLine.split()[1:]
            if len(parts) >= 8:
                spectro.D_outer = float(parts[0])
                spectro.centobs = float(parts[1])
                spectro.rfov = float(parts[2])
                spectro.EFL = np.array([float(parts[i]) for i in range(3, 8)], dtype=np.float64)
            else:
                print(f"Error: OPTICS line has insufficient arguments")
                sys.exit(1)
        
        # SPOT keyword
        elif InfoLine.startswith("SPOT"):
            parts = InfoLine.split()[1:]
            if len(parts) >= 5:
                spectro.rms_spot = np.array([float(parts[i]) for i in range(5)], dtype=np.float64)
            else:
                print(f"Error: SPOT line has insufficient arguments")
                sys.exit(1)
        
        # FIBER keyword
        elif InfoLine.startswith("FIBER"):
            parts = InfoLine.split()[1:]
            if len(parts) >= 1:
                spectro.fiber_ent_rad = float(parts[0])
            else:
                print(f"Error: FIBER line has insufficient arguments")
                sys.exit(1)
        
        # ARMS keyword
        elif InfoLine.startswith("ARMS"):
            parts = InfoLine.split()[1:]
            if len(parts) >= 1:
                spectro.N_arms = int(parts[0])
                for i in range(spectro.N_arms):
                    spectro.lmin[i] = -1.0
                    spectro.fratio[i] = -1.0
                    spectro.nline[i] = 1e12
            else:
                print(f"Error: ARMS line has insufficient arguments")
                sys.exit(1)
        
        # PARAM keyword
        elif InfoLine.startswith("PARAM"):
            parts = InfoLine.split()[1:]
            if len(parts) >= 5:
                i = int(parts[0])
                lmin = float(parts[1])
                lmax = float(parts[2])
                npix = int(parts[3])
                width = int(parts[4])
                
                if i < 0 or i >= spectro.N_arms:
                    print(f"Error: illegal PARAM line: arm #{i} does not exist")
                    sys.exit(1)
                
                spectro.lmin[i] = lmin
                spectro.lmax[i] = lmax
                spectro.npix[i] = npix
                spectro.dl[i] = (lmax - lmin) / npix
                spectro.width[i] = width
            else:
                print(f"Error: PARAM line has insufficient arguments")
                sys.exit(1)
        
        # CAMERA keyword
        elif InfoLine.startswith("CAMERA"):
            parts = InfoLine.split()[1:]
            if len(parts) >= 10:
                i = int(parts[0])
                
                if i < 0 or i >= spectro.N_arms:
                    print(f"Error: illegal CAMERA line: arm #{i} does not exist")
                    sys.exit(1)
                
                spectro.fratio[i] = float(parts[1])
                spectro.thick[i] = float(parts[2])
                spectro.pix[i] = float(parts[3])
                spectro.temperature[i] = float(parts[4])
                spectro.rms_cam[i] = float(parts[5])
                spectro.diam[i] = float(parts[6])
                spectro.dark[i] = float(parts[7])
                spectro.read[i] = float(parts[8])
                spectro.sep[i] = float(parts[9])
            else:
                print(f"Error: CAMERA line has insufficient arguments")
                sys.exit(1)
        
        # THRPUT keyword
        elif InfoLine.startswith("THRPUT"):
            i = 0
            i_arm = 0
            spectro.istart[0] = 0
            
            while line_idx < len(lines):
                InfoLine = lines[line_idx].strip()
                line_idx += 1
                
                if not InfoLine or InfoLine.startswith('#'):
                    continue
                
                # 'D' marks delimiter between arms
                if InfoLine.startswith("D"):
                    i_arm += 1
                    spectro.istart[i_arm] = i
                    if i_arm == spectro.N_arms:
                        spectro.N_thr = i
                        break
                    continue
                
                parts = InfoLine.split()
                if len(parts) >= 6:
                    spectro.l[i] = float(parts[0])
                    temp = [float(parts[j]) for j in range(1, 6)]
                    spectro.T[i] = temp[0] * temp[1] * temp[2] * temp[3] * temp[4]
                    i += 1
                else:
                    print(f"Error: illegal throughput table line")
                    sys.exit(1)
            
            # Fix: Ensure the last arm's end index is set
            spectro.istart[i_arm + 1] = i
            spectro.N_thr = i
        
        # VIGNET keyword
        elif InfoLine.startswith("VIGNET"):
            parts = InfoLine.split()[1:]
            if len(parts) >= 5:
                spectro.vignette = np.array([float(parts[i]) for i in range(5)], dtype=np.float64)
            else:
                print(f"Error: VIGNET line has insufficient arguments")
                sys.exit(1)
        
        # HGCDTE keyword
        elif InfoLine.startswith("HGCDTE"):
            parts = InfoLine.split()[1:]
            if len(parts) >= 1:
                i = int(parts[0])
                if i < 0 or i >= spectro.N_arms:
                    print(f"Error: HGCDTE {i}: illegal arm index")
                    sys.exit(1)
                spectro.Dtype[i] = 1
        
        # NLINES keyword
        elif InfoLine.startswith("NLINES"):
            parts = InfoLine.split()[1:]
            if len(parts) >= 2:
                i = int(parts[0])
                nlines = float(parts[1])
                if i < 0 or i >= spectro.N_arms:
                    print(f"Error: NLINES {i}: illegal arm index")
                    sys.exit(1)
                if nlines < 10:
                    print(f"Error: NLINES {i}: nlines={nlines} is illegal")
                    sys.exit(1)
                spectro.nline[i] = nlines
    
    # Validation tests
    if spectro.D_outer <= 0:
        print("Error: illegal outer diameter or no OPTICS line")
        sys.exit(1)
    
    for i in range(5):
        if spectro.rms_spot[i] < 0:
            print(f"Error: illegal rms spot or no SPOT line: spot[{i}]={spectro.rms_spot[i]:.5e}")
            sys.exit(1)
    
    if spectro.fiber_ent_rad <= 0:
        print("Error: illegal fiber radius or no FIBER line")
        sys.exit(1)
    
    if spectro.N_arms <= 0:
        print(f"Error: {spectro.N_arms} arms illegal or no ARMS line")
        sys.exit(1)
    
    for i in range(spectro.N_arms):
        if spectro.lmin[i] < 0:
            print(f"Error: illegal lambda min = {spectro.lmin[i]} or no PARAM {i} line")
            sys.exit(1)
    
    # Initially set no systematics
    spectro.sysfrac = 0.0
    
    return


# ==================== MAIN PROGRAM ====================

def main():
    """
    Main program - exposure time calculator
    """
    global lambda_inmag, mag_inmag, lambda_inmag2, mag_inmag2, num_inmag, num_inmag2
    
    print("Compiler flags:")
    print(" NOTE: Python version (Optimized) - check code for flag implementations")
    print()
    
    # Initialize
    ngal = np.zeros(NZ_OII, dtype=np.int32)
    
    # Get spectrograph properties
    FileName = input("Enter spectrograph configuration file: ")
    spectro = SpectroAttrib()
    gsReadSpectrographConfig(FileName, spectro)
    
    # Allocate noise vectors
    spNoise = []
    spSky = []
    spSample = []
    for ia in range(spectro.N_arms):
        spNoise.append(np.zeros(spectro.npix[ia]))
        spSky.append(np.zeros(spectro.npix[ia]))
        spSample.append(1.0)
    
    snrType = 2
    
    # Get observational conditions
    obs = ObsAttrib()
    
    skytype_input = input("Enter observational conditions [hexadecimal code; suggested=10003]: ")
    obs.skytype = int(skytype_input, 16)
    
    obs.seeing_fwhm_800 = float(input("Enter seeing [arcsec FWHM @ lambda=800nm]: "))
    obs.zenithangle = float(input("Enter zenith angle [degrees]: "))
    obs.EBV = float(input("Enter Galactic dust column [magnitudes E(B-V)]: "))
    fieldang = float(input("Enter field angle [degrees]: "))
    decent = float(input("Enter fiber astrometric offset [arcsec]: "))
    
    # Moon conditions - default below horizon
    obs.lunarZA = 135.0
    obs.lunarangle = 90.0
    obs.lunarphase = 0.25
    
    # Exposure time and systematics
    t = float(input("Enter time per exposure [s]: "))
    n_exp = int(input("Enter number of exposures: "))
    spectro.sysfrac = float(input("Enter systematic sky subtraction floor [rms per 1D pixel]: "))
    spectro.sysfrac *= math.sqrt(n_exp)  # Prevent averaging down
    spectro.diffuse_stray = float(input("Enter diffuse stray light [fraction of total]: "))
    
    # Output files
    flag_reused = int(input("Noise data reused?: [1=yes/0=no] ") or 0)
    OutFileNoise = input("Enter output file for noise vector: ")
    OutFileSNR = input("Enter output file for ELG S/N curve: [- for no output] ")

    # Single line SNR inputs
    OutFileSNR2 = input("Enter output file for single line S/N curve: [- for no output] ")
    flux_emi = float(input("Enter flux of the emission line [erg cm-2 s-1]: ") or 1.0e-17)
    sigma_emi = float(input("Enter velocity width of the emission line [km s-1]: ") or 70.0)

    OutFileSNRAB = input("Enter output file for continuum curve: [- for no output] ")
    
    # [OII] detection files
    InFileOII = input("Enter [OII] input catalogue file: [- for no OII computation] ")
    if InFileOII != "-":
        OutFileOII = input("Enter [OII] output catalogue file: ")
        min_SNR = float(input("Enter minimum SNR: "))

    # Magnitude input file
    InFileMag = input("Enter magnitude input file: [- for no designated file] ")

    # Read magnitude file if provided
    mag_val_for_calc = 22.5 # Default if not using file
    
    if InFileMag != "-":
        try:
            # We can use numpy loadtxt/genfromtxt or manual reading.
            # C code does manual reading effectively.
            data = np.loadtxt(InFileMag, comments='#')
            if data.ndim == 1:
                data = data.reshape(1, -1) # Handle single line case
            
            # Sort by lambda just in case
            data = data[data[:, 0].argsort()]
            
            lambda_inmag = data[:, 0]
            mag_inmag = data[:, 1]
            num_inmag = len(lambda_inmag)
            
            # Create the padded arrays (lambda_inmag2) as in C code
            # "Modified by K.Yabe 20160703"
            
            lambda_inmag2 = np.zeros(num_inmag + 2)
            mag_inmag2 = np.zeros(num_inmag + 2)
            
            # Pad beginning
            if lambda_inmag[0] > 300.0:
                lambda_inmag2[0] = 300.0
                mag_inmag2[0] = 99.9
            else:
                lambda_inmag2[0] = lambda_inmag[0] - 1.0
                mag_inmag2[0] = mag_inmag[0]
                
            # Copy data
            lambda_inmag2[1:-1] = lambda_inmag
            mag_inmag2[1:-1] = mag_inmag
            
            # Pad end
            if lambda_inmag[-1] < 20000.0:
                lambda_inmag2[-1] = 20000.0
                mag_inmag2[-1] = 99.9
            else:
                lambda_inmag2[-1] = lambda_inmag[-1] + 1.0
                mag_inmag2[-1] = mag_inmag[-1]
                
            num_inmag2 = len(lambda_inmag2)
            mag_val_for_calc = -99.9 # Flag to use file
            
            print(f"Loaded {num_inmag} points from magnitude file.")
            
        except Exception as e:
            print(f"Error reading magnitude file: {e}")
            sys.exit(1)
    else:
        # Reset globals just in case
        lambda_inmag2 = []
        mag_inmag2 = []
    
    # Encircled energy in fiber
    print(f"Fiber aperture factor [@800nm, r_eff=0.3\"(exp)] = {gsGeometricThroughput(spectro, obs, 800, 0.3, 0.0, 0, 0x0):.8f}")
    print(f"Fiber aperture factor [@800nm,    point source] = {gsGeometricThroughput(spectro, obs, 800, 0.0, 0.0, 0, 0x0):.8f}")
    
    if MODEL_DATA_AVAILABLE:
        gsSKY_UVES_NLINES = MODEL_GS_SKY_UVES_NLINES
        gsSKY_UVES_LAMBDA = MODEL_GS_SKY_UVES_LAMBDA
        gsSKY_UVES_INT = MODEL_GS_SKY_UVES_INT
        N_IR_OH_LINE = MODEL_N_IR_OH_LINE
        OHDATA = MODEL_OHDATA
        if N_IR_OH_LINE is None and OHDATA is not None:
            N_IR_OH_LINE = OHDATA.size // 2
        AtmTransKP = MODEL_ATMTRANS_KP
        MKTrans_3mm = MODEL_MKTRANS_3MM
        print("\nLoaded atmospheric and sky emission tables from modeldata.py\n")
    else:
        gsSKY_UVES_NLINES = None
        gsSKY_UVES_LAMBDA = None
        gsSKY_UVES_INT = None
        N_IR_OH_LINE = None
        OHDATA = None
        AtmTransKP = None
        MKTrans_3mm = None
        print("\n*** WARNING: External data tables not found ***")
        print("Missing Gemini/modeldata.py — continuing with limited functionality.\n")
    
    # Generate and write noise vector
    calc_noise = True
    if flag_reused == 1:
        pass

    t0 = time.time()
    print("Computing noise vector ...")
    for ia in range(spectro.N_arms):
        (spNoise[ia], spSky[ia], spSample[ia]) = gsGetNoise(
            spectro, obs, ia, fieldang, t, 0x0,
            gsSKY_UVES_NLINES, gsSKY_UVES_LAMBDA, gsSKY_UVES_INT,
            N_IR_OH_LINE, OHDATA, AtmTransKP, MKTrans_3mm)
    
    with open(OutFileNoise, 'w') as fp:
        for ia in range(spectro.N_arms):
            arm_id = spectro_arm(spectro, ia)
            for i in range(spectro.npix[ia]):
                lambda_nm = spectro.lmin[ia] + spectro.dl[ia] * (i + 0.5)
                fp.write(
                    f"{arm_id:1d} {i:4d} {lambda_nm:7.4f} {spNoise[ia][i]:11.5e} {spSky[ia][i]:11.5e}\n"
                )
            fp.write("\n")
    print(" Done.\n")
    

    # Generate ELG S/N curve
    t1 = time.time()
    if OutFileSNR != "-":
        print("Computing ELG SNR curve for fiducial parameters ...")
        REF_SIZE = 0.30
        
        # Stride for Z-step
        stride = CALC_CONFIG.get('snr_stride', 1)
        z_step = 0.0002 * stride
        
        try:
            with open(OutFileSNR, 'w') as fp:
                z = 0.1
                while z < 1.5001:
                    snrtot = 0.0
                    snr = [0.0] * spectro.N_arms
                    
                    for ia in range(spectro.N_arms):
                        if (spectro.lmin[ia] < 373.8 * (1 + z) and
                            371.8 * (1 + z) < spectro.lmax[ia]):
                            snr[ia] = (gsGetSNR_OII(spectro, obs, ia, z, 1e-16, 70.0, REF_SIZE,
                                                   0.0, 1.0, decent, fieldang, spNoise[ia], t, 0x0,
                                                   snrType, AtmTransKP, MKTrans_3mm) *
                                      math.sqrt(n_exp))
                        snrtot += snr[ia] ** 2
                    
                    snrtot = math.sqrt(snrtot)
                    
                    # Compute effective area
                    Aeff = 0.0
                    for ia in range(spectro.N_arms):
                        if (spectro.lmin[ia] < 372.845 * (1 + z) and
                            372.845 * (1 + z) < spectro.lmax[ia]):
                            Aeff += gsAeff(spectro, obs, ia, 372.845 * (1 + z), fieldang)
                    
                    fp.write(f"{z:6.4f} {372.71*(1+z):7.2f} {372.98*(1+z):7.2f} "
                            f"{gsGeometricThroughput(spectro, obs, 372.845*(1+z), REF_SIZE, decent, 0, 0x0):8.6f} "
                            f"{Aeff:8.5f}")
                    
                    for ia in range(spectro.N_arms):
                        fp.write(f" {snr[ia]:8.4f}")
                    fp.write(f" {snrtot:8.4f}\n")
                    
                    z += z_step
        except Exception as e:
            print(f"Error writing ELG file: {e}")
        
        print(" Done.\n")
    print(f" [Timing] ELG SNR Calculation: {time.time() - t1:.2f} s")

    t2 = time.time()
    # Generate Single Line S/N curve
    if OutFileSNR2 != "-":
        print(f"Computing SNR curve for a single line with f={flux_emi:.2e} [erg cm-2 s-1], sigma={sigma_emi:.0f} [km s-1] ...")
        
        ref_input = 0.0 
        
        # Stride for Z-step
        stride = CALC_CONFIG.get('snr_stride', 1)
        z_step = 0.0002 * stride
        
        with open(OutFileSNR2, 'w') as fp:
            lambda_line_min = 345.5
            zmin = 0.1
            zmax = 2.7627
            
            z = zmin
            while z < zmax + 0.00001:
                target_lambda = lambda_line_min * (1 + z)
                
                snrtot = 0.0
                Aeff = 0.0
                snr_arm = [0.0] * spectro.N_arms
                
                for ia in range(spectro.N_arms):
                    if (spectro.lmin[ia] < target_lambda and target_lambda < spectro.lmax[ia]):
                        snr_val = gsGetSNR_Single(spectro, obs, ia, mag_val_for_calc, target_lambda,
                                                 flux_emi, sigma_emi, ref_input, decent, fieldang,
                                                 spNoise[ia], t, 0x0, 0, AtmTransKP, MKTrans_3mm)
                        snr_arm[ia] = snr_val * math.sqrt(n_exp)
                        Aeff += gsAeff(spectro, obs, ia, target_lambda, fieldang)
                    
                    snrtot += snr_arm[ia] ** 2
                
                snrtot = math.sqrt(snrtot)
                fiber_aperture = gsGeometricThroughput(spectro, obs, target_lambda, ref_input, decent, fieldang, 0x0)
                
                fp.write(f"{target_lambda:7.2f} {fiber_aperture:8.6f} {Aeff:8.5f}")
                for ia in range(spectro.N_arms):
                    fp.write(f" {snr_arm[ia]:8.4f}")
                fp.write(f" {snrtot:8.4f}\n")
                
                z += z_step
        print(" Done.\n")
    print(f" [Timing] Single Line SNR Calculation: {time.time() - t2:.2f} s")

    t3 = time.time()
    # Generate continuum S/N curve
    if OutFileSNRAB != "-":
        print("Computing continuum SNR curve for fiducial parameters ...")
        
        with open(OutFileSNRAB, 'w') as fp:
            target_mag = mag_val_for_calc if mag_val_for_calc == -99.9 else 22.5
            for ia in range(spectro.N_arms):
                # gsGetSNR_Continuum returns (snr_curve, signal_counts, noise_variance)
                snrcont, _sig, _nvar = gsGetSNR_Continuum(
                    spectro, obs, ia, target_mag, 0.0, decent,
                    fieldang, spNoise[ia], t, 0x0,
                    AtmTransKP, MKTrans_3mm)

                # Write full curve (interpolated inside gsGetSNR_Continuum if stride>1)
                for j in range(spectro.npix[ia]):
                    lambda_nm = spectro.lmin[ia] + spectro.dl[ia] * j
                    fp.write(f"{ia:1d} {j:4d} {lambda_nm:9.3f} "
                            f"{snrcont[j] * math.sqrt(n_exp):8.4f}\n")
        
        print(" Done.\n")
    print(f" [Timing] Continuum SNR Calculation: {time.time() - t3:.2f} s")

    t4 = time.time()
    # [OII] catalog processing
    if InFileOII != "-":
        print("Processing [OII] emitter catalog ...")
        
        # Determine maximum detection limit
        snrmax16 = 1.0


if __name__ == "__main__":
    # To run the program, you need to:
    # 1. Load external data files (modeldata.h equivalent)
    # 2. Prepare spectrograph configuration file
    # 3. Optionally prepare [OII] catalog file
    
    # Example standalone usage:
    # main()
    
    print("="*60)
    print("PFS Exposure Time Calculator - Python Version (Optimized)")
    print("="*60)
    print()
    print("NOTES FOR USAGE:")
    if MODEL_DATA_AVAILABLE:
        print("1. Atmospheric and sky tables are available via Gemini/modeldata.py.")
    else:
        print("1. WARNING: Gemini/modeldata.py not found — generate it from ETC-JUST_v1/src/modeldata.h.")
    print("2. Prepare a spectrograph configuration file")
    print()
    print("3. C-specific features that need attention:")
    print("   - Compiler flags (#ifdef) are noted in comments")
    print("   - File I/O uses Python's native functions")
    print("   - scanf() replaced with input()")
    print("   - malloc/free replaced with Python lists/numpy arrays")
    print()
    print("4. To run: uncomment main() call at end of script")
    print("="*60)
    main()


