import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path('/Users/rain/JUST_ETC/ETC_py_v1').resolve()))
import ETC_py_optimized as ETC_py
from just_etc_api import JUSTExposureTimeCalculator

etc = JUSTExposureTimeCalculator(calc_mode='fast')
sp = etc._spectro
de = etc._obs_params['decenter']
fa = etc._obs_params['field_angle']

seeing_values = [0.6, 0.8, 1.2, 1.5]
lam = 600.0  # Calculate at 600 nm

print(f"Fiber radius: {sp.fiber_ent_rad} arcsec (wait, let's check fiber_ent_rad units - actually the config loads it in arcsec or microns?)")
# Actually the config spec.dat has fiber_ent_rad in microns. Let's just calculate the throughput directly.

for seeing in seeing_values:
    etc.set_obs_conditions(seeing_fwhm_800=seeing)
    obs = etc._make_obs()
    eff = ETC_py.gsGeometricThroughput(sp, obs, lam, 0.0, de, fa, 0x0)
    print(f"Seeing {seeing}\": {eff * 100:.2f}%")
