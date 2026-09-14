import numpy as np
from astropy.io import fits

from just_etc import convet_to_redrock_format
from just_etc import spectral_library


class _FakeCalculator:
    seen_fluxes = []

    def __init__(self, calc_mode):
        self.calc_mode = calc_mode

    def set_obs_conditions(self, **kwargs):
        self.conditions = kwargs

    def simulate_mock_observation(self, wave, flux, t_exp, n_exp, seed):
        self.seen_fluxes.append(np.array(flux, copy=True))
        result = []
        for arm in range(3):
            intrinsic = np.asarray(flux, dtype=np.float64)
            result.append({
                "arm": arm,
                "wave_aa": np.asarray(wave, dtype=np.float64),
                "flux_intrinsic": intrinsic,
                "flux_mock": intrinsic * 1.25,
                "signal_e": np.full(intrinsic.shape, 4.0),
                "noise_e": np.full(intrinsic.shape, 2.0),
                "snr": np.full(intrinsic.shape, 2.0),
            })
        return result


def test_convet_to_redrock_format_writes_spectral_library(tmp_path, monkeypatch):
    _FakeCalculator.seen_fluxes = []
    monkeypatch.setattr(spectral_library, "JUSTExposureTimeCalculator", _FakeCalculator)
    wave = np.array([4000.0, 5000.0, 6000.0])
    flux = np.array([
        [1.0, 2.0, 3.0],
        [np.nan, -1.0, 0.5],
    ])
    input_copy = flux.copy()
    output = tmp_path / "library_redrock.fits"

    result = convet_to_redrock_format(
        wave,
        flux,
        output,
        t_exp=120.0,
        n_exp=2,
        nproc=1,
        target_ids=[77, 88],
    )

    assert result == output
    np.testing.assert_array_equal(flux, input_copy)
    np.testing.assert_allclose(
        _FakeCalculator.seen_fluxes[0], [1.0e-17, 2.0e-17, 3.0e-17]
    )
    np.testing.assert_allclose(
        _FakeCalculator.seen_fluxes[1], [0.0, 0.0, 0.5e-17]
    )

    with fits.open(output, checksum=True) as hdul:
        assert hdul["FIBERMAP"].data["TARGETID"].tolist() == [77, 88]
        assert hdul["B_WAVELENGTH"].data.shape == (3,)
        assert hdul["B_FLUX"].data.shape == (2, 3)
        assert hdul["B_IVAR"].data.shape == (2, 3)
        assert hdul["B_RESOLUTION"].data.shape == (2, 1, 3)
        np.testing.assert_allclose(hdul["B_FLUX"].data[0], [1.25, 2.5, 3.75])
        assert np.all(np.isfinite(hdul["B_IVAR"].data))
        assert hdul[0].header["EXPTIME"] == 120.0
        assert hdul[0].header["NEXP"] == 2
        assert all(hdu.verify_checksum() == 1 for hdu in hdul)


def test_convet_to_redrock_format_accepts_single_spectrum(tmp_path, monkeypatch):
    monkeypatch.setattr(spectral_library, "JUSTExposureTimeCalculator", _FakeCalculator)
    wave = np.array([4000.0, 5000.0, 6000.0])
    output = tmp_path / "single_redrock.fits"

    convet_to_redrock_format(
        wave,
        np.ones(wave.size),
        output,
        nproc=1,
        target_ids=[9],
    )

    with fits.open(output) as hdul:
        assert hdul["B_FLUX"].data.shape == (1, wave.size)
        assert hdul["SCORES"].data["TARGETID"].tolist() == [9]


def test_convet_to_redrock_format_validates_input_shapes(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "strictly increasing"):
        convet_to_redrock_format(
            [4000.0, 3999.0],
            [1.0, 2.0],
            tmp_path / "invalid.fits",
            nproc=1,
        )

    with np.testing.assert_raises_regex(ValueError, "flux must have shape"):
        convet_to_redrock_format(
            [4000.0, 5000.0],
            [[1.0, 2.0, 3.0]],
            tmp_path / "invalid.fits",
            nproc=1,
        )
