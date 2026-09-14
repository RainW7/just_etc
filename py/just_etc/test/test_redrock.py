import numpy as np
from astropy.io import fits

from just_etc import write_redrock_mock_spectrum


def _mock_arm(arm):
    wave = np.array([4000.0, 4001.0]) + 2000.0 * arm
    intrinsic = np.array([2.0e-17, 4.0e-17])
    return {
        "arm": arm,
        "wave_aa": wave,
        "flux_intrinsic": intrinsic,
        "flux_mock": np.array([3.0e-17, 5.0e-17]),
        "signal_e": np.array([10.0, 20.0]),
        "noise_e": np.array([5.0, 10.0]),
        "snr": np.array([2.0, 2.0]),
    }


def test_write_redrock_mock_spectrum(tmp_path):
    output = tmp_path / "mock.fits"
    result = write_redrock_mock_spectrum(
        output,
        [_mock_arm(arm) for arm in range(3)],
        target_id=123,
        ra=12.5,
        dec=-1.5,
        primary_metadata={"SIMZ": (0.1, "Input redshift")},
    )

    assert result == output
    with fits.open(output, checksum=True) as hdul:
        assert [hdu.name for hdu in hdul] == [
            "PRIMARY",
            "FIBERMAP",
            "B_WAVELENGTH",
            "B_FLUX",
            "B_IVAR",
            "B_RESOLUTION",
            "R_WAVELENGTH",
            "R_FLUX",
            "R_IVAR",
            "R_RESOLUTION",
            "Z_WAVELENGTH",
            "Z_FLUX",
            "Z_IVAR",
            "Z_RESOLUTION",
            "SCORES",
        ]
        assert hdul[0].header["SIMZ"] == 0.1
        assert hdul["FIBERMAP"].data["TARGETID"][0] == 123
        np.testing.assert_allclose(hdul["B_FLUX"].data, [[3.0, 5.0]])
        np.testing.assert_allclose(hdul["B_IVAR"].data, [[1.0, 0.25]])
        assert hdul["B_RESOLUTION"].data.shape == (1, 1, 2)
        assert all(hdu.verify_checksum() == 1 for hdu in hdul)


def test_write_redrock_mock_spectrum_rejects_wrong_arm_count(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "expected 3 spectrograph arms"):
        write_redrock_mock_spectrum(tmp_path / "bad.fits", [_mock_arm(0)])
