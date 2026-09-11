from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from just_etc import JUSTExposureTimeCalculator, list_templates, load_template, normalize_to_mag
from just_etc import ETC_py_optimized as engine
from just_etc.resources import DATA_DIR


def test_installed_resources_from_other_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert engine.MODEL_DATA_AVAILABLE
    assert len(list_templates()) == 40
    for name in list_templates():
        wave, flux = load_template(name)
        assert wave.shape == flux.shape
        assert np.all(np.diff(wave) >= 0)
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    assert etc._spectro.N_arms == 3
    assert etc.spec_file == DATA_DIR / 'spec.dat'


def test_custom_template_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = DATA_DIR / 'templates/galaxy/elliptical_001.fits'
    local = tmp_path / 'local.fits'
    local.write_bytes(source.read_bytes())
    a = load_template('local.fits')
    b = load_template('templates/galaxy/elliptical_001.fits')
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    with pytest.raises(FileNotFoundError):
        load_template(tmp_path / 'missing.fits')


def test_magnitude_scaling():
    wave = np.linspace(3000., 10000., 200)
    flux = np.ones_like(wave) * 1e-17
    bright, _ = normalize_to_mag(wave, flux, 20., 'r')
    faint, _ = normalize_to_mag(wave, flux, 22.5, 'r')
    np.testing.assert_allclose(bright / faint, 10.)


def small_calculator():
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    # Exercise physical kernels on a small native grid, not a mocked engine.
    etc._spectro.npix[:] = 64
    return etc


def test_exposure_scaling_and_seeded_noise():
    wave = np.linspace(3000., 10000., 200)
    flux = np.ones_like(wave) * 1e-17
    etc = small_calculator()
    etc.set_obs_conditions(sysfrac=0.)
    one = etc.compute_snr(wave, flux, t_exp=900, n_exp=1)
    four = etc.compute_snr(wave, flux, t_exp=900, n_exp=4)
    for a, b in zip(one, four):
        assert np.all(np.isfinite(b['snr']))
        assert np.all(b['noise_var'] > 0)
        np.testing.assert_allclose(b['snr'], 2 * a['snr'], rtol=1e-12)
    a = etc.simulate_mock_observation(wave, flux, 900, seed=42)
    b = etc.simulate_mock_observation(wave, flux, 900, seed=42)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x['flux_mock'], y['flux_mock'])


def test_cli_imported_entrypoint(tmp_path, monkeypatch, capsys):
    from just_etc import cli
    # Exercise imported main(), including its NumPy usage, on the small grid.
    monkeypatch.setattr(cli, 'JUSTExposureTimeCalculator', lambda **kwargs: small_calculator())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['just-etc', '--texp', '900', '--nexp', '1'])
    cli.main()
    assert 'Mean SNR' in capsys.readouterr().out


def test_cli_help_from_other_directory(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'just_etc.cli', '--help'],
                            cwd=str(tmp_path), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    assert result.returncode == 0, result.stderr
    assert '--target-snr' in result.stdout


def test_config_example_matches_installed_default():
    root = Path(__file__).resolve().parents[3]
    example = root / 'etc/spec.dat'
    if example.exists():
        assert example.read_bytes() == (DATA_DIR / 'spec.dat').read_bytes()
