"""Explicit spectral axes and response-weighted interpolation without extrapolation."""
import numpy as np

def wavelength_axis(values, *, unit):
    wave = np.asarray(values, dtype=float)
    if unit not in {"nm", "um"}:
        raise ValueError("Wavelength unit must be declared as nm or um")
    wave = wave * (1000 if unit == "um" else 1)
    if wave.ndim != 1 or len(wave) < 2 or not np.isfinite(wave).all() or (wave <= 0).any():
        raise ValueError("Invalid wavelength axis")
    if not (np.diff(wave) > 0).all():
        raise ValueError("Wavelengths must be strictly increasing and unique")
    return wave

def response_weighted_reflectance(wave_nm, reflectance, response_nm, response, *, maximum_gap_nm):
    wave = wavelength_axis(wave_nm, unit="nm")
    grid = wavelength_axis(response_nm, unit="nm")
    values, weights = np.asarray(reflectance, float), np.asarray(response, float)
    if values.shape != wave.shape or weights.shape != grid.shape:
        raise ValueError("Spectral axes and values do not align")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Nonfinite spectrum or invalid response")
    if not np.isfinite(maximum_gap_nm) or maximum_gap_nm <= 0:
        raise ValueError("Declare maximum interpolation gap")
    if grid[0] < wave[0] or grid[-1] > wave[-1]:
        raise ValueError("Spectrum does not cover the response domain")
    if (np.diff(wave)[(wave[:-1] < grid[-1]) & (wave[1:] > grid[0])] > maximum_gap_nm).any():
        raise ValueError("Spectral sampling gap exceeds policy")
    # Integrate on the union of measured and response grids.
    nodes = np.union1d(grid, wave[(wave > grid[0]) & (wave < grid[-1])])
    w = np.interp(nodes, grid, weights)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    area = trapezoid(w, nodes)
    if area <= 0:
        raise ValueError("Response has zero integrated weight")
    interpolated = np.interp(nodes, wave, values)
    # The product of two piecewise-linear functions is quadratic; integrate exactly per segment.
    left, right = interpolated[:-1], interpolated[1:]
    a, b = w[:-1], w[1:]
    integral = np.sum(np.diff(nodes) * (2*left*a + left*b + right*a + 2*right*b)/6)
    return float(integral / area)
