# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Provide stationary psychoacoustic metrics and their Acoular integration.

.. autosummary::
    :toctree: generated/

    loudness_stationary
    sharpness_stationary
    resample_to_48k
    LoudnessStationary
    SharpnessStationary

The implementations follow DIN ISO 532-1 and DIN 45692
:cite:`DINISO532-1,DIN45692`.
"""

from math import gcd

from .base import SamplesGenerator
from .internal import digest

import numba as nb
import numpy as np
from scipy.signal import resample_poly
from traits.api import (
    CInt,
    Enum,
    Float,
    HasStrictTraits,
    Instance,
    Int,
    Property,
    Union,
    cached_property,
)

# ISO 532-1:2017(E) Loudness (stationary Zwicker method)
# ============================================================
# Reference Table A.1
# Coefficients per stage: [b0, b1, b2, a0, a1, a2]
# ============================================================

REFERENCE_SOS = np.array(
    [
        [1.0, 2.0, 1.0, 1.0, -2.0, 1.0],  # stage 1
        [1.0, 0.0, -1.0, 1.0, -2.0, 1.0],  # stage 2
        [1.0, -2.0, 1.0, 1.0, -2.0, 1.0],  # stage 3
    ],
    dtype=np.float64,
)


# ============================================================
# one-third octave band center freqs
# ============================================================

CENTER_FREQUENCIES = np.array(
    [
        25.0,
        31.5,
        40.0,
        50.0,
        63.0,
        80.0,
        100.0,
        125.0,
        160.0,
        200.0,
        250.0,
        315.0,
        400.0,
        500.0,
        630.0,
        800.0,
        1000.0,
        1250.0,
        1600.0,
        2000.0,
        2500.0,
        3150.0,
        4000.0,
        5000.0,
        6300.0,
        8000.0,
        10000.0,
        12500.0,
    ],
    dtype=np.float64,
)


# ============================================================
# band coefficient corrections and gains from Table A.2
# delta = deviation relative to REFERENCE_SOS
# actual SOS = REFERENCE_SOS - delta
# ============================================================

BAND_SPECS = {
    25.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -6.70260e-4, 6.59453e-4],
                [0.0, 0.0, 0.0, 0.0, -3.75071e-4, 3.61926e-4],
                [0.0, 0.0, 0.0, 0.0, -3.06523e-4, 2.97634e-4],
            ],
            dtype=np.float64,
        ),
        'gain': 4.30764e-11,
    },
    31.5: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -8.47258e-4, 8.30131e-4],
                [0.0, 0.0, 0.0, 0.0, -4.76448e-4, 4.55616e-4],
                [0.0, 0.0, 0.0, 0.0, -3.88773e-4, 3.74685e-4],
            ],
            dtype=np.float64,
        ),
        'gain': 8.59340e-11,
    },
    40.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -1.07210e-3, 1.04496e-3],
                [0.0, 0.0, 0.0, 0.0, -6.06567e-4, 5.73553e-4],
                [0.0, 0.0, 0.0, 0.0, -4.94004e-4, 4.71677e-4],
            ],
            dtype=np.float64,
        ),
        'gain': 1.71424e-10,
    },
    50.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -1.35836e-3, 1.31535e-3],
                [0.0, 0.0, 0.0, 0.0, -7.74327e-4, 7.22007e-4],
                [0.0, 0.0, 0.0, 0.0, -6.29154e-4, 5.93771e-4],
            ],
            dtype=np.float64,
        ),
        'gain': 3.41944e-10,
    },
    63.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -1.72380e-3, 1.65564e-3],
                [0.0, 0.0, 0.0, 0.0, -9.91780e-4, 9.08866e-4],
                [0.0, 0.0, 0.0, 0.0, -8.03529e-4, 7.47455e-4],
            ],
            dtype=np.float64,
        ),
        'gain': 6.82035e-10,
    },
    80.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -2.19188e-3, 2.08388e-3],
                [0.0, 0.0, 0.0, 0.0, -1.27545e-3, 1.14406e-3],
                [0.0, 0.0, 0.0, 0.0, -1.02976e-3, 9.40900e-4],
            ],
            dtype=np.float64,
        ),
        'gain': 1.36026e-9,
    },
    100.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -2.79386e-3, 2.62274e-3],
                [0.0, 0.0, 0.0, 0.0, -1.64828e-3, 1.44006e-3],
                [0.0, 0.0, 0.0, 0.0, -1.32520e-3, 1.18438e-3],
            ],
            dtype=np.float64,
        ),
        'gain': 2.71261e-9,
    },
    125.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -3.57182e-3, 3.30071e-3],
                [0.0, 0.0, 0.0, 0.0, -2.14252e-3, 1.81258e-3],
                [0.0, 0.0, 0.0, 0.0, -1.71397e-3, 1.49082e-3],
            ],
            dtype=np.float64,
        ),
        'gain': 5.40870e-9,
    },
    160.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -4.58305e-3, 4.15355e-3],
                [0.0, 0.0, 0.0, 0.0, -2.80413e-3, 2.28135e-3],
                [0.0, 0.0, 0.0, 0.0, -2.23006e-3, 1.87646e-3],
            ],
            dtype=np.float64,
        ),
        'gain': 1.07826e-8,
    },
    200.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -5.90655e-3, 5.22622e-3],
                [0.0, 0.0, 0.0, 0.0, -3.69947e-3, 2.87118e-3],
                [0.0, 0.0, 0.0, 0.0, -2.92205e-3, 2.36178e-3],
            ],
            dtype=np.float64,
        ),
        'gain': 2.14910e-8,
    },
    250.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -7.65243e-3, 6.57493e-3],
                [0.0, 0.0, 0.0, 0.0, -4.92540e-3, 3.61318e-3],
                [0.0, 0.0, 0.0, 0.0, -3.86007e-3, 2.97240e-3],
            ],
            dtype=np.float64,
        ),
        'gain': 4.28228e-8,
    },
    315.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -1.00023e-2, 8.29610e-3],
                [0.0, 0.0, 0.0, 0.0, -6.63788e-3, 4.55999e-3],
                [0.0, 0.0, 0.0, 0.0, -5.15982e-3, 3.75306e-3],
            ],
            dtype=np.float64,
        ),
        'gain': 8.54316e-8,
    },
    400.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -1.31230e-2, 1.04220e-2],
                [0.0, 0.0, 0.0, 0.0, -9.02274e-3, 5.73132e-3],
                [0.0, 0.0, 0.0, 0.0, -6.94543e-3, 4.71734e-3],
            ],
            dtype=np.float64,
        ),
        'gain': 1.70009e-7,
    },
    500.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -1.73693e-2, 1.30947e-2],
                [0.0, 0.0, 0.0, 0.0, -1.24176e-2, 7.20526e-3],
                [0.0, 0.0, 0.0, 0.0, -9.46002e-3, 5.93145e-3],
            ],
            dtype=np.float64,
        ),
        'gain': 3.38215e-7,
    },
    630.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -2.31934e-2, 1.64308e-2],
                [0.0, 0.0, 0.0, 0.0, -1.73009e-2, 9.04761e-3],
                [0.0, 0.0, 0.0, 0.0, -1.30358e-2, 7.44926e-3],
            ],
            dtype=np.float64,
        ),
        'gain': 6.71990e-7,
    },
    800.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -3.13292e-2, 2.06370e-2],
                [0.0, 0.0, 0.0, 0.0, -2.44342e-2, 1.13731e-2],
                [0.0, 0.0, 0.0, 0.0, -1.82108e-2, 9.36778e-3],
            ],
            dtype=np.float64,
        ),
        'gain': 1.33531e-6,
    },
    1000.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -4.28261e-2, 2.59325e-2],
                [0.0, 0.0, 0.0, 0.0, -3.49619e-2, 1.43046e-2],
                [0.0, 0.0, 0.0, 0.0, -2.57855e-2, 1.17912e-2],
            ],
            dtype=np.float64,
        ),
        'gain': 2.65172e-6,
    },
    1250.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -5.91733e-2, 3.25054e-2],
                [0.0, 0.0, 0.0, 0.0, -5.06072e-2, 1.79513e-2],
                [0.0, 0.0, 0.0, 0.0, -3.69401e-2, 1.48094e-2],
            ],
            dtype=np.float64,
        ),
        'gain': 5.25477e-6,
    },
    1600.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -8.26348e-2, 4.05894e-2],
                [0.0, 0.0, 0.0, 0.0, -7.40348e-2, 2.24476e-2],
                [0.0, 0.0, 0.0, 0.0, -5.34977e-2, 1.85371e-2],
            ],
            dtype=np.float64,
        ),
        'gain': 1.03780e-5,
    },
    2000.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -1.17018e-1, 5.08116e-2],
                [0.0, 0.0, 0.0, 0.0, -1.09516e-1, 2.81387e-2],
                [0.0, 0.0, 0.0, 0.0, -7.85097e-2, 2.32872e-2],
            ],
            dtype=np.float64,
        ),
        'gain': 2.04870e-5,
    },
    2500.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -1.67714e-1, 6.37872e-2],
                [0.0, 0.0, 0.0, 0.0, -1.63378e-1, 3.53729e-2],
                [0.0, 0.0, 0.0, 0.0, -1.16419e-1, 2.93723e-2],
            ],
            dtype=np.float64,
        ),
        'gain': 4.05198e-5,
    },
    3150.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -2.42528e-1, 7.98576e-2],
                [0.0, 0.0, 0.0, 0.0, -2.45161e-1, 4.43370e-2],
                [0.0, 0.0, 0.0, 0.0, -1.73972e-1, 3.70015e-2],
            ],
            dtype=np.float64,
        ),
        'gain': 7.97914e-5,
    },
    4000.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -3.53142e-1, 9.96330e-2],
                [0.0, 0.0, 0.0, 0.0, -3.69163e-1, 5.53535e-2],
                [0.0, 0.0, 0.0, 0.0, -2.61399e-1, 4.65428e-2],
            ],
            dtype=np.float64,
        ),
        'gain': 1.56511e-4,
    },
    5000.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -5.16316e-1, 1.24177e-1],
                [0.0, 0.0, 0.0, 0.0, -5.55473e-1, 6.89403e-2],
                [0.0, 0.0, 0.0, 0.0, -3.93998e-1, 5.86715e-2],
            ],
            dtype=np.float64,
        ),
        'gain': 3.04954e-4,
    },
    6300.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -7.56635e-1, 1.55023e-1],
                [0.0, 0.0, 0.0, 0.0, -8.34281e-1, 8.58123e-2],
                [0.0, 0.0, 0.0, 0.0, -5.94547e-1, 7.43960e-2],
            ],
            dtype=np.float64,
        ),
        'gain': 5.99157e-4,
    },
    8000.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -1.10165e0, 1.91713e-1],
                [0.0, 0.0, 0.0, 0.0, -1.23939e0, 1.05243e-1],
                [0.0, 0.0, 0.0, 0.0, -8.91666e-1, 9.40354e-2],
            ],
            dtype=np.float64,
        ),
        'gain': 1.16544e-3,
    },
    10000.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -1.58477e0, 2.39049e-1],
                [0.0, 0.0, 0.0, 0.0, -1.80505e0, 1.28794e-1],
                [0.0, 0.0, 0.0, 0.0, -1.32500e0, 1.21333e-1],
            ],
            dtype=np.float64,
        ),
        'gain': 2.27488e-3,
    },
    12500.0: {
        'delta': np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -2.50630e0, 1.42308e-1],
                [0.0, 0.0, 0.0, 0.0, -2.19464e0, 2.76470e-1],
                [0.0, 0.0, 0.0, 0.0, -1.90231e0, 1.47304e-1],
            ],
            dtype=np.float64,
        ),
        'gain': 3.91006e-3,
    },
}


# ============================================================
# Filter bank as described in Section 4, implemented as
# 6th-order Chebyshev filters using three second-order
# sections (SOS)
# ============================================================
def build_sos_and_gains(center_frequencies, band_specs, reference_sos):
    """Calculate filter-bank coefficients and gains from the reference data."""
    n_bands = len(center_frequencies)
    n_stages = reference_sos.shape[0]

    sos_bank = np.empty((n_bands, n_stages, 6), dtype=np.float64)
    gains = np.empty(n_bands, dtype=np.float64)

    for i, fc in enumerate(center_frequencies):
        if fc not in band_specs:
            msg = f'band {fc} Hz missing in BAND_SPECS'
            raise KeyError(msg)

        delta = band_specs[fc]['delta']
        gain = band_specs[fc]['gain']

        if delta.shape != reference_sos.shape:
            msg = f'band {fc} Hz: delta has shape {delta.shape}, expected {reference_sos.shape}'
            raise ValueError(msg)

        sos_bank[i] = reference_sos - delta
        gains[i] = gain

    return sos_bank, gains


@nb.njit(cache=True)
def biquad_filter(x, coeffs):
    """Process an input signal through a Direct Form II biquad filter."""
    b0, b1, b2, a0, a1, a2 = coeffs

    z1 = 0.0
    z2 = 0.0

    if a0 != 1.0:
        b0 = b0 / a0
        b1 = b1 / a0
        b2 = b2 / a0
        a1 = a1 / a0
        a2 = a2 / a0

    y = np.empty_like(x, dtype=np.float64)

    for n in range(x.shape[0]):
        w = x[n] - a1 * z1 - a2 * z2
        y[n] = b0 * w + b1 * z1 + b2 * z2
        z2 = z1
        z1 = w

    return y


@nb.njit(cache=True)
def sos_filter(x, sos, gain=1.0):
    """Filter an input signal through cascaded second-order sections."""
    y = x * gain
    n_stages = sos.shape[0]

    for stage in range(n_stages):
        y = biquad_filter(y, sos[stage])

    return y


# built filter bank SOS coefficients and gains
SOS_BANK, GAINS = build_sos_and_gains(
    center_frequencies=CENTER_FREQUENCIES, band_specs=BAND_SPECS, reference_sos=REFERENCE_SOS
)


@nb.njit(cache=True)
def filterbank(x, sos_bank=SOS_BANK, gains=GAINS):
    """Decompose a signal by applying the full SOS filter bank and gains."""
    n_bands = sos_bank.shape[0]
    n_samples = x.shape[0]
    y = np.empty((n_bands, n_samples), dtype=np.float64)

    for band in range(n_bands):
        y_band = sos_filter(x=x, sos=sos_bank[band], gain=gains[band])

        y[band] = y_band

    return y


@nb.njit(cache=True)
def third_octave_levels(x, analysis_start=0, sos_bank=SOS_BANK, gains=GAINS):
    """Calculate third-octave levels without retaining filtered signals."""
    levels = np.empty(sos_bank.shape[0], dtype=np.float64)
    reference_pressure_squared = (20e-6) ** 2

    for band in range(sos_bank.shape[0]):
        y_band = sos_filter(x, sos_bank[band], gains[band])
        # Earlier samples are filtered to settle the filter state, but they do
        # not contribute to the stationary level average.
        mean_square = np.mean(y_band[analysis_start:] ** 2)
        if mean_square == 0.0:
            levels[band] = -np.inf
        else:
            levels[band] = 10.0 * np.log10(mean_square / reference_pressure_squared)

    return levels


def third_octave_levels_multichannel(x, analysis_start=0):
    """Calculate third-octave levels for sample-major multichannel data.

    The channel loop avoids allocating all band-filtered time signals for all
    channels at once. Only the ``(28, n_channels)`` level matrix is retained.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        msg = f'expected shape (n_samples, n_channels), got {x.shape}'
        raise ValueError(msg)

    levels = np.empty((SOS_BANK.shape[0], x.shape[1]), dtype=np.float64)
    for channel in range(x.shape[1]):
        levels[:, channel] = third_octave_levels(x[:, channel], analysis_start)
    return levels


# ============================================================
# Low-frequency weighting from Table A.3 to account for reduced
# hearing sensitivity at low frequencies
# Applies to 1/3-octave bands with center frequencies < 300 Hz
# L_T_weighted = L_T + delta_L (as per ISO 532-1, Section 5.3)
# ============================================================

# table A.3
DELTA_L = np.array(
    [
        [-32, -24, -16, -10, -5, 0, -7, -3, 0, -2, 0],  # 45
        [-29, -22, -15, -10, -4, 0, -7, -2, 0, -2, 0],  # 55
        [-27, -19, -14, -9, -4, 0, -6, -2, 0, -2, 0],  # 65
        [-25, -17, -12, -9, -3, 0, -5, -2, 0, -2, 0],  # 71
        [-23, -16, -11, -7, -3, 0, -4, -1, 0, -1, 0],  # 80
        [-20, -14, -10, -6, -3, 0, -4, -1, 0, -1, 0],  # 90
        [-18, -12, -9, -6, -2, 0, -3, -1, 0, -1, 0],  # 100
        [-15, -10, -8, -4, -2, 0, -3, -1, 0, -1, 0],  # 120
    ],
    dtype=np.float64,
)

# BANDS = np.array([25, 32, 40, 50, 63, 80, 100, 125, 160, 200, 250])
THRESHOLDS = np.array([45, 55, 65, 71, 80, 90, 100, 120], dtype=np.float64)


def get_correction_1d(lp):
    """Return low-frequency level corrections for one channel."""
    if lp.ndim != 1:
        msg = 'expected 1D input'
        raise ValueError(msg)
    if lp.shape[0] != 11:
        msg = 'expected shape (11,)'
        raise ValueError(msg)

    out = np.empty_like(lp)

    rap_minus_dll = THRESHOLDS[:, None] - DELTA_L  # shape (8, 11)

    for i in range(11):
        k = 0
        while k < 7 and lp[i] > rap_minus_dll[k, i]:
            k += 1
        out[i] = DELTA_L[k, i]

    return out


def apply_correction(lp):
    """Apply low-frequency level corrections to one or more channels."""
    lp = np.asarray(lp, dtype=np.float64)
    if lp.ndim == 1:
        return lp + get_correction_1d(lp)
    if lp.ndim == 2:
        return np.stack([lp[:, t] + get_correction_1d(lp[:, t]) for t in range(lp.shape[1])], axis=1)
    msg = 'expected 1D or 2D input'
    raise ValueError(msg)


# Apply field correction and calculate core loudness according to Section 5.2, step 3.

# ============================================================
# Field corrections in dB are applied to critical bands
# depending on sound field and proximity to the hearing threshold.
# Then core loudness per critical band is computed according
# to formula A.2.
# ============================================================

# table A.4 - level correction for outer and middle ear in free-field (HRTF approx.?)
A0 = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -1.6, -3.2, -5.4, -5.6, -4.0, -1.5, 2, 5.0, 12.0],
    dtype=float,
)

# table A.5 - level deviation for diffuse-field
DELTA_L_DF = np.array(
    [0.0, 0.0, 0.5, 0.9, 1.2, 1.6, 2.3, 2.8, 3.0, 2.0, 0.0, -1.4, -2.0, -1.9, -1.0, 0.5, 3.0, 4.0, 4.3, 4.0],
    dtype=float,
)

# table A.6 - cb-levels close to hearing threshold (without outer and middle ear)
LTQ = np.array(
    [30.0, 18.0, 12.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=float
)

# table A.7 - correction of cb levels to account for different bandwidth of center freqs
DELTA_LCB = np.array(
    [
        -0.25,
        -0.60,
        -0.80,
        -0.80,
        -0.50,
        0.00,
        0.50,
        1.10,
        1.50,
        1.70,
        1.80,
        1.80,
        1.70,
        1.60,
        1.40,
        1.20,
        0.80,
        0.50,
        0.00,
        -0.50,
    ],
    dtype=float,
)


def core_loudness_from_bands(critical_bands_raw, diffuse=False):
    """Apply field corrections and compute core loudness per critical band.

    Accepts:
        (20,)   -> returns (20,)
        (20, n) -> returns (20, n)
    """
    x = np.asarray(critical_bands_raw, dtype=np.float64)

    if x.ndim == 1:
        if x.shape[0] != 20:
            msg = 'expected shape (20,)'
            raise ValueError(msg)
    elif x.ndim == 2:
        if x.shape[0] != 20:
            msg = 'expected shape (20, n)'
            raise ValueError(msg)
    else:
        msg = 'expected 1D or 2D input'
        raise ValueError(msg)

    s = 0.25

    le = x - A0[:, None] if x.ndim == 2 else x - A0

    if diffuse:
        le = le + (DELTA_L_DF[:, None] if x.ndim == 2 else DELTA_L_DF)

    ltq = LTQ[:, None] if x.ndim == 2 else LTQ
    dlcb = DELTA_LCB[:, None] if x.ndim == 2 else DELTA_LCB

    nm = np.zeros_like(le)
    mask = le > ltq

    le_corr = le - dlcb

    nm_val = 0.0635 * 10.0 ** (0.025 * ltq) * ((1.0 - s + s * 10.0 ** (0.1 * (le_corr - ltq))) ** 0.25 - 1.0)

    nm[mask] = np.maximum(nm_val[mask], 0.0)
    return nm


def tau_from_center_frequency(f_t: float) -> float:
    """Return the time constant specified by the standard."""
    if f_t <= 1000.0:
        return (2.0 / 3.0) * (1.0 / f_t)
    return (2.0 / 3.0) * (1.0 / 1000.0)


def lowpass1_coeffs(fs: float, tau: float) -> tuple[float, float]:
    """Return the coefficients of a first-order low-pass filter."""
    a1 = np.exp(-1.0 / (fs * tau))
    b0 = 1.0 - a1
    return b0, a1


def lowpass1(x: np.ndarray, b0: float, a1: float, y_prev: float = 0.0):
    """Apply the first-order low-pass filter ``y[n] = b0*x[n] + a1*y[n-1]``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)

    prev = float(y_prev)
    for n in range(x.shape[0]):
        prev = b0 * x[n] + a1 * prev
        y[n] = prev

    return y, prev


def smooth_specific_band(x_band: np.ndarray, fs: float, f_t: float, y_prev=None):
    """Apply three identical first-order low-pass filters in series.

    Parameters
    ----------
    x_band : ndarray, shape (n_samples,)
        Output of a one-third-octave band filter.
    fs : float
        Sampling frequency.
    f_t : float
        Center frequency of the band.
    y_prev : ndarray shape (3,), optional
        States of the three low-pass filter stages.

    Returns
    -------
    y : ndarray
        Smoothed signal.
    state : ndarray, shape (3,)
        Updated filter states.
    """
    if y_prev is None:
        y_prev = np.zeros(3, dtype=np.float64)

    tau = tau_from_center_frequency(f_t)
    b0, a1 = lowpass1_coeffs(fs, tau)

    xq = np.asarray(x_band, dtype=np.float64)

    y1, s1 = lowpass1(xq, b0, a1, y_prev[0])
    y2, s2 = lowpass1(y1, b0, a1, y_prev[1])
    y3, s3 = lowpass1(y2, b0, a1, y_prev[2])

    state = np.array([s1, s2, s3], dtype=np.float64)
    return y3, state


def smooth_filterbank_outputs(y_bands: np.ndarray, fs: float, center_frequencies: np.ndarray, state=None):
    """y_bands: shape (n_bands, n_samples)."""
    y_bands = np.asarray(y_bands, dtype=np.float64)
    center_frequencies = np.asarray(center_frequencies, dtype=np.float64)

    n_bands, n_samples = y_bands.shape

    if state is None:
        state = np.zeros((n_bands, 3), dtype=np.float64)

    out = np.empty((n_bands, n_samples), dtype=np.float64)
    new_state = np.empty_like(state)

    for i in range(n_bands):
        out[i], new_state[i] = smooth_specific_band(y_bands[i], fs, center_frequencies[i], state[i])

    return out, new_state


###########################################################################
#
#  Builds the specific-loudness pattern from core loudness values.
#
#
#  Algorithm:
#  - step through the 21 Bark sections
#  - use the core loudness as plateau where possible
#  - otherwise apply ISO-defined falling slopes (table A.9)
#  - sample the resulting curve on a 0.1-Bark grid
#  - integrate the area under the curve to obtain total loudness
#
#   Comment: As this is apparently the trickiest part of the loudness
#   calculation, it is commented a little more detailed. Inspiration was
#   also drawn from the reference C implementation, SQAT, and MOSQITO.
#
#########################################################################


# table A.9, contains the upper Bark limits of the 21 sections.
# The small offset (+1e-4) helps avoid boundary issues.
ZUP = (
    np.array(
        [
            0.9,
            1.8,
            2.8,
            3.5,
            4.4,
            5.4,
            6.6,
            7.9,
            9.2,
            10.6,
            12.3,
            13.8,
            15.2,
            16.7,
            18.1,
            19.3,
            20.6,
            21.8,
            22.7,
            23.6,
            24.0,
        ],
        dtype=float,
    )
    + 1e-4
)  # as in the ISO reference implementation

# RNS contains reference specific-loudness values.
# These are used to select the correct slope region.
RNS = np.array(
    [21.5, 18.0, 15.1, 11.5, 9.0, 6.1, 4.4, 3.1, 2.13, 1.36, 0.82, 0.42, 0.30, 0.22, 0.15, 0.10, 0.035, 0.0],
    dtype=float,
)

# USL contains the ISO-defined upper slope values.
# Rows correspond to loudness regions selected via RNS.
# Columns correspond to frequency-group-dependent slope classes.
USL = np.array(
    [
        [13.0, 8.2, 6.3, 5.5, 5.5, 5.5, 5.5, 5.5],
        [9.0, 7.5, 6.0, 5.1, 4.5, 4.5, 4.5, 4.5],
        [7.8, 6.7, 5.6, 4.9, 4.4, 3.9, 3.9, 3.9],
        [6.2, 5.4, 4.6, 4.0, 3.5, 3.2, 3.2, 3.2],
        [4.5, 3.8, 3.6, 3.2, 2.9, 2.7, 2.7, 2.7],
        [3.7, 3.0, 2.8, 2.35, 2.2, 2.2, 2.2, 2.2],
        [2.9, 2.3, 2.1, 1.9, 1.8, 1.7, 1.7, 1.7],
        [2.4, 1.7, 1.5, 1.35, 1.3, 1.3, 1.3, 1.3],
        [1.95, 1.45, 1.3, 1.15, 1.1, 1.1, 1.1, 1.1],
        [1.5, 1.2, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82],
        [0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62],
        [0.59, 0.53, 0.51, 0.50, 0.42, 0.42, 0.42, 0.42],
        [0.40, 0.33, 0.26, 0.24, 0.22, 0.22, 0.22, 0.22],
        [0.27, 0.21, 0.20, 0.18, 0.17, 0.17, 0.17, 0.17],
        [0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11],
        [0.12, 0.11, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08],
        [0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05],
        [0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02],
    ],
    dtype=float,
)


def _rns_index(x):
    """Return the first index whose RNS value is less than or equal to ``x``."""
    return min(np.count_nonzero(x < RNS), len(RNS) - 1)


# @njit(cache=True)
def _specific_loudness_1d_exact(core, dz=0.1):
    """Build the specific-loudness pattern from one core-loudness vector.

    Parameters
    ----------
    core : ndarray
        Core loudness values for 20 or 21 critical bands.
    dz : float, optional
        Bark step size for the output grid, by default 0.1.

    Returns
    -------
    z_grid : ndarray
        Bark axis from dz to 24 Bark.
    spec : ndarray
        Specific loudness values sampled on z_grid.
    loudness_sone : float
        Total loudness obtained as the area under the specific-loudness curve.
    """
    # If only 20 values are given, append a final zero band. Upper band is always zero after ISO.
    # This lets the loudness pattern decay properly toward the upper end.
    if core.shape[0] == 20:
        core21 = np.empty(21, dtype=core.dtype)
        core21[:20] = core
        core21[20] = 0.0
        core = core21
    elif core.shape[0] != 21:
        msg = f'expected 20 or 21 values, got {core.shape}'
        raise ValueError(msg)

    # Output Bark grid, usually 0.1, 0.2, ..., 24.0 Bark
    z_grid = np.arange(dz, 24.0 + 1e-12, dz)

    # Output specific-loudness profile on the Bark grid
    spec = np.zeros_like(z_grid)

    # Running integral = total loudness
    loudness_sone = 0.0

    # Current start point of the profile segment:
    # z1 = Bark position
    # n1 = specific loudness at that position
    z1 = 0.0
    n1 = 0.0

    # Indexing for writing onto the discrete Bark grid
    z = dz
    iz = 0

    # Start with the lowest RNS region
    j = len(RNS) - 1  # equivalent to MATLAB j=18

    # Process each of the 21 Bark sections
    for i in range(21):
        nm_i = core[i]  # Core loudness target for this Bark section
        # col = min(i, 7)

        # Select the slope column depending on the band index.
        # Low and high frequency groups use different slope classes.
        col = max(0, min(i - 1, 7))
        # col = 0 if i == 0 else min(i - 1, 7)

        # Continue until the current section reaches its upper Bark boundary
        while z1 < ZUP[i]:
            # Case 1:
            # Current profile is below or equal to the target core loudness.
            # Build a flat plateau at level nm_i.
            if n1 <= nm_i:
                if n1 < nm_i:
                    # Update the RNS class for the new higher level
                    j = _rns_index(nm_i)

                # Plateau extends to the upper boundary of the current Bark section
                z2 = ZUP[i]
                n2 = nm_i

                # Add rectangular area under the plateau
                loudness_sone += n2 * (z2 - z1)

                # Write plateau values onto the discrete Bark grid
                while iz < len(z_grid) and z <= z2:
                    spec[iz] = n2
                    iz += 1
                    z += dz

            # Case 2:
            # Current profile is above the target core loudness.
            # Reduce it gradually using the ISO-defined upper slope.
            else:
                # Do not drop below the current RNS level or below the target core loudness
                n2 = max(RNS[j], nm_i)
                slope = USL[j, col]  # Slope for this loudness region and band group

                z2 = z1 + (n1 - n2) / slope  # Bark position where the linearly falling slope reaches n2

                # If the slope would extend beyond the section boundary,
                # cut it at the boundary and recompute n2 accordingly.
                if z2 > ZUP[i]:
                    z2 = ZUP[i]
                    n2 = n1 - (z2 - z1) * slope

                # Add trapezoidal area under the falling segment
                loudness_sone += (z2 - z1) * (n1 + n2) / 2.0

                # Write falling values onto the discrete Bark grid
                while iz < len(z_grid) and z <= z2:
                    spec[iz] = n1 - (z - z1) * slope
                    iz += 1
                    z += dz

            # If the current level has dropped to or below the active RNS threshold,
            # move to the next lower loudness region.
            if n2 <= RNS[j] and j < len(RNS) - 1:
                j += 1

            # Continue from the end of the just-processed segment
            z1 = z2
            n1 = n2
    #  Final ISO-style rounding
    if loudness_sone < 0:
        loudness_sone = 0.0
    elif loudness_sone <= 16:
        loudness_sone = np.floor(loudness_sone * 1000 + 0.5) / 1000
    else:
        loudness_sone = np.floor(loudness_sone * 100 + 0.5) / 100

    return z_grid, spec, loudness_sone


# @njit(cache=True)
def _specific_loudness_from_core_2d_21(core, dz):
    """Apply the 1D algorithm to every column of a core-loudness matrix."""
    z, spec0, loudness0_sone = _specific_loudness_1d_exact(core[:, 0], dz)

    spec = np.empty((z.shape[0], core.shape[1]), dtype=np.float64)
    loudness_sone = np.empty(core.shape[1], dtype=np.float64)

    spec[:, 0] = spec0
    loudness_sone[0] = loudness0_sone

    for c in range(1, core.shape[1]):
        _, spec[:, c], loudness_sone[c] = _specific_loudness_1d_exact(core[:, c], dz)

    return z, spec, loudness_sone


def specific_loudness_from_core(core, dz=0.1):
    """Calculate specific loudness for one- or two-dimensional input.

    Parameters
    ----------
    core : ndarray
        Either:
        - shape (20,) or (21,)
        - shape ``(20, n_channels)`` or ``(21, n_channels)``
    dz : float, optional
        Bark spacing of the output specific-loudness grid.

    Returns
    -------
    z : ndarray
        Bark axis
    spec : ndarray
        Specific loudness pattern
    loudness_sone : float or ndarray
        Total loudness
    """
    core = np.asarray(core, dtype=np.float64)

    if core.ndim == 1:
        return _specific_loudness_1d_exact(core, dz)

    if core.ndim == 2:
        if core.shape[0] == 20:
            core21 = np.empty((21, core.shape[1]), dtype=np.float64)
            core21[:20, :] = core
            core21[20, :] = 0.0
            core = core21
        elif core.shape[0] != 21:
            msg = f'expected shape (20, n_channels) or (21, n_channels), got {core.shape}'
            raise ValueError(msg)

        return _specific_loudness_from_core_2d_21(core, dz)

    msg = f'expected 1D or 2D input, got {core.shape}'
    raise ValueError(msg)


def resample_to_48k(signal, fs):
    """Resample a time signal to 48 kHz using polyphase filtering.

    The implementation follows the approach used by
    ``pyfar.dsp.resample`` and uses ``scipy.signal.resample_poly``.
    The time-domain amplitude is preserved without additional
    normalization.

    Parameters
    ----------
    signal : array_like
        Time signal with shape ``(n_samples,)`` or
        ``(n_samples, n_channels)``. The sample dimension is the first
        dimension.
    fs : int or float
        Sampling frequency of the input signal in Hz. It must represent
        a whole-number sampling rate.

    Returns
    -------
    signal_48k : ndarray
        Signal resampled to 48 kHz. The time-domain amplitude is
        preserved.

    Notes
    -----
    The implementation follows the polyphase approach used by
    :func:`scipy.signal.resample_poly`. The complete signal is resampled at
    once, so this function does not maintain state between Acoular blocks.
    """
    signal = np.asarray(signal, dtype=np.float64)
    fs = float(fs)

    if signal.ndim not in (1, 2) or signal.shape[0] == 0:
        msg = 'signal must contain at least one sample'
        raise ValueError(msg)

    if signal.ndim == 2 and signal.shape[1] == 0:
        msg = 'signal must contain at least one channel'
        raise ValueError(msg)

    if not np.all(np.isfinite(signal)):
        msg = 'signal must contain only finite values'
        raise ValueError(msg)

    if not np.isfinite(fs) or fs <= 0:
        msg = 'fs must be positive and finite'
        raise ValueError(msg)

    if not fs.is_integer():
        msg = 'fs must represent a whole-number sampling rate'
        raise ValueError(msg)

    fs = int(fs)

    if fs == 48_000:
        return signal.copy()

    divisor = gcd(48_000, fs)
    up = 48_000 // divisor
    down = fs // divisor

    return resample_poly(signal, up, down, axis=0)


# *********************************************************************#
#
#   Implementation of stationary Zwicker Loudness
#   according to ISO 532-1 (DIN ISO 532-1:2022-03).
#
# *********************************************************************#


def loudness_st(signal, fs, field='F', loudness_level=False, analysis_start=0):
    """Calculate stationary loudness according to ISO 532-1.

    This is the internal 48 kHz implementation of the Zwicker method described
    in DIN ISO 532-1:2022-03.

    Processing steps
    ----
    1. Input calibrated sound pressure signal, resample to 48 kHz if necessary
    2. Split signal into third octave frequency bands
    3. Apply low frequency weighting factors
    4. Combine bands into critical bands
    5. Apply level corrections for HRTF and field types (a0, ΔLDF, ΔLCB)
    6. Compute core loudness for each critical band
    7. Map the result to the Bark scale
    8. For each Bark position from 0.1 to 24 in steps of 0.1:
    - estimate specific loudness
    - compute slope loudness
    - use the larger value
    9. Sum specific loudness over Bark
    10. Return total loudness, Bark axis, and specific loudness

    Parameters
    ----------
    signal : array-like
        Calibrated sound pressure in Pa with shape ``(n_samples,)`` for
        mono data or ``(n_samples, n_channels)`` for multichannel data.
    fs : int
        Sampling frequency in Hz. The algorithm core requires 48 kHz.
    field : str
        Sound field, "F" for free field and "DF" for diffuse field.
    loudness_level : bool
        If True, also return the loudness level in phon.

    Returns
    -------
    loudness_sone : float or ndarray, shape (n_channels,)
        Overall loudness in sone. A scalar is returned for one-dimensional
        mono input.
    bark_axis : array-like
        Frequency groups / Bark axis from 0 to 24 in increments of 0.1.
    specific_loudness_sone_per_bark : ndarray
        Specific loudness in sone per Bark, with shape ``(240,)`` for
        one-dimensional mono input or ``(240, n_channels)`` for
        multichannel input.
    """
    signal_input = np.asarray(signal, dtype=np.float64)
    mono_input = signal_input.ndim == 1

    # ============================================================
    # Input validation: check fs and field type
    # ============================================================

    if signal_input.ndim not in (1, 2):
        msg = f'signal must have shape (n_samples,) or (n_samples, n_channels), got {signal_input.shape}'
        raise ValueError(msg)
    if signal_input.shape[0] == 0 or (signal_input.ndim == 2 and signal_input.shape[1] == 0):
        msg = 'signal must contain at least one sample and one channel'
        raise ValueError(msg)
    if not np.all(np.isfinite(signal_input)):
        msg = 'signal must contain only finite values'
        raise ValueError(msg)
    try:
        fs = float(fs)
    except (TypeError, ValueError) as error:
        msg = 'fs must be positive and finite'
        raise ValueError(msg) from error
    if not np.isfinite(fs) or fs <= 0:
        msg = 'fs must be positive and finite'
        raise ValueError(msg)

    if fs != 48_000:
        msg = 'the stationary loudness algorithm core requires 48 kHz'
        raise ValueError(msg)

    if field not in ('F', 'DF'):
        msg = "field type must be 'F' for free-field or 'DF' for diffuse-field"
        raise ValueError(msg)

    if mono_input:
        signal_input = signal_input[:, None]

    # ******************* beginning of loudness computation *****************#

    # ===================== third-octave filters =====================#
    # ============================================================
    # Apply the third-octave filter bank as described in Section 4,
    # implemented as 6th-order Chebyshev filters using three
    # second-order sections (SOS) with hard coded coefficients
    # and gains. --> see _third_octave_filter
    # The filtered signals are reduced immediately to 28 third-octave levels,
    # avoiding a large bands-by-samples-by-channels intermediate array.
    # ============================================================

    band_spls = third_octave_levels_multichannel(signal_input, analysis_start)

    # ====== apply low frequency correction to first 11 bands  ===================#
    # ============================================================
    # Low-frequency weighting from Table A.3 to account for reduced
    # hearing sensitivity at low frequencies
    # Applies to 1/3-octave bands with center frequencies < 300 Hz
    # L_T_weighted = L_T + delta_L (as per ISO 532-1, Section 5.3)
    # ============================================================

    band_spls[:11] = apply_correction(band_spls[:11])

    # =======  sum lower bands as described in 5.2 step 2  ===================#
    # ============================================================
    # Since third-octave bands as approximation of critical bands
    # are only valid above 300 Hz, several of the lower bands have
    # (25-250Hz) to be summed to obtain approximation of the
    # critical bands. So in this step, the 28  third octave bands
    # are transformed into 20 critical bands.
    # ============================================================

    def power_sum_db_axis0(levels_db):
        """Function to power sum several band SPLs."""
        levels_db = np.asarray(levels_db, dtype=float)
        with np.errstate(divide='ignore'):
            return 10 * np.log10(np.sum(10 ** (levels_db / 10.0), axis=0))

    # first 6, further 3 (7-9) and 10-11 third-octave bands are combined
    power_sum_band_slices = [slice(0, 6), slice(6, 9), slice(9, 11)]

    critical_band_1 = power_sum_db_axis0(band_spls[power_sum_band_slices[0]])
    critical_band_2 = power_sum_db_axis0(band_spls[power_sum_band_slices[1]])
    critical_band_3 = power_sum_db_axis0(band_spls[power_sum_band_slices[2]])

    critical_bands_raw = np.vstack(
        (critical_band_1, critical_band_2, critical_band_3, band_spls[11:])
    )  # remaining bands stay unchanged

    # Apply field correction and calculate core loudness according to Section 5.2, step 3.
    # ============================================================
    # Field corrections in dB are applied to critical bands
    # depending on sound field and proximity to the hearing threshold.
    # Then core loudness per critical band is computed according
    # to formula A.2.
    # ============================================================

    # free or diffuse field
    if field == 'F':
        diffuse = False
    elif field == 'DF':
        diffuse = True

    critical_bands = core_loudness_from_bands(critical_bands_raw, diffuse=diffuse)

    # ==================== first critical band correction =============================#
    # ============================================================
    # Correction of the specific loudness within the lowest critical band for the
    # consideration of the run of threshold in quiet within this critical band.
    # ============================================================================

    corr_cl = 0.4 + 0.32 * (critical_bands[0] ** 0.2)
    # Limit the correction factor to 1.0 so that it never adds gain.
    corr_cl = np.minimum(corr_cl, 1.0)
    critical_bands[0] *= corr_cl

    # Construct the specific-loudness profile and overall loudness.
    # ============================================================================
    # Construct the specific-loudness profile from corrected
    # third-octave-band levels by applying band-dependent slopes
    # toward higher frequency groups and summing the area under the
    # resulting distribution.
    #
    # Overall loudness is obtained by integrating over specific loudness pattern.
    # ============================================================================

    bark_axis, specific_loudness_sone_per_bark, loudness_sone = specific_loudness_from_core(critical_bands)

    # ==================== calculate loudness level =============================#
    # ============================================================================
    # Loudness level in phon according to Section 5.3, formulas 2 and 3
    # ============================================================================

    if mono_input:
        loudness_sone = loudness_sone[0]
        specific_loudness_sone_per_bark = specific_loudness_sone_per_bark[:, 0]

    if loudness_level:
        phon = np.where(
            loudness_sone >= 1,
            40 + 33.22 * np.log10(np.maximum(loudness_sone, np.finfo(float).tiny)),
            40 * (loudness_sone + 0.0005) ** 0.35,
        )
        return loudness_sone, bark_axis, specific_loudness_sone_per_bark, phon

    return loudness_sone, bark_axis, specific_loudness_sone_per_bark


def sharpness_stationary(
    specific_loudness_sone_per_bark,
    bark_axis,
    method='din45692',
    normalization=0.11,
    loudness_sone=None,
):
    """Calculate stationary sharpness from a specific-loudness pattern.

    DIN 45692, Aures, and Bismarck weighting methods are supported.

    Parameters
    ----------
    specific_loudness_sone_per_bark : array_like
        Specific loudness in sone per Bark. Accepted shapes are ``(240,)``
        and ``(240, channels)``.
    bark_axis : array_like
        Critical-band-rate axis from 0.1 to 24 Bark in increments of 0.1 Bark.
    method : {'din45692', 'aures', 'bismarck'}, optional
        Sharpness weighting method. The default is ``'din45692'``.
    normalization : float, optional
        Normalization constant. The DIN 45692 value is 0.11.
    loudness_sone : float or array_like, optional
        Overall loudness in sone. It is required for the Aures method and
        otherwise calculated from ``specific_loudness_sone_per_bark`` if
        omitted.

    Returns
    -------
    sharpness_acum : float or ndarray
        Sharpness in acum, with one value per channel.
    """
    if method not in ['din45692', 'aures', 'bismarck']:
        msg = "method must be 'din45692', 'aures' or 'bismarck'"
        raise ValueError(msg)

    if method == 'aures' and loudness_sone is None:
        msg = "loudness_sone must be provided for method 'aures'"
        raise ValueError(msg)

    # Ensure inputs are NumPy arrays. The first dimension always represents
    # the Bark axis; an optional second dimension represents channels.
    specific_loudness_sone_per_bark = np.asarray(specific_loudness_sone_per_bark, dtype=np.float64)
    z = np.asarray(bark_axis, dtype=np.float64)
    mono = specific_loudness_sone_per_bark.ndim == 1
    if mono:
        specific_loudness_sone_per_bark = specific_loudness_sone_per_bark[:, np.newaxis]
    if specific_loudness_sone_per_bark.ndim != 2 or z.ndim != 1 or specific_loudness_sone_per_bark.shape[0] != z.size:
        msg = 'specific_loudness_sone_per_bark and bark_axis have incompatible shapes'
        raise ValueError(msg)
    expected_bark_axis = np.arange(0.1, 24.0 + 1e-12, 0.1)
    if z.shape != expected_bark_axis.shape or not np.allclose(z, expected_bark_axis):
        msg = 'bark_axis must run from 0.1 to 24 Bark in steps of 0.1 Bark'
        raise ValueError(msg)
    if not np.all(np.isfinite(specific_loudness_sone_per_bark)) or np.any(specific_loudness_sone_per_bark < 0):
        msg = 'specific_loudness_sone_per_bark must contain finite, non-negative values'
        raise ValueError(msg)

    if loudness_sone is None:
        loudness_sone = np.sum(specific_loudness_sone_per_bark * 0.1, axis=0)
    else:
        loudness_sone = np.asarray(loudness_sone, dtype=np.float64)
        if loudness_sone.ndim == 0:
            loudness_sone = np.full(specific_loudness_sone_per_bark.shape[1], loudness_sone)
        if loudness_sone.shape != (specific_loudness_sone_per_bark.shape[1],):
            msg = 'loudness_sone must be scalar or contain one value per channel'
            raise ValueError(msg)
    if not np.all(np.isfinite(loudness_sone)) or np.any(loudness_sone < 0):
        msg = 'loudness_sone must contain finite, non-negative values'
        raise ValueError(msg)

    if method == 'din45692':
        g = np.where(z <= 15.8, 1, 0.15 * np.exp(0.42 * (z - 15.8)) + 0.85)
        weighted = specific_loudness_sone_per_bark * g[:, np.newaxis]
    elif method == 'aures':
        g = 0.078 * np.exp(0.171 * z) / z
        loudness_weight = np.divide(
            loudness_sone,
            np.log1p(0.05 * loudness_sone),
            out=np.zeros_like(loudness_sone),
            where=loudness_sone > 0,
        )
        g = g[:, np.newaxis] * loudness_weight
        weighted = specific_loudness_sone_per_bark * g
    elif method == 'bismarck':
        g = np.where(z <= 15, 1, 0.2 * np.exp(0.308 * (z - 15)) + 0.8)
        weighted = specific_loudness_sone_per_bark * g[:, np.newaxis]

    numerator = normalization * np.sum(weighted * z[:, np.newaxis] * 0.1, axis=0)
    sharpness_acum = np.divide(
        numerator,
        loudness_sone,
        out=np.zeros_like(numerator),
        where=loudness_sone > 0,
    )
    return sharpness_acum[0] if mono else sharpness_acum


# ============================================================================
# Public Acoular integration
# ============================================================================

MIN_ANALYSIS_DURATION = 1.0


def loudness_stationary(signal, sample_freq, field='free', analysis_start=0):
    """Calculate stationary loudness for calibrated pressure time data.

    The complete input, including any pre-roll samples before
    ``analysis_start``, is resampled to 48 kHz when necessary. The stationary
    loudness algorithm itself always operates at 48 kHz.

    Parameters
    ----------
    signal : array_like
        Sound pressure in Pa with shape (samples, channels). A one-dimensional
        mono signal is accepted as well.
    sample_freq : float
        Sampling frequency in Hz. Whole-number rates are supported and are
        resampled to 48 kHz when necessary.
    field : {'free', 'diffuse'}, optional
        Sound-field condition.
    analysis_start : int, optional
        First sample included in the level average. Earlier samples are used
        only to settle the third-octave filters.

    Returns
    -------
    loudness_sone : float or ndarray
        Overall loudness in sone per channel.
    bark_axis : ndarray
        Critical-band-rate axis in Bark.
    specific_loudness_sone_per_bark : ndarray
        Specific loudness in sone/Bark.
    """
    fields = {'free': 'F', 'diffuse': 'DF'}
    if field not in fields:
        msg = "field must be 'free' or 'diffuse'"
        raise ValueError(msg)
    data = np.asarray(signal, dtype=np.float64)
    if data.ndim not in (1, 2) or data.shape[0] == 0:
        msg = 'signal must have shape (samples, channels) and contain data'
        raise ValueError(msg)
    if analysis_start < 0 or analysis_start >= data.shape[0]:
        msg = 'analysis_start must select a sample inside the signal'
        raise ValueError(msg)
    try:
        sample_freq = float(sample_freq)
    except (TypeError, ValueError) as error:
        msg = 'sample_freq must be positive and finite'
        raise ValueError(msg) from error
    if not np.isfinite(sample_freq) or sample_freq <= 0:
        msg = 'sample_freq must be positive and finite'
        raise ValueError(msg)
    if data.shape[0] - analysis_start < sample_freq * MIN_ANALYSIS_DURATION:
        msg = 'stationary loudness requires an analysis interval of at least 1 second'
        raise ValueError(msg)

    data_48k = resample_to_48k(data, sample_freq)
    analysis_start_48k = round(analysis_start * 48_000 / sample_freq)
    return loudness_st(data_48k, 48_000, field=fields[field], analysis_start=analysis_start_48k)


class LoudnessStationary(HasStrictTraits):
    """Calculate stationary loudness from an Acoular time-data source.

    The source is read in blocks, while the selected analysis interval is
    passed to :func:`loudness_stationary` as one continuous array. Thus,
    Acoular's input block size does not define the psychoacoustic analysis
    interval. The complete interval is resampled to 48 kHz when necessary.

    A :class:`~acoular.tbeamform.BeamformerTime` source produces one loudness
    value per grid point. Squared sources such as
    :class:`~acoular.tbeamform.BeamformerTimeSq` are not suitable because the
    ISO 532-1 filter bank requires the signed pressure waveform.
    """

    #: Time-domain source containing calibrated sound pressure in Pa.
    source = Instance(SamplesGenerator)

    #: First source sample included in the analysis interval.
    start = CInt(0)

    #: First source sample after the analysis interval. ``None`` uses the source end.
    stop = Union(None, CInt)

    #: Settling data read before start, in seconds, when available.
    pre_roll = Float(0.5)

    #: Block size used only while reading the source.
    block_size = Int(4096)

    #: Sound-field condition used by ISO 532-1.
    field = Enum('free', 'diffuse')

    #: Identifier of all parameters that affect the numerical result.
    digest = Property(depends_on=['source.digest', 'start', 'stop', 'pre_roll', 'field'])

    #: Overall loudness in sone, shaped (source.num_channels,).
    loudness = Property(depends_on='digest')

    #: Loudness level in phon, shaped (source.num_channels,).
    loudness_level = Property(depends_on='digest')

    #: Critical-band-rate axis, shaped (240,).
    bark_axis = Property(depends_on='digest')

    #: Specific loudness in sone/Bark, shaped (240, source.num_channels).
    specific_loudness = Property(depends_on='digest')

    #: Cached tuple containing all numerical results.
    results = Property(depends_on='digest')

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _validate_parameters(self):
        if self.start < 0:
            msg = 'start must be non-negative'
            raise ValueError(msg)
        if self.stop is not None and self.stop <= self.start:
            msg = 'stop must be greater than start'
            raise ValueError(msg)
        if self.pre_roll < 0:
            msg = 'pre_roll must be non-negative'
            raise ValueError(msg)
        if self.block_size <= 0:
            msg = 'block_size must be positive'
            raise ValueError(msg)
        if self.stop is None and self.source.num_samples < 0:
            msg = 'stop must be set for an indefinite source'
            raise ValueError(msg)

    def _collect_analysis_data(self):
        """Collect the selected source interval independently of block size."""
        requested_stop = self.source.num_samples if self.stop is None else self.stop
        explicit_stop = self.stop is not None
        collect_start = max(0, self.start - round(self.pre_roll * self.source.sample_freq))
        chunks = []
        source_index = 0

        for block in self.source.result(self.block_size):
            block = np.asarray(block, dtype=np.float64)
            if block.ndim != 2 or block.shape[1] != self.source.num_channels:
                msg = 'source blocks must have shape (samples, source.num_channels)'
                raise ValueError(msg)
            block_start = source_index
            block_stop = source_index + block.shape[0]
            source_index = block_stop
            if block_stop <= collect_start:
                continue
            if block_start >= requested_stop:
                break
            left = max(collect_start, block_start) - block_start
            right = min(requested_stop, block_stop) - block_start
            chunks.append(block[left:right])
            if block_stop >= requested_stop:
                break

        if explicit_stop and source_index < requested_stop:
            msg = 'source ended before stop was reached'
            raise ValueError(msg)
        if not chunks:
            msg = 'analysis interval contains no samples'
            raise ValueError(msg)
        data = np.concatenate(chunks, axis=0)
        analysis_start = self.start - collect_start
        if data.shape[0] - analysis_start < self.source.sample_freq * MIN_ANALYSIS_DURATION:
            msg = 'stationary loudness requires an analysis interval of at least 1 second'
            raise ValueError(msg)
        return data, analysis_start

    @cached_property
    def _get_results(self):
        self._validate_parameters()
        data, analysis_start = self._collect_analysis_data()
        loudness, bark_axis, specific_loudness = loudness_stationary(
            data,
            self.source.sample_freq,
            field=self.field,
            analysis_start=analysis_start,
        )
        loudness_level = np.where(
            loudness >= 1,
            40 + 33.22 * np.log10(np.maximum(loudness, np.finfo(float).tiny)),
            40 * (loudness + 0.0005) ** 0.35,
        )
        return loudness, loudness_level, bark_axis, specific_loudness

    def _get_loudness(self):
        return self.results[0]

    def _get_loudness_level(self):
        return self.results[1]

    def _get_bark_axis(self):
        return self.results[2]

    def _get_specific_loudness(self):
        return self.results[3]


class SharpnessStationary(HasStrictTraits):
    """Calculate stationary sharpness from stationary loudness results.

    The class reuses the overall and specific loudness calculated by a
    :class:`~acoular.psychoacoustics.LoudnessStationary` instance. It does not
    read the time-data source or execute the beamformer a second time.
    """

    #: Stationary loudness results used as input.
    loudness = Instance(LoudnessStationary)

    #: Sharpness weighting method.
    method = Enum('din45692', 'aures', 'bismarck')

    #: Normalization constant. DIN 45692 specifies 0.11.
    normalization = Float(0.11)

    #: Identifier of all inputs that affect the sharpness result.
    digest = Property(depends_on=['loudness.digest', 'method', 'normalization'])

    #: Stationary sharpness in acum, shaped (loudness.source.num_channels,).
    sharpness = Property(depends_on='digest')

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_sharpness(self):
        return sharpness_stationary(
            self.loudness.specific_loudness,
            self.loudness.bark_axis,
            method=self.method,
            normalization=self.normalization,
            loudness_sone=self.loudness.loudness,
        )
