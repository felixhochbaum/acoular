"""Tests for psychoacoustic metrics."""

import acoular as ac

import numpy as np
import pytest

SAMPLE_FREQ = 48_000


def sine_pressure(frequency, level, duration, sample_freq=SAMPLE_FREQ):
    """Create a sine with the requested RMS sound pressure level."""
    time = np.arange(round(duration * sample_freq)) / sample_freq
    rms = 20e-6 * 10 ** (level / 20)
    return np.sqrt(2) * rms * np.sin(2 * np.pi * frequency * time)


def band_limited_noise(lower_frequency, upper_frequency, level, seed):
    """Create deterministic flat-spectrum noise inside a frequency band."""
    frequencies = np.fft.rfftfreq(SAMPLE_FREQ, 1 / SAMPLE_FREQ)
    selected = (frequencies >= lower_frequency) & (frequencies <= upper_frequency)
    generator = np.random.default_rng(seed)
    spectrum = np.zeros(frequencies.size, dtype=complex)
    spectrum[selected] = np.exp(1j * generator.uniform(0, 2 * np.pi, selected.sum()))
    signal = np.fft.irfft(spectrum, SAMPLE_FREQ)
    signal /= np.sqrt(np.mean(signal**2))
    return signal * 20e-6 * 10 ** (level / 20)


@pytest.mark.parametrize(
    ('channel', 'lower', 'upper'),
    [
        (0, 13.921775, 15.387225),  # ISO 532-1 Annex B.2, 250 Hz at 80 dB
        (1, 3.818240, 4.220160),  # ISO 532-1 Annex B.2, 1 kHz at 60 dB
        (2, 1.449400, 1.649400),  # ISO 532-1 Annex B.2, 4 kHz at 40 dB
    ],
)
def test_stationary_loudness_iso_reference_tones(channel, lower, upper):
    """Check multichannel results against the ISO permitted ranges."""
    data = np.column_stack(
        [
            sine_pressure(250, 80, 10),
            sine_pressure(1000, 60, 10),
            sine_pressure(4000, 40, 10),
        ]
    )
    loudness, bark, specific = ac.loudness_stationary(data, SAMPLE_FREQ)

    assert lower <= loudness[channel] <= upper
    assert bark.shape == (240,)
    assert specific.shape == (240, 3)


def test_stationary_loudness_block_size_independent():
    """Changing the source read block size must not change the result."""
    data = sine_pressure(1000, 60, 2)[:, np.newaxis]
    source = ac.TimeSamples(data=data, sample_freq=SAMPLE_FREQ)
    first = ac.LoudnessStationary(source=source, block_size=4096)
    second = ac.LoudnessStationary(source=source, block_size=7919)

    np.testing.assert_array_equal(first.loudness, second.loudness)
    np.testing.assert_array_equal(first.loudness_level, second.loudness_level)
    np.testing.assert_array_equal(first.specific_loudness, second.specific_loudness)
    np.testing.assert_array_equal(first.bark_axis, second.bark_axis)


def test_stationary_loudness_analysis_interval_and_pre_roll():
    """Pre-roll must settle filters without entering the level average."""
    data = sine_pressure(1000, 60, 2)[:, np.newaxis]
    source = ac.TimeSamples(data=data, sample_freq=SAMPLE_FREQ)
    metric = ac.LoudnessStationary(
        source=source,
        start=SAMPLE_FREQ,
        stop=2 * SAMPLE_FREQ,
        pre_roll=0.5,
        block_size=5003,
    )
    collect_start = SAMPLE_FREQ // 2
    expected, _, expected_specific = ac.loudness_stationary(
        data[collect_start:],
        SAMPLE_FREQ,
        analysis_start=SAMPLE_FREQ - collect_start,
    )

    np.testing.assert_array_equal(metric.loudness, expected)
    np.testing.assert_array_equal(metric.specific_loudness, expected_specific)


def test_stationary_loudness_rejects_short_analysis_interval():
    """Stationary analysis requires at least one second of evaluated data."""
    data = np.zeros((SAMPLE_FREQ - 1, 1))

    with pytest.raises(ValueError, match='at least 1 second'):
        ac.loudness_stationary(data, SAMPLE_FREQ)

    source = ac.TimeSamples(data=data, sample_freq=SAMPLE_FREQ)
    with pytest.raises(ValueError, match='at least 1 second'):
        _ = ac.LoudnessStationary(source=source).loudness


def test_stationary_loudness_resamples_complete_interval_to_48_khz():
    """Whole-interval resampling must preserve the stationary result."""
    reference = sine_pressure(1000, 60, 2)
    resampled_input = sine_pressure(1000, 60, 2, sample_freq=44_100)
    expected, _, _ = ac.loudness_stationary(reference, SAMPLE_FREQ)
    actual, _, _ = ac.loudness_stationary(resampled_input, 44_100)

    assert actual == pytest.approx(expected, rel=0.01)

    metric = ac.LoudnessStationary(
        source=ac.TimeSamples(data=resampled_input[:, np.newaxis], sample_freq=44_100),
        pre_roll=0,
    )
    assert metric.loudness[0] == pytest.approx(expected, rel=0.01)


def test_stationary_sharpness_din_reference_anchor_and_multichannel_property():
    """Check the DIN 45692 one-acum anchor and the class add-on."""
    pressure = band_limited_noise(920, 1080, 60, seed=1006)
    loudness, bark, specific = ac.loudness_stationary(pressure, SAMPLE_FREQ)
    sharpness = ac.sharpness_stationary(
        specific,
        bark,
        method='din45692',
        loudness_sone=loudness,
    )

    assert sharpness == pytest.approx(1.0, abs=0.05)

    data = np.column_stack((pressure, 0.5 * pressure))
    loudness_metric = ac.LoudnessStationary(
        source=ac.TimeSamples(data=data, sample_freq=SAMPLE_FREQ),
        pre_roll=0,
    )
    sharpness_metric = ac.SharpnessStationary(loudness=loudness_metric)
    expected = ac.sharpness_stationary(
        loudness_metric.specific_loudness,
        loudness_metric.bark_axis,
        method='din45692',
        loudness_sone=loudness_metric.loudness,
    )

    np.testing.assert_array_equal(sharpness_metric.sharpness, expected)
    assert sharpness_metric.sharpness.shape == (2,)


@pytest.mark.parametrize('method', ['din45692', 'aures', 'bismarck'])
def test_stationary_sharpness_is_zero_for_silence(method):
    """All sharpness methods must return zero without numerical warnings."""
    bark_axis = np.arange(0.1, 24.0 + 1e-12, 0.1)
    specific_loudness = np.zeros((240, 2))

    sharpness = ac.sharpness_stationary(
        specific_loudness,
        bark_axis,
        method=method,
        loudness_sone=np.zeros(2),
    )

    np.testing.assert_array_equal(sharpness, np.zeros(2))
