# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
Stationary psychoacoustic mapping
=================================

This example combines time-domain beamforming with stationary loudness and
sharpness calculations. A 1 kHz reference tone with a sound pressure level of
40 dB at 1 m is placed in front of a square microphone array. This corresponds
to approximately 1 sone at the source position. Its level-dependent Aures
sharpness is approximately 0.89 acum. The beamformer reconstructs a pressure
time signal at every grid point before the psychoacoustic metrics are
calculated.

Sharpness describes spectral shape rather than level. It is therefore shown
only at grid points whose loudness is at least 10 percent of the map maximum.
"""

import acoular as ac

import matplotlib.pyplot as plt
import numpy as np

# Simulation setup
# ----------------
# A compact 6 x 6 planar array keeps the example fast enough for the
# documentation build. The source amplitude describes RMS sound pressure at
# the steering-vector reference distance of 1 m.
sample_freq = 48_000
duration = 1.25
num_samples = round(duration * sample_freq)

coordinates = (np.arange(6) - 2.5) * 0.1
x_mic, y_mic = np.meshgrid(coordinates, coordinates)
microphones = ac.MicGeom(
    pos_total=np.vstack((x_mic.ravel(), y_mic.ravel(), np.zeros(x_mic.size))),
)

pressure_rms = 20e-6 * 10 ** (40 / 20)

source = ac.PointSource(
    signal=ac.SineGenerator(
        sample_freq=sample_freq,
        num_samples=num_samples,
        freq=1000,
        amplitude=np.sqrt(2) * pressure_rms,
    ),
    mics=microphones,
    loc=(0.0, 0.0, 1.5),
)

# Time-domain beamforming
# -----------------------
# ``BeamformerTime`` returns one signed pressure signal per grid point. A
# squared time-domain beamformer would not be suitable because the ISO 532-1
# filter bank requires the pressure waveform.
grid = ac.RectGrid(
    x_min=-0.75,
    x_max=0.75,
    y_min=-0.75,
    y_max=0.75,
    z=1.5,
    increment=0.25,
)
steering = ac.SteeringVector(
    grid=grid,
    mics=microphones,
    ref=1.0,
    steer_type='true level',
)
beamformer = ac.BeamformerTime(source=source, steer=steering)

# Stationary loudness and sharpness
# ---------------------------------
# The first 0.2 s settle the filters but are excluded from the stationary
# analysis interval. The interval continues to the actual end of the
# beamformer output, which is slightly shorter than the simulated source due to
# propagation delays. Sharpness reuses the cached loudness results, so it does
# not execute the beamformer a second time.
loudness = ac.LoudnessStationary(
    source=beamformer,
    start=sample_freq // 5,
    pre_roll=0.2,
)
sharpness = ac.SharpnessStationary(loudness=loudness, method='aures')

loudness_map = loudness.loudness.reshape(grid.shape)
sharpness_map = sharpness.sharpness.reshape(grid.shape)
loudness_mask = loudness_map < 0.1 * loudness_map.max()
sharpness_map = np.ma.masked_where(loudness_mask, sharpness_map)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
maps = (
    (loudness_map, 'Loudness / sone'),
    (sharpness_map, r'Sharpness$_{\mathrm{Aures}}$ / acum'),
)
for axis, (data, colorbar_label) in zip(axes, maps, strict=True):
    image = axis.imshow(data.T, origin='lower', extent=grid.extent)
    axis.scatter([0.0], [0.0], marker='x', color='red', label='Source')
    axis.set(xlabel='x / m', ylabel='y / m')
    axis.legend()
    fig.colorbar(image, ax=axis, label=colorbar_label)

print(f'Maximum loudness: {loudness_map.max():.3f} sone')
print(f'Aures sharpness at maximum loudness: {sharpness_map.ravel()[loudness_map.argmax()]:.3f} acum')

# Specific loudness
# -----------------
# The specific-loudness spectrum at the source grid point shows the critical-
# band-rate distribution used to calculate total loudness and sharpness.
fig, axis = plt.subplots()
location = np.asarray((0.0, 0.0, 1.5))
index = np.argmin(np.linalg.norm(grid.pos - location[:, np.newaxis], axis=0))
axis.plot(loudness.bark_axis, loudness.specific_loudness[:, index], label='1 kHz reference tone')

axis.set(
    xlabel='Critical-band rate / Bark',
    ylabel='Specific loudness / sone/Bark',
    xlim=(0, 24),
)
axis.grid()
axis.legend()
plt.show()
