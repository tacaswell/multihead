# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
# ---

# %% [markdown]
# # Corrections Analysis: Arm vs Scattering Angle Differences
#
# This notebook demonstrates the geometric corrections needed for analyzing scattering data
# from a multi-head diffractometer. It shows two key relationships:
#
# 1. How the arm angle differs from the nominal scattering angle for various scattering angles
# 2. How the corrected scattering angle differs from the arm angle for various arm positions
#
# These corrections account for the finite size of the detector and the geometry of the
# analyzer crystals in the diffractometer setup.

# %% [markdown]
# ## Setup and Configuration
#
# Import required libraries and configure the analyzer with realistic parameters.

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from multihead.config import AnalyzerConfig
from multihead.corrections import arm_from_z, tth_from_z

# %%
# Configure analyzer with realistic parameters
cfg = AnalyzerConfig(
    910,  # R: sample to crystal distance (mm)
    120,  # Rd: crystal to detector distance (mm)
    np.rad2deg(np.arcsin(0.8 / (2 * 3.1355))),  # theta_i: incident angle
    2 * np.rad2deg(np.arcsin(0.8 / (2 * 3.1355))),  # theta_d: diffraction angle
    detector_roll=0,
)

# Define detector positions along the axial direction
z = np.array(np.linspace(-10, 10, 256))

# %% [markdown]
# ## Arm Angle vs Scattering Angle Difference
#
# This plot shows how the arm angle (2Θ) differs from the scattering angle (2θ)
# as a function of axial position on the detector for various scattering angles.
# The difference accounts for the geometric corrections due the missalignment of the analyzer
# crystals and the finite size of the detector.
#
# This is useful to understand the effect of the analyzer crystal with area detector
# passing through the Debye-Scherrer cone.  This is not useful for data reduction.
#
# Shown as the difference so everything fits on the same axis.

# %%
fig, ax = plt.subplots(layout="constrained")
thetas = np.arange(5, 50, 5)[::-1]
cmap = mpl.colormaps["viridis"]

arm_tths, _ = arm_from_z(z.reshape(1, -1), thetas.reshape(-1, 1), cfg)

for arm_tth, tth, color in zip(
    arm_tths, thetas, cmap(np.linspace(0, 1, len(thetas))), strict=True
):
    ax.plot(z, arm_tth - tth, label=rf"$2\theta = {tth:g}°$", color=color)

ax.legend()
ax.set_ylabel(r"$2\Theta - 2\theta$ (deg)")
ax.set_xlabel(r"axial offset from center of detector (mm)")
ax.set_title("Arm Angle Correction vs Detector Position")

plt.show(block=False)

# %% [markdown]
# ## Corrected Scattering Angle vs Arm Angle Difference
#
# This plot shows how the corrected scattering angle (2θ) differs from the arm angle (2Θ)
# as a function of axial position on the detector for various fixed arm angles.
#
# This represents the correction, needed for data reduction as we experimentally
# know the arm position and need to compute the true scattering angle.
# %%
fig,ax = plt.subplots(layout="constrained")
arm_angles = np.arange(5, 50, 5)[::-1]
cmap = mpl.colormaps["viridis"]

corrected_tths, _ = tth_from_z(z.reshape(1, -1), arm_angles.reshape(-1, 1), cfg)

for corr_tth, arm_tth, color in zip(
    corrected_tths, arm_angles, cmap(np.linspace(0, 1, len(arm_angles))), strict=True
):
    ax.plot(z,  - corr_tth + arm_tth, label=rf"$2\Theta = {arm_tth:g}°$", color=color)

ax.legend()
ax.set_ylabel(r"$2\Theta - 2\theta$ (deg)")
ax.set_xlabel(r"axial offset from center of detector (mm)")
ax.set_title("Scattering Angle Correction vs Detector Position")

plt.show(block=True)
