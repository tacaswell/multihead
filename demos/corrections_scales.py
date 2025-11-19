import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from multihead.config import AnalyzerConfig
from multihead.corrections import arm_from_z

cfg = AnalyzerConfig(
    910,
    120,
    np.rad2deg(np.arcsin(0.8 / (2 * 3.1355))),
    2 * np.rad2deg(np.arcsin(0.8 / (2 * 3.1355))),
    detector_roll=0,
)

z = np.array(np.linspace(-10, 10, 128))

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

plt.show()
