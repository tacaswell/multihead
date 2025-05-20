# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm
from matplotlib.patches import Rectangle

from multihead.file_io import RawHRPD11BM
from multihead.raw_proc import find_crystal_range

root = Path("/mnt/scratch/hrd/data_cache/Lab6_testdata_share/data/")
f1 = "test_04_15_2025_0009"
f2 = "LaB6_WoSlits_04_30_0000"
f3 = "AL2O3_WoSlits_04_30_0000"

t = RawHRPD11BM.from_root(root / f2)

# %%
sums = t.get_detector_sums()
# %%


rois = {}
thresholds = [0, 1, 2, 3, 5]
fig = plt.figure(layout="compressed")
figs = fig.subfigures(len(sums))
for sfig, (k, v) in zip(figs, sums.items(), strict=True):
    ax_arr = sfig.subplots(1, len(thresholds) + 1, sharex=True, sharey=True)
    ax_orig, *ax_masked = ax_arr
    # ax_orig.set_title("Detector sum")
    ax_orig.axis("off")
    ax_orig.imshow(v, vmax=max(1, float(np.percentile(v, 90))))
    for th, ax in zip(thresholds, ax_masked, strict=True):
        mask, croi = find_crystal_range(v > th, closing_radius=10, opening_radius=10)
        ax.imshow(mask, cmap="gray")
        ax.add_artist(
            Rectangle(
                (croi.cslc.start, croi.rslc.start),
                croi.cslc.stop - croi.cslc.start,
                croi.rslc.stop - croi.rslc.start,
                facecolor="none",
                edgecolor="r",
            )
        )
        ax.axis("off")
        # ax.set_title(f"threshold > {th}")
    rois[k] = croi
