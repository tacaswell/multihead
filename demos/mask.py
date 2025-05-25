# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider

from multihead.file_io import RawHRPD11BM
from multihead.raw_proc import CrystalROI, find_crystal_range

# %%

root = Path("/mnt/scratch/hrd/data_cache/Lab6_testdata_share")
data_root = root / "data"
calib_root = root / "calib"

# f = "test_04_15_2025_0009"
# f = "LaB6_WoSlits_04_30_0000"
f = "AL2O3_WoSlits_04_30_0000"

t = RawHRPD11BM.from_root(data_root / f)

opening_radius = 7
closing_radius = 15

# %%
sums = t.get_detector_sums()
# %%


rois = {}
thresholds = [0, 1, 2, 3, 5]
fig = plt.figure(figsize=(15, 9), layout="compressed")
fig.suptitle(f"{f}")
data_fig, input_fig = fig.subfigures(2, height_ratios=[7, 1])


def make_figure(fig, sums, opening_radius, closing_radius):
    figs = fig.subfigures(1, len(sums) + 1, width_ratios=(0.5,) + (1,) * len(sums))
    images: dict[tuple[int, int], AxesImage] = {}
    rects: dict[tuple[int, int], Rectangle] = {}
    for sfig, (k, v) in zip(figs[1:], sums.items(), strict=True):
        ax_arr = sfig.subplots(len(thresholds) + 1, sharex=True, sharey=True)
        ax_orig, *ax_masked = ax_arr
        # ax_orig.set_title("Detector sum")
        ax_orig.axis("off")
        ax_orig.imshow(v, vmax=max(1, float(np.percentile(v, 90))))
        ax_orig.set_title(f"Crystal {k}")
        for th, ax in zip(thresholds, ax_masked, strict=True):
            mask, croi = find_crystal_range(
                v > th, closing_radius=closing_radius, opening_radius=opening_radius
            )
            images[(k, th)] = ax.imshow(mask, cmap="gray")
            rects[(k, th)] = ax.add_artist(
                Rectangle(
                    (croi.cslc.start, croi.rslc.start),
                    croi.cslc.stop - croi.cslc.start,
                    croi.rslc.stop - croi.rslc.start,
                    facecolor="none",
                    edgecolor="r",
                )
            )
            ax.axis("off")
            if sfig is figs[0]:
                txt.set_in_layout(False)
            rois[(k, th)] = croi

    text_axes = figs[0].subplots(len(thresholds) + 1, sharex=True, sharey=True)
    [ax.axis("off") for ax in text_axes]
    text_axes[0].annotate(
        "raw",
        (0.5, 0.5),
        xycoords="axes fraction",
        rotation="vertical",
        va="center",
        ha="center",
    )
    for ax, th in zip(text_axes[1:], thresholds):
        ax.annotate(
            f"{th=}",
            (0.5, 0.5),
            xycoords="axes fraction",
            rotation="vertical",
            va="center",
            ha="center",
        )
    return figs, images, rects


def compute_masks_rois(sums, opening_radius, closing_radius):
    masks: dict[tuple[int, int], npt.NDArray[np.integer]] = {}
    rois: dict[tuple[int, int], CrystalROI] = {}
    for k, v in sums.items():
        for th in thresholds:
            mask, croi = find_crystal_range(
                v > th, closing_radius=closing_radius, opening_radius=opening_radius
            )
            masks[(k, th)] = mask
            rois[(k, th)] = croi

    return masks, rois


def update_masks(
    images: dict[tuple[int, int], AxesImage],
    rects: dict[tuple[int, int], Rectangle],
    masks: dict[tuple[int, int], npt.NDArray[np.integer]],
    rois: dict[tuple[int, int], CrystalROI],
):
    for k, mask in masks.items():
        images[k].set_data(mask)
    for k, croi in rois.items():
        rect = rects[k]
        rect.set_xy((croi.cslc.start, croi.rslc.start))
        rect.set_width(croi.cslc.stop - croi.cslc.start)
        rect.set_height(croi.rslc.stop - croi.rslc.start)


def make_interaction(fig, opening_radius, closing_radius, rects, images):
    state = {"opening_radius": opening_radius, "closing_radius": closing_radius}

    def _shared_callback():
        update_masks(rects, images, *compute_masks_rois(sums, **state))
        fig.canvas.draw_idle()

    def _update_opening(val):
        state["opening_radius"] = int(val)
        _shared_callback()

    def _update_closing(val):
        state["closing_radius"] = int(val)
        _shared_callback()

    ax_o, ax_c = fig.subplots(1, 2)

    s1 = Slider(ax_c, "closing radius", 1, 25, valinit=closing_radius, valstep=1)
    s2 = Slider(ax_o, "opening_radius", 1, 25, valinit=opening_radius, valstep=1)
    s1.on_changed(_update_closing)
    s2.on_changed(_update_opening)

    return s2, s1


figs, rects, images = make_figure(data_fig, sums, opening_radius, closing_radius)
keep_alive = make_interaction(input_fig, opening_radius, closing_radius, rects, images)

# %%
