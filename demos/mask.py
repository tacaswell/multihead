# %%
import argparse
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import yaml
from matplotlib.figure import Figure, SubFigure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

import multihead
from multihead.file_io import RawHRPD11BM
from multihead.raw_proc import (
    CrystalROI,
    compute_rois,
    find_crystal_range,
)

# %%


# %%

# %%


def make_figure(
    fig: Figure,
    sums: Mapping[int, npt.NDArray[np.uint64]],
    thresholds: list[int],
    opening_radius: int,
    closing_radius: int,
) -> tuple[
    list[SubFigure],
    dict[tuple[int, int], Rectangle],
    dict[tuple[int, int], AxesImage],
]:
    figs = cast(
        list[SubFigure],
        fig.subfigures(1, len(sums) + 1, width_ratios=(0.5,) + (1,) * len(sums)),
    )
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
    for ax, th in zip(text_axes[1:], thresholds, strict=True):
        ax.annotate(
            f"{th=}",
            (0.5, 0.5),
            xycoords="axes fraction",
            rotation="vertical",
            va="center",
            ha="center",
        )
    return figs, rects, images


def compute_masks_rois(
    sums: Mapping[int, npt.NDArray[np.integer]],
    opening_radius: int,
    closing_radius: int,
    thresholds: list[int],
) -> tuple[
    dict[tuple[int, int], npt.NDArray[np.integer]],
    dict[tuple[int, int], CrystalROI],
]:
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


def make_interaction(
    fig: Figure,
    sums: Mapping[int, npt.NDArray[np.uint64]],
    thresholds: list[int],
    opening_radius: int,
    closing_radius: int,
    rects: dict[tuple[int, int], Rectangle],
    images: dict[tuple[int, int], AxesImage],
    f: str,
    calib_root: Path,
) -> tuple[Slider, Slider, Slider, Button]:
    state = {"opening_radius": opening_radius, "closing_radius": closing_radius}

    def _shared_callback():
        update_masks(
            images, rects, *compute_masks_rois(sums, **state, thresholds=thresholds)
        )
        fig.canvas.draw_idle()

    def _update_opening(val):
        state["opening_radius"] = int(val)
        _shared_callback()

    def _update_closing(val):
        state["closing_radius"] = int(val)
        _shared_callback()

    ax_dict = fig.subplot_mosaic("ac;bd")
    ax_c = ax_dict["a"]
    ax_o = ax_dict["b"]

    closing_slider = Slider(
        ax_c, "closing radius", 1, 25, valinit=closing_radius, valstep=1
    )
    opening_slider = Slider(
        ax_o, "opening_radius", 1, 25, valinit=opening_radius, valstep=1
    )
    closing_slider.on_changed(_update_closing)
    opening_slider.on_changed(_update_opening)

    th_slider = Slider(
        ax_dict["c"],
        "threshold",
        min(thresholds),
        max(thresholds),
        valinit=2,
        valstep=thresholds,
    )

    b = Button(
        ax_dict["d"],
        "Save",
        color="xkcd:bubble gum pink",
        hovercolor="xkcd:carnation pink",
    )

    def _on_save(event):  # noqa: ARG001
        compute_kwargs = dict(
            th=int(th_slider.val),
            closing_radius=int(closing_slider.val),
            opening_radius=int(opening_slider.val),
        )
        res = compute_rois(sums, **compute_kwargs)
        print(res)
        print(asdict(res))
        roi_root = calib_root / "rois"
        roi_root.mkdir(exist_ok=True, parents=True)
        dump_data = {
            **asdict(res),
            "settings": compute_kwargs,
            "software": {
                "func": f"{compute_rois.__module__}.{compute_rois.__qualname__}",
                "version": multihead.__version__,
            },
        }
        with open((roi_root / f).with_suffix(".yaml"), "w") as fout:
            yaml.dump(dump_data, fout)
        print(f"wrote to {fout=}")
        print(yaml.dump(dump_data))

    b.on_clicked(_on_save)

    return opening_slider, closing_slider, th_slider, b


# %%


def parse_args():
    parser = argparse.ArgumentParser(description="Process and visualize crystal masks")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory for data",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        help="Input filename",
    )
    parser.add_argument(
        "--opening-radius",
        type=int,
        default=7,
        help="Opening radius for mask processing",
    )
    parser.add_argument(
        "--closing-radius",
        type=int,
        default=15,
        help="Closing radius for mask processing",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 5],
        help="List of threshold values",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root
    f = args.filename
    opening_radius = args.opening_radius
    closing_radius = args.closing_radius
    thresholds = args.thresholds

    data_root = root / "data"
    calib_root = root / "calib"

    t = RawHRPD11BM.from_root(data_root / f)
    sums = t.get_detector_sums()

    fig = plt.figure(figsize=(15, 9), layout="compressed")
    fig.suptitle(f"{f}")
    data_fig, input_fig = fig.subfigures(2, height_ratios=[7, 1])

    figs, rects, images = make_figure(
        data_fig, sums, thresholds, opening_radius, closing_radius
    )
    _keep_alive = make_interaction(
        input_fig,
        sums,
        thresholds,
        opening_radius,
        closing_radius,
        rects,
        images,
        f,
        calib_root,
    )

    plt.show()


if __name__ == "__main__":
    main()
