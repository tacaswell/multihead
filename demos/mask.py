# %%
from pathlib import Path
from typing import TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Rectangle

from multihead.file_io import RawHRPD11BM
from multihead.raw_proc import CrystalROI, find_crystal_range

dt = TypeVar("dt", bound=np.generic, covariant=True)
# %%

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
        rois[(k, th)] = croi


# %%
def compute_rois(
    sums: dict[int, npt.NDArray[np.uint16]],
    th: int = 2,
    closing_radius: int = 10,
    opening_radius: int = 10,
) -> dict[int, CrystalROI]:
    out: dict[int, CrystalROI] = {}
    for det, data in sums.items():
        _mask, croi = find_crystal_range(
            data > th, closing_radius=closing_radius, opening_radius=opening_radius
        )
        out[det] = croi
    return out


# %%


def extract_khymo(
    detector_series: npt.NDArray[dt], roi: CrystalROI | None = None
) -> npt.NDArray[dt]:
    """
    Extract the khymograph for a single detector sequence.

    Parameters
    ----------
    detector_series : array
       Expected order is (time, row, col)

    roi: CrystalROI
       The ROI to consider.  If None, sum whole detector

    Returns
    -------

    """
    if roi is None:
        rslc = cslc = slice(-1)
    else:
        rslc = roi.rslc
        cslc = roi.cslc

    return cast(npt.NDArray[dt], np.sum(detector_series[:, rslc, cslc], axis=1))


# %%
def get_arm_tth(self) -> npt.NDArray[np.float64]:
    sc = self._mda.scan_config
    (steps_per_bin,) = sc["MCS prescale"].value
    (step_size,) = sc["encoder resolution"].value

    bin_size: float = steps_per_bin * step_size

    (Npts,) = sc["NPTS"].value
    start_tth: float
    (start_tth,) = sc["start_tth_rbk"].value

    return start_tth + bin_size * np.arange(Npts, dtype=float)


def get_monitor(self):
    return {_.desc: _.data for _ in self._mda.scan.d}


# %%
def get_khymograph(
    raw: RawHRPD11BM, detector: int, roi: CrystalROI | None = None
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.uint16]]:
    """
    Extract the khymograph for a single detector sequence.

    Parameters
    ----------
    raw :
       Expected order is (time, row, col)

    roi: CrystalROI
       The ROI to consider.  If None, sum whole detector

    """
    series = raw.get_detector(detector)
    print(roi)
    khymo = extract_khymo(series, roi)

    tth = get_arm_tth(raw)

    return tth, khymo


# %%


def all_khymographs(
    raw: RawHRPD11BM, rois: dict[int, CrystalROI]
) -> dict[int, tuple[npt.NDArray[np.floating], npt.NDArray[np.uint16]]]:
    return {det: get_khymograph(raw, det, roi) for det, roi in rois.items()}


# %%


def integrate_simple(tth: npt.NDArray[np.floating], khymo: npt.NDArray[dt]):
    return tth, khymo.sum(axis=1)


# %%

rois2 = compute_rois(sums)
all_khymos = all_khymographs(t, rois2)

# %%

flats = {det: integrate_simple(tth, khymo) for det, (tth, khymo) in all_khymos.items()}

# %%


def estimate_crystal_offsets_ref(
    raw: RawHRPD11BM,
    flats: dict[int, tuple[npt.NDArray[np.floating], npt.NDArray[np.uint16]]],
    ref_indx: int = 0,
) -> dict[int, float]:
    sc = raw._mda.scan_config
    (steps_per_bin,) = sc["MCS prescale"].value
    (step_size,) = sc["encoder resolution"].value

    bin_size: float = steps_per_bin * step_size
    ref = flats[ref_indx][1]
    (Npts,) = ref.shape
    out: dict[int, float] = {}
    for det, (_, I) in flats.items():
        offset = np.argmax(np.correlate(ref, I, mode="full")) - Npts
        out[det] = offset * bin_size

    return out


def estimate_crystal_offsets(
    raw: RawHRPD11BM,
    flats: dict[int, tuple[npt.NDArray[np.floating], npt.NDArray[np.uint16]]],
) -> dict[int, float]:
    sc = raw._mda.scan_config
    (steps_per_bin,) = sc["MCS prescale"].value
    (step_size,) = sc["encoder resolution"].value

    bin_size: float = steps_per_bin * step_size

    out: dict[int, float] = {}
    iterator = iter(flats.items())
    det, (_, ref) = next(iterator)
    (Npts,) = ref.shape
    out[det] = cum_offset = 0.0

    for det, (_, I) in iterator:
        offset = np.argmax(np.correlate(ref, I, mode="full")) - Npts
        cum_offset += offset * bin_size
        out[det] = cum_offset
        ref = I

    return out


# %%

offsets = estimate_crystal_offsets(t, flats)

# %%

fig, ax = plt.subplots(layout="constrained")

{
    ax.plot(tth + offsets[d], I + d * 200, label=str(d))[0]
    for d, (tth, I) in flats.items()
}
ax.legend()
ax.set_xlabel("tth")
ax.set_ylabel("I")

plt.show()
