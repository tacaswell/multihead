"""
Helpers for processing raw detector images.
"""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import scipy.signal
import skimage.measure
import sparse
from skimage.morphology import isotropic_closing, isotropic_opening

import multihead

from .config import CrystalROI, DetectorROIs, SimpleSliceTuple
from .file_io import HRDRawProtocol

__all__ = ["compute_rois", "find_crystal_range"]


def find_crystal_range(
    photon_mask: npt.ArrayLike, opening_radius: float = 5, closing_radius: float = 10
) -> tuple[npt.NDArray[np.int_], CrystalROI]:
    """
    Find the ROI on the detector that capture the passed photons.

    Parameters
    ----------
    photon_mask
        Pixels that have photons in the total sum

    Returns
    -------
    CrystalROI

    Notes
    -----
    Adapted from
    https://discuss.python.org/t/how-can-i-detect-and-crop-the-rectangular-frame-in-the-image/32378/2
    """

    seg_cleaned = isotropic_closing(
        isotropic_opening(photon_mask, opening_radius), closing_radius
    )

    def get_main_component(segments: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        labels: npt.NDArray[np.int_] = skimage.measure.label(segments)
        if labels.max() == 0:
            return segments
        ret: npt.NDArray[np.int_] = (
            labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        )
        return ret

    mask = get_main_component(seg_cleaned)

    mask_c = mask.max(axis=0)
    mask_r = mask.max(axis=1)
    indices_r = mask_r.nonzero()[0]
    indices_c = mask_c.nonzero()[0]
    minr, maxr = int(indices_r[0]), int(indices_r[-1])
    minc, maxc = int(indices_c[0]), int(indices_c[-1])
    return mask, CrystalROI(SimpleSliceTuple(minr, maxr), SimpleSliceTuple(minc, maxc))


def automatic_roi_selection(
    raw_data: HRDRawProtocol,
    *,
    th: int | None = None,
    closing_radius: int = 10,
    opening_radius: int = 10,
) -> DetectorROIs:
    out: dict[int, CrystalROI] = {}
    th_vec: list[int] = []
    for det_number, data in raw_data.iter_detector_data():
        simple = data.sum(axis=(2), dtype=np.uint32).todense().sum(axis=1)
        locs, props = scipy.signal.find_peaks(
            simple, width=(None, None), height=(None, None)
        )

        indx = np.argsort(props["peak_heights"])
        peak_center = locs[indx[-1]]
        window = props["widths"][indx[-1]] * 2
        slc = slice(peak_center - window // 2, peak_center + window // 2)
        summed_image = data[slc].sum(axis=(0), dtype=np.uint32).todense()
        if th is None:
            _th = int(np.percentile(summed_image, 95))
        else:
            _th = th
        th_vec.append(_th)
        _mask, croi = find_crystal_range(
            summed_image > _th,
            closing_radius=closing_radius,
            opening_radius=opening_radius,
        )
        out[det_number] = croi

        # import matplotlib.pyplot as plt
        #
        # fig, axd = plt.subplot_mosaic("AA;BC", layout="constrained")
        # fig.suptitle(f"det {det_number}")
        # axd["A"].plot(simple)
        # axd["A"].plot(locs[indx], props["peak_heights"][indx])
        # axd["B"].imshow(summed_image, origin="lower")
        # axd["C"].imshow(_mask, origin="lower")
        # axd["C"].set_title(
        #     f"{_th=:d} {opening_radius=} {closing_radius=}", size="small"
        # )

    return DetectorROIs(
        rois=out,
        software={
            "name": "multihead",
            "version": multihead.__version__,
            "function": "compute_rois",
            "module": "multihead.raw_proc",
        },
        parameters={
            "threshold": th_vec,
            "closing_radius": closing_radius,
            "opening_radius": opening_radius,
        },
    )


def compute_rois(
    sums: Mapping[int, npt.NDArray[np.integer]],
    th: int = 2,
    closing_radius: int = 10,
    opening_radius: int = 10,
) -> DetectorROIs:
    out: dict[int, CrystalROI] = {}
    for det, data in sums.items():
        _mask, croi = find_crystal_range(
            data > th, closing_radius=closing_radius, opening_radius=opening_radius
        )
        out[det] = croi
    return DetectorROIs(
        rois=out,
        software={
            "name": "multihead",
            "version": multihead.__version__,
            "function": "compute_rois",
            "module": "multihead.raw_proc",
        },
        parameters={
            "threshold": th,
            "closing_radius": closing_radius,
            "opening_radius": opening_radius,
        },
    )


def scale_tth(tth, wavelength: float, target_wavelength: float):
    """
    Bring tth values to common wavelength.

    Parameters
    ----------
    tth : array-like
        Scatter angles in deg

    wavelength : float
        The wavelength the of the measured data.

        Units must match *target_wavelength*

    target_wavelength : float
        The wavelength the to compute the scattering angles at.

        Units must match *wavelength*
    """
    π = np.pi

    def ttheta_to_q(tth, λ):
        return 4 * π / λ * np.sin(np.deg2rad(tth / 2))

    def q_to_ttheta(q, λ):
        return 2 * np.rad2deg(np.arcsin(λ / (4 * π) * q))

    return q_to_ttheta(ttheta_to_q(tth, wavelength), target_wavelength)


def get_roi_sum(detector_series: sparse.COO, roi: CrystalROI) -> npt.NDArray:
    rslc, cslc = roi.to_slices()
    roi_data = detector_series[:, rslc, cslc]
    return roi_data.sum(axis=(1, 2), dtype=np.uint32).todense()
