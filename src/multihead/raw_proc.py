"""
Helpers for processing raw detector images.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import skimage.measure
from skimage.morphology import isotropic_closing, isotropic_opening

__all__ = ["CrystalROI", "DetectorROIs", "find_crystal_range"]


class SimpleSliceTuple(NamedTuple):
    start: int
    stop: int


@dataclass
class CrystalROI:
    rslc: SimpleSliceTuple
    cslc: SimpleSliceTuple

    def to_slices(self) -> tuple[slice, slice]:
        return slice(*self.rslc), slice(*self.cslc)


@dataclass
class DetectorROIs:
    rois: dict[int, CrystalROI]


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


def compute_rois(
    sums: dict[int, npt.NDArray[np.uint16]],
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
    return DetectorROIs(out)
