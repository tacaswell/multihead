import numpy as np
import numpy.typing as npt
import skimage.measure
import skimage.morphology


def find_crystal_range(
    photon_mask: npt.ArrayLike,
) -> tuple[npt.NDArray[np.int_], slice, slice]:
    """
    Find the ROI on the detector that capture the passed photons.

    Parameters
    ----------
    photon_mask
        Pixels that have photons in the total sum

    Returns
    -------
    (slice, slice)


    Notes
    -----
    Adapted from https://discuss.python.org/t/how-can-i-detect-and-crop-the-rectangular-frame-in-the-image/32378/2
    """

    seg_cleaned = skimage.morphology.isotropic_closing(
        skimage.morphology.isotropic_opening(photon_mask, 5), 10
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
    return mask, slice(minr, maxr), slice(minc, maxc)
