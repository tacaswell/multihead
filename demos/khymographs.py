from typing import cast
import tqdm
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np
import numpy.typing as npt
from multihead.config import CrystalROI
from multihead.raw_proc import compute_rois

from multihead.file_io import HRDRawBase, open_data
from multihead.cli import get_base_parser


def extract_khymo(
    detector_series: npt.NDArray, roi: CrystalROI | None = None
) -> npt.NDArray:
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
        rslc, cslc = roi.to_slices()

    return cast(npt.NDArray, np.sum(detector_series[:, rslc, cslc], axis=1))


# %%

# %%
def get_khymograph(
    raw: HRDRawBase, detector: int, roi: CrystalROI | None = None
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

    tth = raw.get_arm_tth()

    return tth, khymo


# %%


def all_khymographs(
    raw: HRDRawBase, rois: dict[int, CrystalROI]
) -> dict[int, tuple[npt.NDArray[np.floating], npt.NDArray[np.uint16]]]:
    return {det: get_khymograph(raw, det, roi) for det, roi in tqdm.tqdm(rois.items(), desc='getting hkymos')}


# %%


def integrate_simple(tth: npt.NDArray[np.floating], khymo: npt.NDArray):
    return tth, khymo.sum(axis=1)


# %%

def parse_args():
    parser = get_base_parser("Generate and visualize khymographs from detector data")
    parser.add_argument(
        "--ref-index",
        type=int,
        default=0,
        help="Reference detector index for offset calculation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize the RawHRPD11BM instance with command line arguments
    t = open_data(args.fname)
    sums = t.get_detector_sums()

    rois2 = compute_rois(sums)
    all_khymos = all_khymographs(t, rois2.rois)

    flats = {det: integrate_simple(tth, khymo) for det, (tth, khymo) in all_khymos.items()}

    offsets = estimate_crystal_offsets(t, flats)

    # Plotting
    fig, ax = plt.subplots(layout="constrained")

    {
        ax.plot(tth + offsets[d], I + d * 200, label=str(d))[0]
        for d, (tth, I) in flats.items()
    }
    ax.legend()
    ax.set_xlabel("tth")
    ax.set_ylabel("I")
    # Set useblit=True on most backends for enhanced performance.
    cursor = Cursor(ax, useblit=True, color='red', linewidth=2)

    plt.show()



def estimate_crystal_offsets(
    raw: HRDRawBase,
    flats: dict[int, tuple[npt.NDArray[np.floating], npt.NDArray[np.uint16]]],
) -> dict[int, float]:

    bin_size = raw.get_nominal_bin()
    out: dict[int, float] = {}
    iterator = iter(flats.items())
    det, (_, ref) = next(iterator)
    (Npts,) = ref.shape
    out[det] = cum_offset = 0.0

    for det, (_, I) in iterator:
        offset = np.argmax(np.correlate(ref, I, mode="full")) - Npts - 2
        cum_offset += offset * bin_size
        out[det] = cum_offset
        ref = I

    return out

if __name__ == "__main__":
    main()
