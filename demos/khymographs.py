from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm
from matplotlib.widgets import Cursor

from multihead.cli import get_base_parser
from multihead.config import BankCalibration, CrystalROI, DetectorROIs, SpectraCalib
from multihead.file_io import HRDRawBase, open_data
from multihead.raw_proc import compute_rois, correct_ttheta


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
    khymo = extract_khymo(series, roi)

    tth = raw.get_arm_tth()

    return tth, khymo


# %%


def all_khymographs(
    raw: HRDRawBase, rois: dict[int, CrystalROI]
) -> dict[int, tuple[npt.NDArray[np.floating], npt.NDArray[np.uint16]]]:
    return {
        det: get_khymograph(raw, det, roi)
        for det, roi in tqdm.tqdm(rois.items(), desc="getting hkymos")
    }


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
    parser.add_argument(
        "--roi-config",
        type=str,
        help="Path to ROI configuration file (YAML format)",
        required=False,
    )
    parser.add_argument(
        "--calibration-config",
        type=str,
        help="Path to calibration configuration file (YAML or text format)",
        required=False,
    )
    parser.add_argument(
        "--detectors",
        type=int,
        nargs="*",
        help="List of detector numbers to process. If not specified, all detectors will be processed.",
        required=False,
    )
    return parser.parse_args()


def load_configs(args) -> tuple[DetectorROIs | None, BankCalibration | None]:
    """Load ROI and calibration configuration files if provided.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments containing roi_config and calibration_config paths

    Returns
    -------
    tuple[DetectorROIs | None, BankCalibration | None]
        Tuple containing loaded ROI config and calibration config, or None if not provided
    """
    roi_config = None
    calibration_config = None

    if args.roi_config:
        roi_path = Path(args.roi_config)
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI config file not found: {roi_path}")
        roi_config = DetectorROIs.from_yaml(roi_path)

    if args.calibration_config:
        calib_path = Path(args.calibration_config)
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration config file not found: {calib_path}")

        # Determine file format by extension
        if calib_path.suffix.lower() in [".yaml", ".yml"]:
            calibration_config = BankCalibration.from_yaml(calib_path)
        else:
            # Assume text format
            calibration_config = BankCalibration.from_text(calib_path)

    return roi_config, calibration_config


def main():
    args = parse_args()

    # Load optional configuration files
    roi_config, calibration_config = load_configs(args)

    # Initialize the RawHRPD11BM instance with command line arguments
    t = open_data(args.filename, args.ver)

    # Use provided ROI config if available, otherwise compute ROIs
    if roi_config is not None:
        print("Using ROI configuration from file")
        rois2 = roi_config
    else:
        print("Computing ROIs from detector sums")
        rois2 = compute_rois(t.get_detector_sums())

    # Filter detectors if specific ones were requested
    if args.detectors is not None:
        filtered_rois = {
            det: roi for det, roi in rois2.rois.items() if det in args.detectors
        }
        rois2.rois = filtered_rois
        print(f"Processing detectors: {sorted(args.detectors)}")
    else:
        print(f"Processing all detectors: {sorted(rois2.rois.keys())}")

    all_khymos = all_khymographs(t, rois2.rois)

    flats = {
        det: integrate_simple(tth, khymo) for det, (tth, khymo) in all_khymos.items()
    }

    # Use calibration config if available, otherwise estimate offsets and create calibration
    if calibration_config is None:
        print("Estimating crystal offsets from data")
        offsets = estimate_crystal_offsets(t, flats)

        # Create BankCalibration object with estimated offsets and default values
        default_wavelength = 0.8272  # Å for 15 keV
        default_scale = 1.0

        calibrations = {
            det: SpectraCalib(
                offset=offset, scale=default_scale, wavelength=default_wavelength
            )
            for det, offset in offsets.items()
        }

        calibration_config = BankCalibration(
            calibrations=calibrations,
            software={
                "name": "multihead",
                "version": "dev",
                "script": "khymographs.py",
            },
            parameters={
                "num_detectors": len(offsets),
                "estimation_method": "correlation_based",
                "default_wavelength_nm": default_wavelength,
                "default_scale": default_scale,
            },
        )
    calibs = calibration_config.calibrations

    mon = t.get_monitor()

    # Plotting
    fig, ax = plt.subplots(layout="constrained")
    lines = [
        ax.plot(
            correct_ttheta(
                tth,
                calibs[d].offset,
                calibs[d].wavelength,
                calibration_config.average_wavelength(),
            ),
            (I / mon) * calibs[d].scale,
            label=str(d),
        )[0]
        for d, (tth, I) in flats.items()
    ]
    ax.legend()
    ax.set_xlabel(r"2$\theta$")
    ax.set_ylabel("I")
    # Set useblit=True on most backends for enhanced performance.
    cursor = Cursor(ax, useblit=True, color="red", linewidth=2)

    cmap = plt.get_cmap("viridis")
    cmap.set_under("w")
    for det, (tth, khymo) in all_khymos.items():
        ctth = correct_ttheta(
            tth,
            calibs[det].offset,
            calibs[det].wavelength,
            calibration_config.average_wavelength(),
        )
        fig_kyho = plt.figure()
        fig_kyho.suptitle(f"Detector {det}")
        ax_d = fig_kyho.subplot_mosaic("AB", width_ratios=(1, 5), sharey=True)
        ax_d["B"].imshow(
            khymo,
            aspect="auto",
            extent=(
                0,
                khymo.shape[1],
                ctth.min(),
                ctth.max(),
            ),
            origin="lower",
            vmin=1,
            cmap=cmap,
        )
        ax_d["A"].plot(flats[det][1], ctth)
    plt.show()
    return fig, lines, cursor


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
