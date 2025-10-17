"""
Configuration and data structure definitions for multihead.
"""

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple, Self, TextIO

import yaml

__all__ = [
    "BankCalibration",
    "CrystalROI",
    "DetectorROIs",
    "SimpleSliceTuple",
    "SpectraCalib",
]


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
    software: dict[str, str]
    parameters: dict[str, int]

    def to_yaml(self, stream: str | Path | TextIO) -> None:
        """
        Write the DetectorROIs to a YAML file.

        Parameters
        ----------
        stream : str, Path, or TextIO
            The output file path or stream to write the YAML data to.
            If a string or Path, the file will be opened and closed automatically.
        """
        with ExitStack() as stack:
            if isinstance(stream, (str, Path)):
                stream = stack.enter_context(open(stream, "w"))

            data = {
                "software": self.software,
                "parameters": self.parameters,
                "rois": [
                    {
                        "detector_number": k,
                        "roi_bounds": {"rslc": list(v.rslc), "cslc": list(v.cslc)},
                    }
                    for k, v in self.rois.items()
                ],
            }
            yaml.dump(data, stream)

    @classmethod
    def from_yaml(cls, stream: str | Path | TextIO) -> "DetectorROIs":
        """
        Read DetectorROIs from a YAML file.

        Parameters
        ----------
        stream : str, Path, or TextIO
            The input file path or stream containing YAML data.
            If a string or Path, the file will be opened and closed automatically.

        Returns
        -------
        DetectorROIs
            A new DetectorROIs instance with data loaded from the YAML file.
        """
        with ExitStack() as stack:
            if isinstance(stream, (str, Path)):
                stream = stack.enter_context(open(stream))

            data = yaml.safe_load(stream)
            rois = {}
            for entry in data["rois"]:
                k = entry["detector_number"]
                v = entry["roi_bounds"]
                rois[int(k)] = CrystalROI(
                    SimpleSliceTuple(*v["rslc"]), SimpleSliceTuple(*v["cslc"])
                )
            return cls(
                rois=rois,
                software=data.get("software", {}),
                parameters=data.get("parameters", {}),
            )


@dataclass
class SpectraCalib:
    """
    Spectral calibration parameters for a single detector.

    Attributes
    ----------
    offset : float
        Angular offset in degrees.
    scale : float
        Scale factor, arbitrary units near 1.
    wavelength : float
        Wavelength per analyzer in angstrom (Å)
    """

    # degrees
    offset: float
    # arb, near 1
    scale: float
    # Å
    wavelength: float


@dataclass
class BankCalibration:
    """
    Calibration data for a bank of detectors.

    Contains calibration parameters for multiple detectors along with
    software metadata and processing parameters.

    Attributes
    ----------
    calibrations : dict[int, SpectraCalib]
        Dictionary mapping detector numbers to their calibration parameters.
    software : dict[str, str]
        Software metadata including version, name, etc.
    parameters : dict[str, int | str]
        Processing parameters including number of detectors, calibration source, etc.
    """

    calibrations: dict[int, SpectraCalib]
    software: dict[str, str]
    parameters: dict[str, Any]

    def average_wavelength(self):
        return sum(det.wavelength for det in self.calibrations.values()) / len(
            self.calibrations
        )

    def to_yaml(self, stream: str | Path | TextIO) -> None:
        """
        Write the Calibration to a YAML file.

        Parameters
        ----------
        stream : str, Path, or TextIO
            The output file path or stream to write the YAML data to.
            If a string or Path, the file will be opened and closed automatically.
        """
        with ExitStack() as stack:
            if isinstance(stream, (str, Path)):
                stream = stack.enter_context(open(stream, "w"))

            data = {
                "software": self.software,
                "parameters": self.parameters,
                "calibrations": [
                    {"detector_number": k, "calibration": asdict(v)}
                    for k, v in self.calibrations.items()
                ],
            }
            yaml.dump(data, stream)

    @classmethod
    def from_yaml(cls, stream: str | Path | TextIO) -> Self:
        """
        Read BankCalibration from a YAML file.

        Parameters
        ----------
        stream : str, Path, or TextIO
            The input file path or stream containing YAML data.
            If a string or Path, the file will be opened and closed automatically.

        Returns
        -------
        Self
            A new BankCalibration instance with data loaded from the YAML file.
        """
        with ExitStack() as stack:
            if isinstance(stream, (str, Path)):
                stream = stack.enter_context(open(stream))

            data: dict[str, Any] = yaml.safe_load(stream)
            calibrations: dict[int, SpectraCalib] = {}
            for entry in data["calibrations"]:
                k = entry["detector_number"]
                v = entry["calibration"]
                calibrations[int(k)] = SpectraCalib(
                    offset=v["offset"], scale=v["scale"], wavelength=v["wavelength"]
                )
            return cls(
                calibrations=calibrations,
                software=data.get("software", {}),
                parameters=data.get("parameters", {}),
            )

    @classmethod
    def from_text(cls, stream: str | Path | TextIO) -> Self:
        """
        Read BankCalibration from a text file.

        Expected format with columns: Parameter, Lam, Zero, Scale.
        The Parameter column should contain detector names with numbers appended,
        e.g., '11bmb_1743_mda_defROI_1:'. The Lam column maps to wavelength,
        Zero column contains offset deviation from expected (2*(num-1)), and
        Scale column maps directly to scale.

        Parameters
        ----------
        stream : str, Path, or TextIO
            The input file path or stream containing CSV data with calibration parameters.
            If a string or Path, the file will be opened and closed automatically.

        Returns
        -------
        Self
            A new BankCalibration instance with data loaded from the CSV file.
            The calibration_source parameter will contain the base name of the
            calibration files.

        Notes
        -----
        The offset values are converted from relative deviations to absolute
        offsets using the formula: absolute_offset = 2*(detector_num-1) + deviation.
        """
        with ExitStack() as stack:
            if isinstance(stream, (str, Path)):
                stream = stack.enter_context(open(stream))

            # Skip header line
            next(stream, None)

            calibrations: dict[int, SpectraCalib] = {}
            calibration_sources: list[str] = []

            for line in (ln.strip() for ln in stream):
                if not line:
                    continue

                # Parse the line - split on whitespace to handle variable spacing
                parts = line.split()
                if len(parts) < 4:
                    continue

                parameter_name = parts[0].rstrip(":")
                wavelength = float(parts[1])
                offset_deviation = float(parts[2])
                scale = float(parts[3])

                # Extract detector number from parameter name (last part after underscore)
                detector_num = int(parameter_name.split("_")[-1])

                # Calculate absolute offset: expected offset is 2*(num-1) + deviation
                expected_offset = 2 * (detector_num - 1)
                absolute_offset = expected_offset + offset_deviation

                # Create calibration
                calibrations[detector_num] = SpectraCalib(
                    offset=absolute_offset, scale=scale, wavelength=wavelength
                )

                # Store calibration source
                calibration_sources.append(parameter_name)

            # Prepare parameters with calibration source info
            parameters: dict[str, Any] = {}

            # Add calibration source as a single string (joined sources)
            if calibration_sources:
                # Extract base name (everything before the detector number)
                base_name = (
                    calibration_sources[0].rsplit("_", 1)[0]
                    if calibration_sources
                    else ""
                )
                parameters["calibration_source"] = base_name

            return cls(
                calibrations=calibrations,
                software={},
                parameters=parameters,
            )
