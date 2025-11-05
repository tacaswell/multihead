"""
Configuration and data structure definitions for multihead.
"""

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple, Self, TextIO

import numpy as np
import yaml

__all__ = [
    "AnalyzerConfig",
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
    parameters: dict[str, int | str | float]

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
class AnalyzerConfig:
    r"""
    Analyzer crystal configuration and geometry parameters.

    Attributes
    ----------
    R : float
        Sample to analyzer distance, mm

        $L$ in Fitch 2021 Fig 2
        *L* in multianalyzer code
    Rd : float
        Analyzer to detector distance, mm

        $L2$ in Fitch 2021 Fig 2
        *L2* in multianalyzer code
    theta_i : float
        The incident angle of the xrays on the analyzer crystal in deg.

        This is effectively the crystal pitch

        $\theta_a$ in Fitch 2021 Fig 2
        *tha* in multianalyzer code
    theta_d : float
        The angle of the normal to the detector in deg.

        $\theta_d$ in Fitch 2021 Fig 2
        *thd* in multianalyzer code
    crystal_roll : float
        The roll miss-alignment of the analyzer crystal rolling relative to the
        beam direction in deg

        $\vartheta_x$ in Fitch 2021 Fig 2
        *rollx* in multianalyzer code
    crystal_yaw : float
        The yaw miss-alignment of the analyzer crystal relative to the
        beam direction in deg

        $\vartheta_y$ in Fitch 2021 Fig 2
        *rolly* in multianalyzer code
    detector_yaw : float
        The yaw miss-alignment of the detector in deg
    detector_roll : float
        The roll miss-alignment of the detector in deg
    center : float
        Where the beam with phi=0 hits the detector.

        *center* in multianalyzer code
    """

    # mm
    R: float
    Rd: float
    # degrees
    theta_i: float
    theta_d: float
    crystal_roll: float = 0
    crystal_yaw: float = 0
    detector_yaw: float = 0
    detector_roll: float = 0
    # pixels
    center: float = 0


@dataclass
class SpectraCalib:
    """
    Spectral calibration parameters for a single detector.

    Attributes
    ----------
    offset : float
        Angular offset in degrees from arm

       *psi* in multianalyzer code
    scale : float
        Scale factor, arbitrary units near 1.  Accounts for different response of
        analyzer crystal and detector.
    wavelength : float
        Wavelength per analyzer in angstrom (Å).

        Due to slight differences in alignment, each analyzer crystal has a
        slightly different average energy of photons passed.
    analyzer : AnalyzerConfig
        Analyzer crystal configuration and geometry parameters.
    """

    # degrees
    offset: float
    # arb, near 1
    scale: float
    # Å
    wavelength: float
    analyzer: AnalyzerConfig


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
    pixel_pitch : float
        Pixel pitch in mm

        *pixel* in multianalyzer code
    """

    calibrations: dict[int, SpectraCalib]
    software: dict[str, str]
    parameters: dict[str, Any]
    pixel_pitch: float

    @property
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
                "pixel_pitch": self.pixel_pitch,
                "calibrations": [
                    {"detector_number": k, "calibration": asdict(v)}
                    for k, v in self.calibrations.items()
                ],
            }
            yaml.dump(data, stream)

    @classmethod
    def from_yaml(
        cls,
        stream: str | Path | TextIO,
        bank_defaults: dict[str, Any] | None = None,
        spectra_defaults: dict[str, Any] | None = None,
    ) -> Self:
        """
        Read BankCalibration from a YAML file.

        Parameters
        ----------
        stream : str, Path, or TextIO
            The input file path or stream containing YAML data.
            If a string or Path, the file will be opened and closed automatically.
        bank_defaults : dict[str, Any], optional
            Dictionary of default values for missing BankCalibration attributes
            (pixel_pitch).
        spectra_defaults : dict[str, Any], optional
            Dictionary of default values for missing SpectraCalib or AnalyzerConfig attributes
            (center, R, Rd, theta_i, theta_d, crystal_roll, crystal_yaw, detector_yaw, detector_roll).

        Returns
        -------
        Self
            A new BankCalibration instance with data loaded from the YAML file.
        """
        if bank_defaults is None:
            bank_defaults = {}
        if spectra_defaults is None:
            spectra_defaults = {}

        with ExitStack() as stack:
            if isinstance(stream, (str, Path)):
                stream = stack.enter_context(open(stream))

            data: dict[str, Any] = yaml.safe_load(stream)

            # Apply bank-level defaults to data
            for k, val in bank_defaults.items():
                data.setdefault(k, val)

            calibrations: dict[int, SpectraCalib] = {}
            for entry in data["calibrations"]:
                k = entry["detector_number"]
                v = entry["calibration"]

                # Apply spectra-level defaults to calibration data
                for spec_k, spec_val in spectra_defaults.items():
                    v.setdefault(spec_k, spec_val)

                # Create AnalyzerConfig from nested analyzer data
                analyzer_data = v["analyzer"]
                analyzer = AnalyzerConfig(
                    R=analyzer_data["R"],
                    Rd=analyzer_data["Rd"],
                    theta_i=analyzer_data["theta_i"],
                    theta_d=analyzer_data["theta_d"],
                    crystal_roll=analyzer_data["crystal_roll"],
                    crystal_yaw=analyzer_data["crystal_yaw"],
                    detector_yaw=analyzer_data["detector_yaw"],
                    detector_roll=analyzer_data["detector_roll"],
                    center=analyzer_data["center"],
                )

                calibrations[int(k)] = SpectraCalib(
                    offset=v["offset"],
                    scale=v["scale"],
                    wavelength=v["wavelength"],
                    analyzer=analyzer,
                )
            return cls(
                calibrations=calibrations,
                software=data["software"],
                parameters=data["parameters"],
                pixel_pitch=data["pixel_pitch"],
            )

    @classmethod
    def from_text(
        cls,
        stream: str | Path | TextIO,
        bank_defaults: dict[str, Any] | None = None,
        spectra_defaults: dict[str, Any] | None = None,
    ) -> Self:
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
        bank_defaults : dict[str, Any], optional
            Dictionary of default values for missing AnalyzerConfig attributes
            (R, Rd, pixel_pitch).
        spectra_defaults : dict[str, Any], optional
            Dictionary of default values for missing SpectraCalib attributes
            (center, crystal_roll, crystal_yaw, detector_yaw, detector_roll).

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
        if bank_defaults is None:
            bank_defaults = {}
        # mp3 55um pixels
        bank_defaults.setdefault("pixel_pitch", 0.055)
        # in mm
        bank_defaults.setdefault("R", 910)
        # in mm
        bank_defaults.setdefault("Rd", 120)
        if spectra_defaults is None:
            spectra_defaults = {}

        # center of mp3
        spectra_defaults.setdefault("center", 128)
        # perfectly aligned!
        spectra_defaults.setdefault("crystal_roll", 0)
        spectra_defaults.setdefault("crystal_yaw", 0)

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

                # Create calibration with spectra defaults applied
                spectra_data = {
                    "offset": absolute_offset,
                    "scale": scale,
                    "wavelength": wavelength,
                }

                # Apply spectra-level defaults
                for spec_k, spec_val in spectra_defaults.items():
                    spectra_data.setdefault(spec_k, spec_val)

                # assumes Si 111
                theta_bragg = np.rad2deg(np.arcsin(wavelength / (2 * 3.1355)))
                spectra_data.setdefault("theta_i", theta_bragg)
                # assume perfectly aligned detector
                spectra_data.setdefault("theta_d", 2 * spectra_data["theta_i"])

                # Create AnalyzerConfig with defaults from bank_defaults and calculated values
                analyzer = AnalyzerConfig(
                    R=bank_defaults["R"],
                    Rd=bank_defaults["Rd"],
                    theta_i=spectra_data["theta_i"],
                    theta_d=spectra_data["theta_d"],
                    crystal_roll=spectra_data["crystal_roll"],
                    crystal_yaw=spectra_data["crystal_yaw"],
                    detector_yaw=spectra_data["detector_yaw"],
                    detector_roll=spectra_data["detector_roll"],
                    center=spectra_data["center"],
                )

                calibrations[detector_num] = SpectraCalib(
                    offset=spectra_data["offset"],
                    scale=spectra_data["scale"],
                    wavelength=spectra_data["wavelength"],
                    analyzer=analyzer,
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

            # Apply bank-level defaults
            bank_data: dict[str, Any] = {}
            for bank_k, bank_val in bank_defaults.items():
                bank_data.setdefault(bank_k, bank_val)

            return cls(
                calibrations=calibrations,
                software={},
                parameters=parameters,
                pixel_pitch=bank_data["pixel_pitch"],
            )
