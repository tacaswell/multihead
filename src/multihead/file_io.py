"""
Functions to read in raw data and extract single detectors.
"""

import ast
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, Self, cast
from typing import overload, Literal

from collections.abc import Generator

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import sparse
import tqdm

from multihead import mda


@dataclass
class SimpleConfigEntry:
    unit: str = field(repr=False)
    value: Any


@dataclass
class ConfigEntry(SimpleConfigEntry):
    epics_type: int = field(repr=False)
    count: int = field(repr=False)
    pv: str


@dataclass
class MDA:
    scan_md: dict[str, Any] = field(repr=False)
    scan_config: dict[str, ConfigEntry] = field(repr=False)
    detectors: dict[str, npt.NDArray[Any]] = field(repr=False)
    scan: mda.scanDim = field(repr=False)


class HRDRawProtocol(Protocol):
    """
    Protocol defining the interface for raw detector data readers.

    All reader implementations should provide these methods.
    """

    def get_detector(self, n: int) -> sparse.COO:
        """
        Get sparse detector data for a single detector.

        Parameters
        ----------
        n : int
            Detector number (1-12)

        Returns
        -------
        sparse.COO
            Sparse array with shape (n_frames, 256, 256)
        """
        ...

    def get_detector_sums(self) -> dict[int, npt.NDArray[np.uint64]]:
        """
        Get sum of all frames for each detector.

        Returns
        -------
        dict[int, NDArray]
            Dictionary mapping detector number to 2D sum array
        """
        ...

    def get_arm_tth(self) -> npt.NDArray[np.float64]:
        """
        Get the 2-theta positions for each frame.

        Returns
        -------
        NDArray
            Array of 2-theta values
        """
        ...

    def get_monitor(self) -> npt.NDArray[np.float64]:
        """
        Get the monitor counts for each frame.

        Returns
        -------
        NDArray
            Array of monitor values
        """
        ...

    def get_nominal_bin(self) -> float:
        """
        Get the nominal bin size.

        Returns
        -------
        float
            Nominal bin size in degrees
        """

    def iter_detector_data(self) -> Generator[tuple[int, sparse.COO], None]:
        """
        Iterate through detectors
        """
        ...


class HRDRawBase:
    _detector_map: dict[int, tuple[int, int]]
    _data_path: ClassVar[str] = "/entry/data/data"
    _h5_file: h5py.File

    def __init__(
        self,
        detector_map: Sequence[Sequence[int]] = (
            (10, 9, 6, 5, 2, 1),
            (12, 11, 8, 7, 4, 3),
        ),
    ):
        self._detector_map = {}
        for k, row in enumerate(detector_map):
            for j, det_number in enumerate(row):
                self._detector_map[det_number] = (k, j)
        super().__init__()

    def get_detector(self, n: int) -> sparse.COO:
        ds = cast(h5py.Dataset, self._h5_file[self._data_path])
        return sparse.COO(load_det(ds, *self._detector_map[n]))

    def iter_detector_data(self) -> Generator[tuple[int, sparse.COO], None]:
        for k in sorted(self._detector_map):
            yield k, self.get_detector(k)

    def get_detector_sums(self) -> dict[int, npt.NDArray[np.uint64]]:
        sums: dict[int, npt.NDArray[np.uint64]] = {
            d + 1: np.zeros((256, 256), dtype=np.uint64)
            for d in range(len(self._detector_map))
        }
        ds = cast(h5py.Dataset, self._h5_file[self._data_path])
        chunks = cast(tuple[int, ...], ds.chunks)
        shape = cast(tuple[int, ...], ds.shape)
        if chunks[1:] == shape[1:]:
            # inefficient chunking for detector acesss
            n_frames = 1_000
            n_blocks = 1 + shape[0] // n_frames

            for j in tqdm.tqdm(range(n_blocks)):
                block = ds[j * n_frames : (j + 1) * n_frames]
                for d in range(1, len(self._detector_map) + 1):
                    sums[d] += block[:, *det_slice(*self._detector_map[d])].sum(axis=0)

        else:
            # efficient chunking for detector acesss
            for d in range(1, len(self._detector_map) + 1):
                sums[d] += self.get_detector(d).sum(axis=0)
        return sums

    def get_arm_tth(self) -> npt.NDArray[np.float64]: ...
    def get_monitor(self) -> npt.NDArray[np.float64]: ...
    def get_nominal_bin(self) -> float: ...


class HRDRawV1(HRDRawBase):
    _mda: MDA

    def __init__(self, mda_path: Path, image_path: Path, **kwargs):
        # TODO make opening this lazy?
        self._h5_file = h5py.File(image_path)

        md, scan = mda.readMDA(str(mda_path))
        scan_md = {k: md[k] for k in md["ourKeys"] if k != "ourKeys"}
        scan_config = {
            v[0]: ConfigEntry(*[*v[1:], k])
            for k, v in md.items()
            if k not in scan_md and k != "ourKeys"
        }
        self._mda = MDA(
            scan_md,
            scan_config,
            {d.desc: d.data for d in scan.d if np.sum(d.data) != 0},
            scan,
        )
        super().__init__(**kwargs)

    @classmethod
    def from_root(cls, root: str | Path, **kwargs) -> Self:
        root_p = Path(root)
        return cls(root_p.with_suffix(".mda"), root_p.with_suffix(".h5"), **kwargs)

    def get_arm_tth(self) -> npt.NDArray[np.float64]:
        sc = self._mda.scan_config
        (steps_per_bin,) = sc["MCS prescale"].value
        (step_size,) = sc["encoder resolution"].value

        bin_size: float = steps_per_bin * step_size

        (Npts,) = sc["NPTS"].value
        start_tth: float
        (start_tth,) = sc["start_tth_rbk"].value

        return start_tth + bin_size * np.arange(Npts, dtype=float)

    def get_monitor(self) -> npt.NDArray[np.float64]:
        d = next(_ for _ in self._mda.scan.d if _.desc == 'Monitor')
        return np.asarray(d.data)

    def get_nominal_bin(self) -> float:
        sc = self._mda.scan_config
        (steps_per_bin,) = sc["MCS prescale"].value
        (step_size,) = sc["encoder resolution"].value

        return float(steps_per_bin * step_size)


class HRDRawV2(HRDRawBase):
    _monitor_path: ClassVar[str] = "/entry/data/Mon"
    _tth_path: ClassVar[str] = "/entry/data/TTH"
    _MD_MAPPING: ClassVar[dict[str, Callable[[str], Any]]] = {
        "Run no.": int,
        "No. steps": int,
        "Scan Comment": str,
        "Sample Comment": str,
        "User sample name": lambda x: ast.literal_eval(x).decode("latin-1"),
        "Composition": str,
        "Proposal number": str,
        "Email(s)": str,
        "Temp (K)": lambda x: float(ast.literal_eval(x)),
        "Barcode ID": str,
        "Start 2theta (deg)": float,
        "End 2theta (deg)": float,
        "Nominal 2theta step (deg)": float,
        "Time per step (sec)": float,
        "Actual 2theta step (deg)": float,
        "200/Ring Current[0]": float,
        "Monitor I0 [0]": float,
        "Monitor I1 [0]": float,
    }

    @classmethod
    def extract_md(cls, md: list[str]) -> dict[str, SimpleConfigEntry]:
        def parse_comments(inp: list[str]) -> dict[str, str]:
            out = {}
            for e in inp:
                k, _, v = e.partition("=")
                out[k[2:].strip()] = v
            return out

        def split_key(k: str) -> tuple[str, str]:
            if "(" not in k:
                return k.strip(), ""
            return tuple(x.strip() for x in re.match(r"([^(]+)\(([^)]+)\)", k).groups())

        out = {}
        for k, v in parse_comments(md).items():
            key, unit = split_key(k)
            out[key] = SimpleConfigEntry(value=cls._MD_MAPPING[k](v.strip()), unit=unit)

        return out

    def __init__(self, path: Path, **kwargs):
        # TODO make opening this lazy?
        self._h5_file = h5py.File(path)
        self._md = self.extract_md(list(self._h5_file["entry"].attrs["Comments"]))
        super().__init__(**kwargs)

    def get_arm_tth(self) -> npt.NDArray[np.float64]:
        return self._h5_file[self._tth_path][:]

    def get_monitor(self) -> npt.NDArray[np.float64]:
        return self._h5_file[self._monitor_path][:]

    def get_nominal_bin(self) -> float:
        return self._md["Nominal 2theta step"].value


class HRDRawV3:
    """
    Reader for parquet-based sparse data format.

    This class reads pre-converted parquet files containing sparse detector data
    and metadata (tth, monitor) and provides the same API as V1 and V2 readers.

    Parameters
    ----------
    data_dir : Path
        Directory containing images.parquet and scalars.parquet files
    """
    _sparse_data: sparse.COO
    _tth: npt.NDArray[np.float64]
    _monitor: npt.NDArray[np.float64]
    _nominal_bin: float
    _detector_map: dict[int, tuple[int, int]]

    def __init__(
        self,
        data_dir: Path,
        detector_map: Sequence[Sequence[int]] = (
            (10, 9, 6, 5, 2, 1),
            (12, 11, 8, 7, 4, 3),
        ),
    ):
        """
        Initialize from a directory containing images.parquet and scalars.parquet.

        Parameters
        ----------
        data_dir : Path
            Directory containing images.parquet and scalars.parquet files
        detector_map : Sequence[Sequence[int]]
            Detector numbering map (same as HRDRawBase)

        Raises
        ------
        FileNotFoundError
            If images.parquet or scalars.parquet are missing
        ValueError
            If data_dir is not a directory
        """
        data_dir = Path(data_dir)

        if not data_dir.is_dir():
            raise ValueError(f"Expected a directory, got: {data_dir}")

        images_path = data_dir / "images.parquet"
        scalars_path = data_dir / "scalars.parquet"

        if not images_path.exists():
            raise FileNotFoundError(f"Missing required file: {images_path}")

        if not scalars_path.exists():
            raise FileNotFoundError(f"Missing required file: {scalars_path}")

        # Set up detector map
        self._detector_map = {}
        for k, row in enumerate(detector_map):
            for j, det_number in enumerate(row):
                self._detector_map[det_number] = (k, j)

        # Read the sparse detector images
        images_table = pq.read_table(images_path)
        self._sparse_data = sparse.COO(
            [images_table[k] for k in ["detector", "frame", "row", "col"]],
            data=images_table["data"],
            shape=json.loads(images_table.schema.metadata[b"shape"]),
        )

        # Read scalars
        scalars_table = pq.read_table(scalars_path)
        self._tth = scalars_table["tth"].to_numpy()
        self._monitor = scalars_table["monitor"].to_numpy()

        # Extract nominal_bin from scalars metadata
        if b"nominal_bin" in scalars_table.schema.metadata:
            self._nominal_bin = float(
                scalars_table.schema.metadata[b"nominal_bin"]
            )
        else:
            raise ValueError(
                f"Missing required metadata 'nominal_bin' in {scalars_path}"
            )

    @classmethod
    def from_data_path(cls, data_path: Path, **kwargs) -> Self:
        """
        Create instance from just the data parquet path.

        This will automatically use the parent directory.

        Parameters
        ----------
        data_path : Path
            Path to the images.parquet file
        **kwargs
            Additional keyword arguments passed to __init__

        Returns
        -------
        HRDRawV3 instance
        """
        return cls(data_path.parent, **kwargs)

    def get_detector(self, n: int) -> sparse.COO:
        """
        Extract a single detector from the sparse data.

        Parameters
        ----------
        n : int
            Detector number (1-12)

        Returns
        -------
        detector_data : sparse.COO
            Sparse COO array with shape (n_frames, 256, 256)
        """
        # Convert to 0-indexed
        detector_idx = n - 1

        # Extract the detector from the sparse array
        return self._sparse_data[detector_idx]

    def iter_detector_data(self) -> Generator[tuple[int, sparse.COO], None]:
        for k in sorted(self._detector_map):
            yield k, self.get_detector(k)

    def get_detector_sums(self) -> dict[int, npt.NDArray[np.uint64]]:
        """
        Get sum of all frames for each detector.

        Returns
        -------
        sums : dict
            Dictionary mapping detector number to 2D sum array
        """
        sums: dict[int, npt.NDArray[np.uint64]] = {}

        for detector_num in range(1, 13):
            detector_data = self.get_detector(detector_num)
            sums[detector_num] = detector_data.sum(axis=0, dtype=np.uint64).todense()

        return sums

    def get_arm_tth(self) -> npt.NDArray[np.float64]:
        """Get the 2-theta positions."""
        return self._tth

    def get_monitor(self) -> npt.NDArray[np.float64]:
        """Get the monitor counts."""
        return self._monitor

    def get_nominal_bin(self) -> float:
        """Get the nominal bin size."""
        return self._nominal_bin


def rechunk(file_in: str | Path, file_out: str | Path, *, n_frames: int = 1000) -> None:
    """
    Re-chunk the main detector array
    """
    with h5py.File(file_in) as fin, h5py.File(file_out, "w") as fout:
        dsname = "/entry/instrument/detector/data"

        read_ds = cast(h5py.Dataset, fin[dsname])
        # block_size = 0 let Bitshuffle choose its value
        block_size = 0

        dataset = fout.create_dataset(
            dsname,
            shape=read_ds.shape,
            chunks=(n_frames, 260, 260),
            compression=32008,
            compression_opts=(block_size, 2),
            dtype=read_ds.dtype,
        )

        for j in range(len(read_ds) // n_frames + 1):
            slc = slice(j * n_frames, (j + 1) * n_frames)
            dataset[slc] = read_ds[slc]

        # TODO: copy everything else!


def rechunk_in_place(file_in: str | Path, *, n_frames: int = 1000) -> None:
    """
    Re-chunk the main detector array
    """
    file_path = Path(file_in)
    working_copy = file_path.with_suffix(file_path.suffix + ".working")
    repacked_file = file_path.with_suffix(file_path.suffix + ".repacked")

    try:
        # Create working copy
        shutil.copy2(file_in, working_copy)

        with h5py.File(working_copy, "a") as fin:
            source_dsname = "/entry/instrument/detector/data"
            dest_dsname = "/entry/data/data"
            read_ds = cast(h5py.Dataset, fin[source_dsname])

            actual_chunk_size = min(read_ds.shape[0], n_frames)
            target_chunks = (actual_chunk_size, 260, 260)

            if dest_dsname in fin and fin[dest_dsname].chunks == target_chunks:
                return

            # block_size = 0 let Bitshuffle choose its value
            block_size = 0

            try:
                del fin[dest_dsname]
            except KeyError:
                ...
            dataset = fin.create_dataset(
                dest_dsname,
                shape=read_ds.shape,
                chunks=target_chunks,
                compression=32008,
                compression_opts=(block_size, 2),
                dtype=read_ds.dtype,
            )

            for j in tqdm.tqdm(
                range(len(read_ds) // actual_chunk_size + 1), desc="re-chunking"
            ):
                slc = slice(j * actual_chunk_size, (j + 1) * actual_chunk_size)
                dataset[slc] = read_ds[slc]
            del read_ds
            del fin[source_dsname]
            fin[source_dsname] = fin[dest_dsname]

        # Repack the working copy
        subprocess.run(["h5repack", working_copy, repacked_file], check=True)

        # Atomically replace the original file
        repacked_file.replace(file_path)

    finally:
        # Clean up temporary files
        for temp_file in [working_copy, repacked_file]:
            if temp_file.exists():
                temp_file.unlink()


def det_slice(n: int, m: int, *, pad: int = 4, npix: int = 256) -> tuple[slice, slice]:
    """
    Generate the slices to extract a single frame from 12 frame monolith

    Parameters
    ----------
    n, m : int
       The coordinates of the detector to generate the slices for

    pad : int, default=4
       The additional padding between the detectors in the monolithic frame

    npix : int, default=256
       The (square) size of a single detector in pixels

    Returns
    -------
    row_slc, col_slc : slice
        slice objects for row, col to extract a single detector
    """
    return (
        slice(n * (npix + pad), (n + 1) * npix + n * pad),
        slice(m * (npix + pad), (m + 1) * npix + m * pad),
    )


def load_det(
    dset: npt.NDArray[np.integer] | h5py.Dataset, n: int, m: int
) -> npt.NDArray[np.uint16]:
    """
    Helper to extract a single detector from a monolith (stack)

    Parameters
    ----------
    dset : h5py.Dataset or NDArray
       Expected to be 3D with dimensions (time, rows, cols).

    n, m : int
       The detector to extract.

    Returns
    -------
    The (stack) for a single detector.
    """
    return cast(npt.NDArray[np.uint16], dset[:, *det_slice(n, m)])


@overload
def open_data(fname: str | Path, version: Literal[1]) -> HRDRawV1: ...
@overload
def open_data(fname: str | Path, version: Literal[2]) -> HRDRawV2: ...
@overload
def open_data(fname: str | Path, version: Literal[3]) -> HRDRawV3: ...


def open_data(fname: str | Path, version: int) -> HRDRawProtocol:
    """
    Open raw data file and return appropriate reader.

    Parameters
    ----------
    fname : str or Path
        Path to the data file. For version 1 and 2, this is the HDF5 file path.
        For version 3, this should be a directory containing images.parquet
        and scalars.parquet files.
    version : int
        Format version: 1, 2, or 3

    Returns
    -------
    reader : HRDRawProtocol
        Appropriate reader instance for the file version
    """
    t: HRDRawProtocol
    if version == 1:
        t = HRDRawV1.from_root(fname)
    elif version == 2:
        t = HRDRawV2(fname)
    elif version == 3:
        fname_path = Path(fname)

        # Version 3 expects a directory containing the parquet files
        if not fname_path.exists():
            raise FileNotFoundError(f"Path does not exist: {fname_path}")

        if not fname_path.is_dir():
            raise ValueError(
                f"Version 3 requires a directory path, got file: {fname_path}"
            )

        t = HRDRawV3(fname_path)
    else:
        raise ValueError(f"only version 1, 2, and 3 supported, not {version}")
    return t
