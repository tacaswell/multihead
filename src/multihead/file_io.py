"""
Functions to read in raw data and extract single detectors.
"""

import ast
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self, cast

import h5py
import numpy as np
import numpy.typing as npt
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


class HRDRawBase:
    _detector_map: dict[int, tuple[int, int]]
    _data_path: ClassVar[str] = "/entry/data/data"

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

    def get_detector(self, n: int) -> npt.NDArray[np.uint16]:
        ds = cast(h5py.Dataset, self._h5_file[self._data_path])
        return load_det(ds, *self._detector_map[n])

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
    _h5_file: h5py.File
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
        return {_.desc: _.data for _ in self._mda.scan.d}

    def get_nominal_bin(self) -> float:
        sc = self._mda.scan_config
        (steps_per_bin,) = sc["MCS prescale"].value
        (step_size,) = sc["encoder resolution"].value

        return steps_per_bin * step_size


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
    target_chunks = (n_frames, 260, 260)
    with h5py.File(file_in, "a") as fin:
        source_dsname = "/entry/instrument/detector/data"
        dest_dsname = "/entry/data/data"
        read_ds = cast(h5py.Dataset, fin[source_dsname])

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

        for j in tqdm.tqdm(range(len(read_ds) // n_frames + 1), desc="re-chunking"):
            slc = slice(j * n_frames, (j + 1) * n_frames)
            dataset[slc] = read_ds[slc]
        del read_ds
        del fin[source_dsname]
        fin[source_dsname] = fin[dest_dsname]

    cache_name = Path(file_in).with_suffix(".cache")
    shutil.move(file_in, cache_name)
    try:
        subprocess.run(["h5repack", cache_name, file_in], check=True)
    except BaseException:
        shutil.move(cache_name, file_in)
    else:
        cache_name.unlink()


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


def open_data(fname: str | Path, version: int) -> HRDRawBase:
    t : HRDRawBase
    if version == 1:
        t = HRDRawV1.from_root(fname)
    elif version == 2:
        t = HRDRawV2(fname)
    else:
        raise ValueError(f"only version 1 and 2 supported, not {version}")
    return t
