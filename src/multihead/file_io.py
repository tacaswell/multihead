"""
Functions to read in raw data and extract single detectors.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, cast

import h5py
import numpy as np
import numpy.typing as npt
import tqdm

from multihead import mda


@dataclass
class ConfigEntry:
    unit: str = field(repr=False)
    value: Any
    epics_type: int = field(repr=False)
    count: int = field(repr=False)
    pv: str


@dataclass
class MDA:
    scan_md: dict[str, Any] = field(repr=False)
    scan_config: dict[str, ConfigEntry] = field(repr=False)
    detectors: dict[str, npt.NDArray[Any]] = field(repr=False)
    scan: mda.scanDim = field(repr=False)


class RawHRPD11BM:
    _image_file: h5py.File
    _mda: MDA
    _detector_map: dict[int, tuple[int, int]]
    _data_path = "/entry/instrument/detector/data"

    def __init__(
        self,
        mda_path: Path,
        image_path: Path,
        detector_map: tuple[tuple[int, ...], ...],
    ):
        # TODO make opening this lazy?
        self._image_file = h5py.File(image_path)
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
        self._detector_map = {}
        for k, row in enumerate(detector_map):
            for j, det_number in enumerate(row):
                self._detector_map[det_number] = (k, j)

    @classmethod
    def from_root(cls, root: str | Path) -> Self:
        root_p = Path(root)
        return cls(
            root_p.with_suffix(".mda"),
            root_p.with_suffix(".h5"),
            (
                (10, 9, 6, 5, 2, 1),
                (12, 11, 8, 7, 4, 3),
            ),
        )

    def get_detector(self, n: int) -> npt.NDArray[np.uint16]:
        return load_det(self._image_file[self._data_path], *self._detector_map[n])

    def get_detector_sums(self) -> dict[int, npt.NDArray[np.uint64]]:
        sums: dict[int, npt.NDArray[np.uint64]] = {
            d + 1: np.zeros((256, 256), dtype=np.uint64)
            for d in range(len(self._detector_map))
        }
        ds = self._image_file[self._data_path]
        chunks = ds.chunks
        shape = ds.shape
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


def rechunk(file_in: str | Path, file_out: str | Path, *, n_frames: int = 1000) -> None:
    """
    Re-chunk the main detector array
    """
    with h5py.File(file_in) as fin, h5py.File(file_out, "w") as fout:
        dsname = "/entry/instrument/detector/data"

        read_ds = fin[dsname]
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


def load_det(dset: Any, n: int, m: int) -> npt.NDArray[np.uint16]:
    """
    Helper to extract a single detector from a monolith (stack)

    Parameters
    ----------
    dset : array-like
       Expected to be 3D with dimensions (time, rows, cols).

    n, m : int
       The detector to extract.

    Returns
    -------
    The (stack) for a single detector.
    """
    return cast(npt.NDArray[np.uint16], dset[:, *det_slice(n, m)])
