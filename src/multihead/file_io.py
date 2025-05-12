"""
Functions to read in raw data and extract single detectors.
"""

from pathlib import Path
from typing import Any

import h5py


def rechunk(file_in: str | Path, file_out: str | Path, N: int = 1000) -> None:
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
            chunks=(N, 260, 260),
            compression=32008,
            compression_opts=(block_size, 2),
            dtype=read_ds.dtype,
        )

        for j in range(len(read_ds) // N + 1):
            dataset[j * N : (j + 1) * N] = read_ds[j * N : (j + 1) * N]

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


def load_det(dset: Any, n: int, m: int) -> Any:
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
    return dset[:, *det_slice(n, m)]
