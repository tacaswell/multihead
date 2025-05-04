import h5py


def rechunk(file_in, file_out, N=1000):
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


def det_slice(n, m, *, pad=4, npix=256):
    return (
        slice(n * (npix + pad), (n + 1) * npix + n * pad),
        slice(m * (npix + pad), (m + 1) * npix + m * pad),
    )


def load_det(dset, n, m):
    return dset[:, *det_slice(n, m)]
