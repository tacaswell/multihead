"""
Convert HDF5 raw data files to sparse parquet format.

This script converts raw detector data and associated metadata (tth, monitor)
from HDF5 files into parquet files with sparse array representation.
"""
import argparse
import json
import sys
from enum import Enum
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import sparse
import tqdm

from multihead.file_io import open_data


class OnExistAction(Enum):
    """Actions to take when output files already exist."""
    FAIL = "fail"
    SKIP = "skip"
    WARN_OVERWRITE = "warn-overwrite"
    OVERWRITE = "overwrite"


def convert_file(
    input_path: Path,
    output_dir: Path,
    version: int,
    on_exist: OnExistAction,
) -> tuple[Path, Path] | None:
    """
    Convert a single HDF5 file to parquet format.

    Parameters
    ----------
    input_path : Path
        Path to the input HDF5 file
    output_dir : Path
        Directory where output parquet files will be written
    version : int
        Version of the file format (1 or 2)

    Returns
    -------
    images_path : Path
        Path to the sparse detector images parquet file
    scalars_path : Path
        Path to the scalars (tth, monitor) parquet file
    """
    # Open the raw data
    raw = open_data(input_path, version)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output file names based on input file
    stem = input_path.stem

    # Create output directory for this file
    file_output_dir = output_dir / stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output file names with consistent naming
    images_path = file_output_dir / "images.parquet"
    scalars_path = file_output_dir / "scalars.parquet"

    # Check if files exist and handle according to on_exist policy
    files_exist = images_path.exists() or scalars_path.exists()

    if files_exist:
        if on_exist == OnExistAction.FAIL:
            raise FileExistsError(
                f"Output files already exist for {input_path.stem}. "
                f"Use --on-exist to control this behavior."
            )
        elif on_exist == OnExistAction.SKIP:
            tqdm.tqdm.write(f"⚠ Skipping {input_path.name} (output exists)")
            return None
        elif on_exist == OnExistAction.WARN_OVERWRITE:
            tqdm.tqdm.write(f"⚠ Overwriting existing files for {input_path.name}")

    # Convert detector data to sparse format
    sparse_data = {
        detector_num: sparse.COO(raw.get_detector(detector_num))
        for detector_num in range(1, 13)
    }

    # Stack all detectors into a single sparse array
    all_data = cast(sparse.COO, sparse.stack(list(sparse_data.values())))

    # Write sparse detector images to parquet
    images_table = pa.Table.from_arrays(
        [*all_data.coords, all_data.data],
        names=("detector", "frame", "row", "col", "data"),
        metadata={"shape": json.dumps(all_data.shape)},
    )
    pq.write_table(images_table, images_path, compression="snappy", write_statistics=False)

    # Extract and write scalars (tth and monitor)
    tth = raw.get_arm_tth()
    monitor = raw.get_monitor()
    nominal_bin = raw.get_nominal_bin()

    scalars_table = pa.Table.from_arrays(
        [pa.array(tth), pa.array(monitor)],
        names=("tth", "monitor"),
        metadata={"nominal_bin": str(nominal_bin)},
    )
    pq.write_table(scalars_table, scalars_path, compression="snappy")

    return images_path, scalars_path


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Convert HDF5 raw data files to sparse parquet format"
    )
    parser.add_argument(
        "input_files",
        type=Path,
        nargs="+",
        help="Input HDF5 file(s) to convert",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=int,
        choices=[1, 2],
        default=2,
        help="File format version (default: 2)",
    )
    parser.add_argument(
        "--on-exist",
        type=str,
        choices=[e.value for e in OnExistAction],
        default=OnExistAction.FAIL.value,
        help=(
            "Action when output files exist: "
            "'fail' (default, raise error), "
            "'skip' (skip with warning), "
            "'warn-overwrite' (overwrite with warning), "
            "'overwrite' (silently overwrite)"
        ),
    )

    args = parser.parse_args()

    # Convert on_exist string to enum
    on_exist = OnExistAction(args.on_exist)

    # Process files with progress bar
    for input_file in tqdm.tqdm(args.input_files, desc="Converting files"):
        try:
            result = convert_file(
                input_file, args.output_dir, args.version, on_exist
            )
            if result is not None:
                images_path, scalars_path = result
                tqdm.tqdm.write(f"✓ Converted {input_file.name}")
                tqdm.tqdm.write(f"  → Images: {images_path}")
                tqdm.tqdm.write(f"  → Scalars: {scalars_path}")
        except FileExistsError as e:
            tqdm.tqdm.write(f"✗ {e}")
            sys.exit(1)
        except Exception as e:
            tqdm.tqdm.write(f"✗ Failed to convert {input_file.name}: {e}")


if __name__ == "__main__":
    main()
