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
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sparse
import tqdm

from multihead.cli import parse_detector_map
from multihead.file_io import open_data


def _to_jsonable(obj: Any) -> Any:
    """
    Recursively convert numpy types to JSON-serializable Python types.

    Parameters
    ----------
    obj : Any
        Object to convert.  Dicts, lists, tuples, and ndarrays are
        traversed recursively.

    Returns
    -------
    Any
        A JSON-serializable equivalent.
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Fallback: stringify
    return str(obj)


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
    detector_map=None,
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
    raw = open_data(input_path, version, detector_map=detector_map)

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
    metadata_path = file_output_dir / "metadata.json"

    # Check if files exist and handle according to on_exist policy
    files_exist = (
        images_path.exists() or scalars_path.exists() or metadata_path.exists()
    )

    if files_exist:
        if on_exist == OnExistAction.FAIL:
            raise FileExistsError(
                f"Output files already exist for {input_path.stem}. "
                f"Use --on-exist to control this behavior."
            )
        if on_exist == OnExistAction.SKIP:
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
    pq.write_table(
        images_table, images_path, compression="snappy", write_statistics=False
    )

    # Extract and write scalars (tth and monitor, plus MDA detector channels for v1)
    tth = raw.get_arm_tth()
    monitor = raw.get_monitor()
    nominal_bin = raw.get_nominal_bin()

    scalar_arrays: list[pa.Array] = [pa.array(tth), pa.array(monitor)]
    scalar_names: list[str] = ["tth", "monitor"]
    seen = {"tth", "monitor"}

    if version == 1:
        for desc, arr in raw.get_detector_scalars().items():
            arr = np.asarray(arr)
            if desc in seen:
                tqdm.tqdm.write(
                    f"  ⚠ Skipping detector scalar '{desc}' (column already present)"
                )
                continue
            if len(arr) != len(tth):
                raise ValueError(
                    f"Detector scalar '{desc}' has length {len(arr)} which does "
                    f"not match tth length {len(tth)} for {input_path.name}"
                )
            scalar_arrays.append(pa.array(arr))
            scalar_names.append(desc)
            seen.add(desc)

    scalars_table = pa.Table.from_arrays(
        scalar_arrays,
        names=scalar_names,
        metadata={"nominal_bin": str(nominal_bin)},
    )
    pq.write_table(scalars_table, scalars_path, compression="snappy")

    # Write metadata.json sidecar (v1 only).
    if version == 1:
        meta = {
            "source": {
                "version": 1,
                "input_file": str(input_path),
                "stem": stem,
            },
            "scan_md": _to_jsonable(raw.get_scan_md()),
            "scan_config": _to_jsonable(raw.get_scan_config()),
            "pre": _to_jsonable(raw.get_pre()),
            "post": _to_jsonable(raw.get_post()),
            "staff_log": _to_jsonable(raw.get_staff_log()),
        }
        with metadata_path.open("w") as fout:
            json.dump(meta, fout, indent=2, default=str)

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
    parser.add_argument(
        "--detector-map",
        type=parse_detector_map,
        help="Detector layout map as JSON list of lists. "
        "Default: '[[10, 9, 6, 5, 2, 1], [12, 11, 8, 7, 4, 3]]' (APS configuration). "
        "For single detector simulations use: '[[1]]'",
        default=None,
    )

    args = parser.parse_args()

    # Convert on_exist string to enum
    on_exist = OnExistAction(args.on_exist)

    # Process files with progress bar
    for input_file in tqdm.tqdm(args.input_files, desc="Converting files"):
        try:
            result = convert_file(
                input_file, args.output_dir, args.version, on_exist, args.detector_map
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
