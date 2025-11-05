#!/usr/bin/env python3
"""
Command line tool to rechunk HDF5 files using the rechunk_in_place function.
"""

import argparse
import sys
from pathlib import Path

import tqdm

from multihead.file_io import rechunk_in_place


def main():
    parser = argparse.ArgumentParser(
        description="Rechunk HDF5 files in place for better detector access patterns"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="HDF5 files to rechunk"
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=1000,
        help="Number of frames per chunk (default: 1000)"
    )

    args = parser.parse_args()

    # Validate all files exist
    file_paths = [Path(f) for f in args.files]
    missing_files = [f for f in file_paths if not f.exists()]

    if missing_files:
        print(f"Error: The following files do not exist:")
        for f in missing_files:
            print(f"  {f}")
        sys.exit(1)

    # Process files
    processed_files: list[Path] = []
    failed_files: list[tuple[Path, Exception]] = []
    current_file: Path | None = None

    try:
        for file_path in tqdm.tqdm(file_paths, desc="Processing files"):
            try:
                current_file = file_path
                rechunk_in_place(file_path, n_frames=args.n_frames)
                current_file = None
                processed_files.append(file_path)
            except Exception as e:
                current_file = None
                failed_files.append((file_path, e))
                tqdm.tqdm.write(f"Error processing {file_path}: {e}")
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user!")
        print(f"Successfully processed {len(processed_files)} files:")
        for f in processed_files:
            print(f"  ✓ {f}")

        if current_file:
            print(f"\nProcessing interrupted while working on: {current_file}")

        not_touched = [f for f in file_paths if f not in processed_files and f != current_file]
        if not_touched:
            print(f"\nFiles not processed ({len(not_touched)}):")
            for f in not_touched:
                print(f"  - {f}")

        sys.exit(130)  # Standard exit code for SIGINT

    # Report results
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(processed_files)} files")

    if failed_files:
        print(f"Failed to process: {len(failed_files)} files")
        for file_path, error in failed_files:
            print(f"  ✗ {file_path}: {error}")
        sys.exit(1)
    else:
        print("All files processed successfully!")


if __name__ == "__main__":
    main()
