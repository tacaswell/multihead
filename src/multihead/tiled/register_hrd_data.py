"""
HRD data discovery and registration for Tiled.

This module provides functions to walk the filesystem, discover HRD data files,
and register them with a Tiled server. It handles both version 1 (.mda + .h5)
and version 2 (.h5) data formats.

Example usage:
    python register_hrd_data.py /path/to/data/root http://localhost:8000 secret
"""

import argparse
import pathlib
import sys
from collections import defaultdict
from typing import Any

from tiled.client import from_uri
from tiled.client.register import dict_or_none, ensure_uri
from tiled.structures.core import StructureFamily
from tiled.structures.data_source import Asset, DataSource, Management

from .tiled import HRDRawAdapter


def detect_hrd_data_version(file_path: pathlib.Path) -> int | None:
    """
    Detect the version of HRD data based on file patterns.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to examine for HRD data

    Returns
    -------
    int or None
        Version number (1 or 2) if HRD data detected, None otherwise
    """
    if file_path.suffix == ".h5":
        # Check if corresponding .mda file exists (version 1)
        mda_path = file_path.with_suffix(".mda")
        if mda_path.exists():
            return 1
        else:
            # Standalone .h5 file (version 2)
            return 2
    elif file_path.suffix == ".mda":
        # Check if corresponding .h5 file exists (version 1)
        h5_path = file_path.with_suffix(".h5")
        if h5_path.exists():
            return 1

    return None


def discover_hrd_files(base_path: pathlib.Path) -> dict[str, list[tuple[pathlib.Path, int]]]:
    """
    Discover HRD data files in a directory tree.

    Parameters
    ----------
    base_path : pathlib.Path
        Root directory to search for HRD data

    Returns
    -------
    Dict[str, List[Tuple[pathlib.Path, int]]]
        Dictionary mapping directory names to lists of (file_path, version) tuples
    """
    hrd_files_by_dir = defaultdict(list)

    for root_dir in base_path.iterdir():
        if not root_dir.is_dir():
            continue

        # Look for HRD data files in this directory
        processed_stems = set()

        for file_path in root_dir.iterdir():
            if not file_path.is_file():
                continue

            # Skip if we've already processed this stem
            if file_path.stem in processed_stems:
                continue

            version = detect_hrd_data_version(file_path)
            if version is not None:
                if version == 1:
                    # For version 1, use the stem (without extension) as the key
                    hrd_files_by_dir[root_dir.name].append((file_path.with_suffix(""), version))
                else:
                    # For version 2, use the full .h5 file
                    hrd_files_by_dir[root_dir.name].append((file_path, version))
                processed_stems.add(file_path.stem)

    return dict(hrd_files_by_dir)


def extract_run_info(file_path: pathlib.Path) -> tuple[str, str]:
    """
    Extract run number and object ID from file name.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the data file

    Returns
    -------
    Tuple[str, str]
        (run_number, object_id) tuple
    """
    stem = file_path.stem
    if "-" in stem:
        run, _, oid = stem.partition("-")
        return run, oid
    else:
        # If no dash, treat entire stem as run number with oid "0"
        return stem, "0"


def find_varied_metadata_keys(metadata_list: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """
    Find metadata keys that vary across a collection of datasets.

    Parameters
    ----------
    metadata_list : List[Dict[str, Any]]
        List of metadata dictionaries

    Returns
    -------
    List[Tuple[str, str]]
        List of (section, key) tuples for varying metadata
    """
    if not metadata_list:
        return []

    varied_keys = []
    first_metadata = metadata_list[0]

    # Compare all metadata entries to find varying keys
    for section, section_data in first_metadata.items():
        if not isinstance(section_data, dict):
            continue

        for key, _ in section_data.items():
            # Check if this key varies across datasets
            values = []
            for metadata in metadata_list:
                if section in metadata and key in metadata[section]:
                    values.append(metadata[section][key])
                else:
                    values.append(None)

            # If not all values are the same, this key varies
            if len({str(v) for v in values}) > 1:
                varied_keys.append((section, key))

    return varied_keys


def register_hrd_data_with_tiled(
    base_path: pathlib.Path,
    tiled_uri: str,
    api_key: str | None = None,
    collection_name: str = "hrd_data",
    clear_existing: bool = False,
) -> None:
    """
    Discover and register HRD data with a Tiled server.

    Parameters
    ----------
    base_path : pathlib.Path
        Root directory containing HRD data
    tiled_uri : str
        URI of the Tiled server
    api_key : str, optional
        API key for authentication
    collection_name : str, default="hrd_data"
        Name of the collection to create in Tiled
    clear_existing : bool, default=False
        Whether to clear existing data in the collection
    """
    # Connect to Tiled server
    client = from_uri(tiled_uri, api_key=api_key)

    # Create or access the collection
    if collection_name in client:
        collection = client[collection_name]
        if clear_existing:
            collection.delete_tree()
            collection = client.new(
                key=collection_name,
                structure_family=StructureFamily.container,
                metadata={"description": "HRD raw diffraction data"},
                data_sources=[],
            )
    else:
        collection = client.new(
            key=collection_name,
            structure_family=StructureFamily.container,
            metadata={"description": "HRD raw diffraction data"},
            data_sources=[],
        )

    # Discover HRD files
    hrd_files_by_dir = discover_hrd_files(base_path)

    for dir_name, file_list in hrd_files_by_dir.items():
        print(f"Processing directory: {dir_name}")

        # Group files by run number
        tiled_objs_by_run = defaultdict(list)

        for file_path, version in file_list:
            try:
                # Create data source
                f_uri = ensure_uri(file_path)

                # Create adapter to validate and extract metadata
                adapter = HRDRawAdapter.from_file_path(file_path, version=version)

                ds = DataSource(
                    structure_family=adapter.structure_family,
                    mimetype="application/x-hdf5" if version == 2 else "application/x-multipart",
                    structure=dict_or_none(adapter.structure()),
                    parameters={"version": version},
                    management=Management.external,
                    assets=[
                        Asset(
                            data_uri=f_uri,
                            is_directory=False,
                            parameter="data_uri",
                        )
                    ],
                )

                # Check if data is valid (has tth data)
                try:
                    tth_data = adapter["tth"]
                    if hasattr(tth_data, 'structure') and tth_data.structure().shape[0] == 0:
                        print(f"  Skipping {file_path.name}: empty tth data")
                        continue
                except Exception as e:
                    print(f"  Skipping {file_path.name}: error accessing tth data - {e}")
                    continue

                # Extract run and object ID
                run, oid = extract_run_info(file_path)
                tiled_objs_by_run[run].append((adapter, ds, oid))

            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
                continue

        # Register each run
        for run, results in tiled_objs_by_run.items():
            if not results:
                continue

            print(f"  Registering run: {run} ({len(results)} datasets)")

            # Sort by object ID
            results.sort(key=lambda x: int(x[-1]) if x[-1].isdigit() else 0)

            # Extract metadata from all datasets in this run
            all_metadata = [r[0].metadata() for r in results]

            # Find varying metadata keys
            varied_keys = find_varied_metadata_keys(all_metadata)

            # Build static metadata (keys that don't vary)
            static_metadata = {}
            if all_metadata:
                first_metadata = all_metadata[0]
                for section, section_data in first_metadata.items():
                    if isinstance(section_data, dict):
                        static_metadata[section] = {}
                        for key, value in section_data.items():
                            if (section, key) not in varied_keys:
                                static_metadata[section][key] = value
                    else:
                        # Non-dict metadata is always static
                        if section not in [k[0] for k in varied_keys]:
                            static_metadata[section] = section_data

            # Build varying metadata
            varied_values = {}
            for (_, _, oid), metadata in zip(results, all_metadata, strict=True):
                varied_values[oid] = {}
                for section, key in varied_keys:
                    if section in metadata and key in metadata[section]:
                        varied_values[oid].setdefault(section, {})
                        varied_values[oid][section][key] = metadata[section][key]

            # Create description
            if varied_keys:
                description = f"Run {run}: Varied {['.'.join(k) for k in varied_keys]}"
            else:
                description = f"Run {run}: Static parameters"

            # Create run collection
            try:
                run_key = f"{dir_name}_{run}"
                run_collection = collection.new(
                    key=run_key,
                    structure_family=StructureFamily.container,
                    metadata={
                        "run_number": run,
                        "directory": dir_name,
                        "varied_keys": [".".join(k) for k in varied_keys],
                        "static_metadata": static_metadata,
                        "varied_values": varied_values,
                        "description": description,
                        "num_datasets": len(results),
                    },
                    data_sources=[],
                )

                # Register individual datasets
                for adapter, ds, oid in results:
                    run_collection.new(
                        structure_family=adapter.structure_family,
                        data_sources=[ds],
                        metadata=adapter.metadata(),
                        key=oid,
                    )

                print(f"    Registered {len(results)} datasets for run {run}")

            except Exception as e:
                print(f"    Error registering run {run}: {e}")
                continue


def main():
    """Main entry point for the registration script."""
    parser = argparse.ArgumentParser(
        description="Discover and register HRD data with Tiled"
    )
    parser.add_argument(
        "data_path",
        type=pathlib.Path,
        help="Root directory containing HRD data",
    )
    parser.add_argument(
        "tiled_uri",
        help="URI of the Tiled server (e.g., http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for Tiled authentication",
    )
    parser.add_argument(
        "--collection",
        default="hrd_data",
        help="Name of the collection to create in Tiled (default: hrd_data)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data in the collection",
    )

    args = parser.parse_args()

    if not args.data_path.exists():
        print(f"Error: Data path {args.data_path} does not exist")
        sys.exit(1)

    try:
        register_hrd_data_with_tiled(
            base_path=args.data_path,
            tiled_uri=args.tiled_uri,
            api_key=args.api_key,
            collection_name=args.collection,
            clear_existing=args.clear,
        )
        print("HRD data registration completed successfully!")

    except Exception as e:
        print(f"Error during registration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()