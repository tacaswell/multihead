"""
Tiled adapter for multihead HRD raw data.

This module provides a custom Tiled adapter that exposes HRDRawBase objects
as a structured data tree. The adapter allows accessing:

- Individual detector data (detector_1, detector_2, ..., detector_12)
- Detector sum data (detector_1_sum, detector_2_sum, ..., detector_12_sum)
- Two-theta angle values (tth)
- Monitor data (monitor)

The adapter supports both version 1 (HRDRawV1) and version 2 (HRDRawV2)
data formats and automatically extracts metadata from the underlying files.

Example usage:
    # For version 2 data
    adapter = HRDRawAdapter.from_file_path("data.h5", version=2)

    # For version 1 data (requires both .mda and .h5 files)
    adapter = HRDRawAdapter.from_file_path("data", version=1)

    # Access detector data
    detector_1 = adapter["detector_1"]
    detector_sums = adapter["detector_1_sum"]
    tth_values = adapter["tth"]
"""

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Self

import numpy
from tiled.adapters.array import ArrayAdapter
from tiled.adapters.utils import IndexersMixin
from tiled.catalog.orm import Node
from tiled.iterviews import ItemsView, KeysView, ValuesView
from tiled.structures.array import ArrayStructure
from tiled.structures.core import Spec, StructureFamily
from tiled.structures.data_source import DataSource
from tiled.type_aliases import JSON
from tiled.utils import node_repr, path_from_uri

from ..file_io import HRDRawBase, open_data

SWMR_DEFAULT = bool(int(os.getenv("TILED_HDF5_SWMR_DEFAULT", "0")))
INLINED_DEPTH = int(os.getenv("TILED_HDF5_INLINED_CONTENTS_MAX_DEPTH", "7"))


class HRDRawAdapter(Mapping[str, ArrayAdapter], IndexersMixin):
    """
    Tiled adapter for HRDRawBase objects.

    This adapter exposes high-resolution diffraction raw data through the HRDRawBase
    interface, providing access to individual detectors, detector sums, TTH values,
    and monitor data as array structures.

    Examples
    --------

    From a file path for version 1 data (requires both .mda and .h5 files):

    >>> HRDRawAdapter.from_uris("file://localhost/path/to/file", version=1)

    From a file path for version 2 data:

    >>> HRDRawAdapter.from_uris("file://localhost/path/to/file.h5", version=2)

    """

    structure_family = StructureFamily.container

    def __init__(
        self,
        hrd_raw: HRDRawBase,
        *,
        structure: ArrayStructure | None = None,  # noqa: ARG002
        metadata: JSON | None = None,
        specs: list[Spec] | None = None,
    ) -> None:
        self._hrd_raw = hrd_raw
        self.specs = specs or []
        self._metadata = metadata or {}

    @classmethod
    def from_catalog(
        cls,
        data_source: DataSource,
        node: Node,
        /,
        version: int = 2,
        **kwargs: Any | None,
    ) -> "HRDRawAdapter":
        """Create adapter from catalog data source."""
        assets = data_source.assets
        data_uri = None
        if len(assets) == 1:
            data_uri = assets[0].data_uri
        else:
            for ast in assets:
                if ast.parameter == "data_uri":
                    data_uri = ast.data_uri
                    break

        if data_uri is None:
            raise ValueError("No data_uri found in assets")

        filepath = path_from_uri(data_uri)

        # Open the HRDRaw data object
        hrd_raw = open_data(filepath, version=version)

        adapter = cls(
            hrd_raw,
            structure=data_source.structure,
            metadata=node.metadata_,
            specs=node.specs,
        )

        # Handle dataset navigation if specified
        dataset = kwargs.get("dataset") or kwargs.get("path") or []
        for segment in dataset:
            adapter = adapter.get(segment)
            if adapter is None:
                raise KeyError(segment)

        return adapter

    @classmethod
    def from_uris(
        cls,
        data_uri: str,
        *,
        version: int = 2,
        **kwargs: Any,
    ) -> "HRDRawAdapter":
        """Create adapter from URI."""
        filepath = path_from_uri(data_uri)

        # Open the HRDRaw data object
        hrd_raw = open_data(filepath, version=version)

        return cls(hrd_raw, **kwargs)

    @classmethod
    def from_file_path(
        cls,
        file_path: str | Path,
        *,
        version: int = 2,
        **kwargs: Any,
    ) -> "HRDRawAdapter":
        """Create adapter from file path."""
        hrd_raw = open_data(file_path, version=version)
        return cls(hrd_raw, **kwargs)

    def __repr__(self) -> str:
        return node_repr(self, list(self))

    def structure(self) -> None:
        return None

    def metadata(self) -> JSON:
        """Return metadata including detector configuration and scan parameters."""
        base_metadata = {**self._metadata}

        # Add detector map information
        base_metadata["detector_map"] = self._hrd_raw._detector_map
        base_metadata["nominal_bin_size"] = self._hrd_raw.get_nominal_bin()

        # Add version-specific metadata with proper type checking
        if hasattr(self._hrd_raw, '_mda'):
            # Version 1 metadata (HRDRawV1)
            mda = getattr(self._hrd_raw, '_mda', None)
            if mda is not None:
                base_metadata["scan_metadata"] = mda.scan_md
                base_metadata["scan_config"] = {
                    k: {"value": v.value, "unit": v.unit, "pv": getattr(v, 'pv', None)}
                    for k, v in mda.scan_config.items()
                }
        elif hasattr(self._hrd_raw, '_md'):
            # Version 2 metadata (HRDRawV2)
            md = getattr(self._hrd_raw, '_md', None)
            if md is not None:
                base_metadata["scan_metadata"] = {
                    k: {"value": v.value, "unit": v.unit}
                    for k, v in md.items()
                }

        return base_metadata

    def __iter__(self) -> Iterator[str]:
        """Iterate over available data keys."""
        # Individual detectors
        for det_num in self._hrd_raw._detector_map:
            yield f"detector_{det_num}"

        # Detector sums
        for det_num in self._hrd_raw._detector_map:
            yield f"detector_{det_num}_sum"

        # Other data
        yield "tth"
        yield "monitor"

    def __getitem__(self, key: str) -> ArrayAdapter:
        """Get array adapter for the specified key."""
        try:
            if key.startswith("detector_") and key.endswith("_sum"):
                # Detector sum data
                det_num = int(key.split("_")[1])
                detector_sums = self._hrd_raw.get_detector_sums()
                data = detector_sums[det_num]
                md = {
                    "detector_number": det_num,
                    "data_type": "detector_sum",
                    "detector_position": self._hrd_raw._detector_map[det_num],
                }
                return ArrayAdapter.from_array(data, metadata=md)

            elif key.startswith("detector_"):
                # Individual detector data
                det_num = int(key.split("_")[1])
                data = self._hrd_raw.get_detector(det_num)
                md = {
                    "detector_number": det_num,
                    "data_type": "detector_frames",
                    "detector_position": self._hrd_raw._detector_map[det_num],
                }
                # Use chunking for time series data
                chunks = (min(1000, data.shape[0]), *data.shape[1:])
                return ArrayAdapter.from_array(data, metadata=md, chunks=chunks)

            elif key == "tth":
                # Two-theta values
                data = self._hrd_raw.get_arm_tth()
                md = {
                    "data_type": "two_theta",
                    "units": "degrees",
                    "nominal_bin_size": self._hrd_raw.get_nominal_bin(),
                }
                return ArrayAdapter.from_array(data, metadata=md)

            elif key == "monitor":
                # Monitor data
                data = self._hrd_raw.get_monitor()
                md = {"data_type": "monitor"}

                # Handle different monitor data formats
                if isinstance(data, dict):
                    # Version 1: return first available monitor
                    if data:
                        first_monitor = next(iter(data.values()))
                        md["monitor_name"] = next(iter(data.keys()))
                        return ArrayAdapter.from_array(first_monitor, metadata=md)
                    else:
                        # No monitor data available
                        return ArrayAdapter.from_array(numpy.array([]), metadata=md)
                else:
                    # Version 2: direct array
                    return ArrayAdapter.from_array(data, metadata=md)

            else:
                raise KeyError(f"Unknown key: {key}")

        except (IndexError, ValueError, KeyError) as e:
            raise KeyError(f"Invalid key '{key}': {e}") from e

    def __len__(self) -> int:
        """Return number of available data items."""
        # Count detectors, detector sums, tth, and monitor
        n_detectors = len(self._hrd_raw._detector_map)
        return n_detectors * 2 + 2  # detectors + sums + tth + monitor

    def keys(self) -> KeysView:
        return KeysView(lambda: len(self), self._keys_slice)

    def values(self) -> ValuesView:
        return ValuesView(lambda: len(self), self._items_slice)

    def items(self) -> ItemsView:
        return ItemsView(lambda: len(self), self._items_slice)

    def search(self, query: Any) -> None:
        """
        Search functionality - not implemented yet.

        Parameters
        ----------
        query : Any
            Search query

        Returns
        -------
        None
            Returns a Tree with a subset of the mapping.
        """
        raise NotImplementedError("Search functionality not yet implemented")

    def read(self, fields: str | None = None) -> Self:
        """
        Read data with optional field selection.

        Parameters
        ----------
        fields : str, optional
            Specific fields to read

        Returns
        -------
        Self
            The adapter instance
        """
        if fields is not None:
            raise NotImplementedError("Field selection not yet implemented")
        return self

    def _keys_slice(self, start: int, stop: int, direction: int) -> list[str]:
        """
        Get a slice of keys for iteration.

        Parameters
        ----------
        start : int
            Start index
        stop : int
            Stop index
        direction : int
            Direction of iteration

        Returns
        -------
        list[str]
            List of keys
        """
        keys = list(self)
        if direction < 0:
            keys = list(reversed(keys))
        return keys[start:stop]

    def _items_slice(
        self, start: int, stop: int, direction: int
    ) -> list[tuple[str, ArrayAdapter]]:
        """
        Get a slice of items for iteration.

        Parameters
        ----------
        start : int
            Start index
        stop : int
            Stop index
        direction : int
            Direction of iteration

        Returns
        -------
        list[tuple[str, ArrayAdapter]]
            List of (key, value) pairs
        """
        items = [(key, self[key]) for key in list(self)]
        if direction < 0:
            items = list(reversed(items))
        return items[start:stop]

    def inlined_contents_enabled(self, depth: int) -> bool:
        """Check if inlined contents are enabled at the given depth."""
        return depth <= INLINED_DEPTH


def read_hrd_raw(
    filepath: str | Path,
    version: int = 2,
    metadata: JSON | None = None,
    **kwargs: Any,
) -> HRDRawAdapter:
    """
    Convenience function to create an HRDRawAdapter.

    Parameters
    ----------
    filepath : str or Path
        Path to the data file(s)
    version : int, default=2
        Version of the HRD raw data format (1 or 2)
    metadata : dict, optional
        Additional metadata to include
    **kwargs
        Additional arguments passed to the adapter

    Returns
    -------
    HRDRawAdapter
        Configured adapter instance
    """
    return HRDRawAdapter.from_file_path(
        filepath, version=version, metadata=metadata, **kwargs
    )
