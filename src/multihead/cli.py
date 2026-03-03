"""Command line interface utilities for multihead package."""

import argparse
import json
from collections.abc import Sequence


def parse_detector_map(value: str) -> Sequence[Sequence[int]]:
    """Parse detector map from JSON string.

    Parameters
    ----------
    value : str
        JSON-formatted string representing detector map,
        e.g., '[[10, 9, 6, 5, 2, 1], [12, 11, 8, 7, 4, 3]]'
        or for single detector: '[[1]]'

    Returns
    -------
    Sequence[Sequence[int]]
        Parsed detector map as nested sequences

    Raises
    ------
    argparse.ArgumentTypeError
        If the string cannot be parsed as valid detector map JSON
    """
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("Detector map must be a list of lists")
        for row in parsed:
            if not isinstance(row, list):
                raise ValueError("Each row in detector map must be a list")
            for det in row:
                if not isinstance(det, int):
                    raise ValueError("Detector numbers must be integers")
        return tuple(tuple(row) for row in parsed)
    except (json.JSONDecodeError, ValueError) as e:
        raise argparse.ArgumentTypeError(
            f"Invalid detector map format: {e}. "
            "Expected JSON list of lists, e.g., '[[10, 9, 6, 5, 2, 1], [12, 11, 8, 7, 4, 3]]'"
        ) from e


def get_base_parser(description: str | None = None) -> argparse.ArgumentParser:
    """Create a base argument parser with common arguments.

    Parameters
    ----------
    description : str, optional
        Description of the program for help text

    Returns
    -------
    argparse.ArgumentParser
        Parser with common arguments added
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-f", "--filename", type=str, help="Input filename", required=True
    )
    parser.add_argument("--ver", type=int, help="file schema version", default=2)
    parser.add_argument(
        "--detector-map",
        type=parse_detector_map,
        help="Detector layout map as JSON list of lists. "
        "Default: '[[10, 9, 6, 5, 2, 1], [12, 11, 8, 7, 4, 3]]' (APS configuration). "
        "For single detector simulations use: '[[1]]'",
        default=None,
    )
    return parser
