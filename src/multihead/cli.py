"""Command line interface utilities for multihead package."""

import argparse
from pathlib import Path


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
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory for data",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        help="Input filename",
        required=True,
    )
    return parser