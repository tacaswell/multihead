"""Command line interface utilities for multihead package."""

import argparse


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
    return parser
