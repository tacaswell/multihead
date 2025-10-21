#!/usr/bin/env python3
"""
Example script for registering HRD data with Tiled.

This script demonstrates how to use the HRD data registration functionality
to discover and register diffraction data with a Tiled server.

Usage:
    python example_register.py /path/to/hrd/data
"""

import pathlib
import sys

from multihead.tiled.register_hrd_data import register_hrd_data_with_tiled


def main():
    """Example of registering HRD data with Tiled."""
    if len(sys.argv) != 2:
        print("Usage: python example_register.py /path/to/hrd/data")
        sys.exit(1)

    data_path = pathlib.Path(sys.argv[1])

    # Configuration
    tiled_server = "http://localhost:8000"
    api_key = "secret"  # Replace with your actual API key
    collection_name = "hrd_raw_data"

    print(f"Registering HRD data from: {data_path}")
    print(f"Tiled server: {tiled_server}")
    print(f"Collection: {collection_name}")
    print("-" * 50)

    try:
        register_hrd_data_with_tiled(
            base_path=data_path,
            tiled_uri=tiled_server,
            api_key=api_key,
            collection_name=collection_name,
            clear_existing=True,  # Clear existing data for clean start
        )

        print("\nRegistration completed successfully!")
        print(f"Data is now available at: {tiled_server}/{collection_name}")

    except Exception as e:
        print(f"Error during registration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()