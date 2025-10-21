# Multihead Tiled Integration

This module provides Tiled adapters and utilities for serving multihead high-resolution diffraction (HRD) data through the Tiled data access framework.

## Components

### `tiled.py` - HRDRawAdapter

The main Tiled adapter that exposes `HRDRawBase` objects as structured data trees.

**Features:**
- Individual detector data access (`detector_1`, `detector_2`, ..., `detector_12`)
- Detector sum data (`detector_1_sum`, `detector_2_sum`, ..., `detector_12_sum`)
- Two-theta angle values (`tth`)
- Monitor data (`monitor`)
- Rich metadata extraction
- Support for both HRDRawV1 and HRDRawV2 formats

**Usage:**
```python
from multihead.tiled.tiled import HRDRawAdapter

# Create adapter
adapter = HRDRawAdapter.from_file_path("data.h5", version=2)

# Access data
detector_1 = adapter["detector_1"]  # 3D array (time, y, x)
tth_values = adapter["tth"]         # 1D array of angles
metadata = adapter.metadata()      # Rich metadata dict
```

### `register_hrd_data.py` - Data Discovery and Registration

Command-line tool and functions for automatically discovering HRD data files and registering them with a Tiled server.

**Features:**
- Automatic detection of HRD data format versions
- File system walking and discovery
- Metadata analysis and grouping
- Batch registration with Tiled server
- Run-based organization

**Command Line Usage:**
```bash
python -m multihead.tiled.register_hrd_data \
    /path/to/data \
    http://localhost:8000 \
    --api-key secret \
    --collection hrd_data \
    --clear
```

**Programmatic Usage:**
```python
from multihead.tiled.register_hrd_data import register_hrd_data_with_tiled

register_hrd_data_with_tiled(
    base_path=Path("/path/to/data"),
    tiled_uri="http://localhost:8000",
    api_key="secret",
    collection_name="hrd_data"
)
```

### `example_register.py` - Example Script

Simple example showing how to register HRD data with Tiled.

## Data Organization

The registration process organizes data as follows:

```
tiled_server/
└── collection_name/
    └── directory_run/          # e.g., "sample1_001"
        ├── 0/                  # First measurement
        ├── 1/                  # Second measurement
        └── ...
```

Each measurement contains:
- Individual detector arrays (12 detectors)
- Detector sum arrays (12 sums)
- Two-theta angle data
- Monitor data
- Rich metadata including scan parameters

## Metadata Structure

The adapter extracts and organizes metadata including:

- **Detector configuration**: Mapping and positions
- **Scan parameters**: Start/stop angles, step size, timing
- **Sample information**: Name, composition, temperature
- **Instrument settings**: Monitor values, ring current
- **Run organization**: Varied vs. static parameters

## Requirements

- `tiled` - Data access framework
- `h5py` - HDF5 file access
- `numpy` - Array operations
- `pathlib` - Path manipulation

## Installation

The module is part of the multihead package. Install with:

```bash
pip install -e /path/to/multihead
```

## Example Workflow

1. **Start Tiled server** (if not already running):
   ```bash
   tiled serve --api-key secret
   ```

2. **Register data**:
   ```bash
   python -m multihead.tiled.register_hrd_data \
       /beamline/data/2024/cycle1 \
       http://localhost:8000 \
       --api-key secret
   ```

3. **Access data via Tiled client**:
   ```python
   from tiled.client import from_uri

   client = from_uri("http://localhost:8000", api_key="secret")
   data = client["hrd_data"]["sample1_001"]["0"]

   # Get detector 1 data
   det1 = data["detector_1"][:]

   # Get two-theta values
   tth = data["tth"][:]

   # Access metadata
   metadata = data.metadata
   ```

This integration enables seamless access to HRD diffraction data through Tiled's web API, making the data discoverable and accessible to analysis tools and workflows.