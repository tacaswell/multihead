# Copilot Instructions for Multihead

## Project Overview
Multihead is a scientific data processing package for High Resolution Diffraction (HRD) multi-detector data from synchrotron beamlines. It processes raw detector images, computes regions of interest (ROIs), and provides interactive analysis tools.

## Core Architecture

### Data Flow Pipeline
1. **Raw Data Ingestion** (`file_io.py`): Handles HDF5 files with detector arrays, supporting both v1 (MDA+HDF5) and v2 (unified HDF5) formats
2. **Image Processing** (`raw_proc.py`): ROI detection using morphological operations on photon masks
3. **Configuration Management** (`config.py`): Dataclass-based YAML serializable configs for detector ROIs and calibration
4. **Interactive Analysis** (`demos/`): Matplotlib-based exploratory tools with widget interfaces

### Key Data Structures
- **HRDRawBase**: Abstract base for raw data access with detector mapping `{detector_id: (row, col)}`
- **DetectorROIs**: Container for crystal ROI definitions with metadata tracking
- **CrystalROI**: Simple slice tuples defining rectangular regions `(rslc, cslc)`
- **MDA**: Legacy scan metadata from EPICS sscan records (in `mda.py`)

## Development Patterns

### File Organization
- `src/multihead/`: Core library modules
- `demos/`: Standalone analysis scripts with `if __name__ == "__main__"` patterns
- ROI configs stored as YAML files (see `detector_rois.yaml`)

### Data Access Patterns
```python
# Always use the factory function for file opening
from multihead.file_io import open_data
raw = open_data(filename, ver=2)  # ver=1 for MDA+HDF5, ver=2 for unified

# Detector access follows consistent mapping
detector_data = raw.get_detector(detector_number)  # 1-indexed
sums = raw.get_detector_sums()  # All detectors at once
```

### Configuration Handling
- All configs are dataclasses with YAML serialization via `to_yaml()` methods
- ROI detection parameters: `threshold`, `opening_radius`, `closing_radius`
- Software provenance automatically tracked in `DetectorROIs.software`

### Interactive Tools Pattern
- All demo scripts follow the pattern: argument parsing → data loading → matplotlib UI setup
- Use `get_base_parser()` from `cli.py` for consistent CLI interfaces
- Widget-based interactions (buttons, sliders, selectors) for parameter tuning

## Build and Test Workflow

### Environment Setup
```bash
# Use pixi for dependency management (preferred)
pixi install
pixi shell

# Or traditional pip install
pip install -e .[dev]
```

### Testing and Linting
```bash
# Run all checks (uses nox)
nox

# Individual sessions
nox -s tests      # pytest
nox -s lint       # pre-commit hooks  
nox -s pylint     # static analysis
```

### Demo Scripts
```bash
# Interactive ROI masking tool
python demos/mask.py -f data.h5 --ver 2

# Image scrubber for frame-by-frame analysis
python demos/scrubber.py -f data.h5 --ver 2

# Kymograph generation
python demos/khymographs.py -f data.h5 --ver 2
```

## Scientific Context

### Detector Layout
- 12 detectors arranged in 2×6 grid (see `HRDRawBase._detector_map`)
- Each detector: 256×256 pixels
- Detectors numbered 1-12, mapped to (row, col) positions

### ROI Detection Algorithm
1. Apply threshold to photon sum images
2. Morphological opening (noise removal) then closing (gap filling)
3. Find largest connected component
4. Extract bounding rectangle as ROI
5. Parameters tunable via interactive tools

### File Format Evolution
- **v1**: Separate MDA metadata + HDF5 images (legacy EPICS scans)  
- **v2**: Unified HDF5 with embedded metadata (modern format)
- Always specify version explicitly when opening files

## Key Integrations
- **HDF5/h5py**: Primary data storage (requires `hdf5plugin` for compression)
- **scikit-image**: Morphological operations and image processing
- **matplotlib**: All visualization and interactive tools
- **YAML**: Human-readable configuration serialization
- **tqdm**: Progress bars for long data operations