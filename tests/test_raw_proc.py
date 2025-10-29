import io

import numpy as np

from multihead.config import CrystalROI, SimpleSliceTuple
from multihead.raw_proc import DetectorROIs, compute_rois


def test_roi_yaml_roundtrip():
    # Create test data
    crystal_roi = CrystalROI(
        rslc=SimpleSliceTuple(10, 20), cslc=SimpleSliceTuple(30, 40)
    )
    detector_rois = DetectorROIs(
        rois={
            1: crystal_roi,
            2: CrystalROI(rslc=SimpleSliceTuple(50, 60), cslc=SimpleSliceTuple(70, 80)),
        },
        software={
            "name": "multihead-test",
            "version": "test-version",
            "function": "test_function",
            "module": "test_module",
        },
        parameters={"threshold": 2, "closing_radius": 10, "opening_radius": 10},
    )

    # Serialize to YAML
    stream = io.StringIO()
    detector_rois.to_yaml(stream)

    # Reset stream position for reading
    stream.seek(0)
    # Deserialize from YAML
    loaded_rois = DetectorROIs.from_yaml(stream)

    # Verify the round trip
    assert isinstance(loaded_rois, DetectorROIs)
    assert len(loaded_rois.rois) == len(detector_rois.rois)
    assert loaded_rois.software == detector_rois.software
    assert loaded_rois.parameters == detector_rois.parameters

    for key in detector_rois.rois:
        assert key in loaded_rois.rois
        original_roi = detector_rois.rois[key]
        loaded_roi = loaded_rois.rois[key]

        assert original_roi.rslc == loaded_roi.rslc
        assert original_roi.cslc == loaded_roi.cslc


def test_compute_rois_metadata():
    # Create dummy data
    sums = {
        1: np.zeros((100, 100), dtype=np.int32),
        2: np.zeros((100, 100), dtype=np.int32),
    }
    # Add some signal to trigger ROI detection
    sums[1][10:20, 30:40] = 10
    sums[2][50:60, 70:80] = 10

    # Set specific parameters
    th = 5
    closing_radius = 15
    opening_radius = 8

    # Compute ROIs
    detector_rois = compute_rois(
        sums, th=th, closing_radius=closing_radius, opening_radius=opening_radius
    )

    # Verify software metadata
    assert isinstance(detector_rois.software, dict)
    assert detector_rois.software["name"] == "multihead"
    assert "version" in detector_rois.software
    assert detector_rois.software["function"] == "compute_rois"
    assert detector_rois.software["module"] == "multihead.raw_proc"

    # Verify parameters
    assert detector_rois.parameters["threshold"] == th
    assert detector_rois.parameters["closing_radius"] == closing_radius
    assert detector_rois.parameters["opening_radius"] == opening_radius
