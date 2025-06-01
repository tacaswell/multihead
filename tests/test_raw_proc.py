import io
import pytest
from multihead.raw_proc import SimpleSliceTuple, CrystalROI, DetectorROIs

def test_roi_yaml_roundtrip():
    # Create test data
    crystal_roi = CrystalROI(
        rslc=SimpleSliceTuple(10, 20),
        cslc=SimpleSliceTuple(30, 40)
    )
    detector_rois = DetectorROIs({
        1: crystal_roi,
        2: CrystalROI(
            rslc=SimpleSliceTuple(50, 60),
            cslc=SimpleSliceTuple(70, 80)
        )
    })

    # Serialize to YAML
    stream = io.StringIO()
    detector_rois.to_yaml(stream)

    # Reset stream position for reading
    stream.seek(0)
    print(stream.read())
    stream.seek(0)
    # Deserialize from YAML
    loaded_rois = DetectorROIs.from_yaml(stream)

    # Verify the round trip
    assert isinstance(loaded_rois, DetectorROIs)
    assert len(loaded_rois.rois) == len(detector_rois.rois)

    for key in detector_rois.rois:
        assert key in loaded_rois.rois
        original_roi = detector_rois.rois[key]
        loaded_roi = loaded_rois.rois[key]

        assert original_roi.rslc == loaded_roi.rslc
        assert original_roi.cslc == loaded_roi.cslc