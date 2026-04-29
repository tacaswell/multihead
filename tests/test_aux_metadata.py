"""Tests for aux_metadata parsers."""

from __future__ import annotations

from pathlib import Path

import pytest

from multihead import aux_metadata


PRE_SAMPLE = """\
# autosave R5.3\tAutomatically generated - DO NOT MODIFY - 251022-192055
! 1 channel(s) not connected - or not all gets were successful
11bmb:m28.RBV -6
11bmb:HeidND261_read.VAL 354.04527
11bmb:AShutter:LastOnTime 2025/10/21 11:46:58.162
11bmb:BeamOn 1
#11bmb:m45.RBV Search Issued
11bmb:S:IOC:timeOfDayForm2SI 10/22/25 19:20:55
<END>
"""

STAFF_LOG_SAMPLE = """\
*********** Run [2025-10-22 17:21:45] ***********
[2025-10-22 17:21:45] Step 1: UR5 Load (50.0 seconds):
Step 1: UR5 Load (46.9 seconds): [2025-10-22 17:22:32]
        Pre-check: Sample stage was empty before load.
[2025-10-22 17:22:32] Step 2: Slew Scan (3400.0 seconds):
Step 2: Slew Scan (3456.2 seconds): [2025-10-22 18:20:09]
        Scan succesful on Calib Al2O3
        Raw data files (11bmb_2384.h5 and 11bmb_2384.mda) saved to /data/oct25
===== Selected Functions =====
1. UR5 Load | {"position": "(0,0)"}
2. Slew Scan | {"sampleName": "Calib Al2O3", "startTTH": 1.0, "endTTH": 30.0}

*********** Run [2025-10-22 19:19:32] ***********
[2025-10-22 19:19:32] Step 1: Slew Scan (3400.0 seconds):
Step 1: Slew Scan (3492.9 seconds): [2025-10-22 20:17:45]
        Scan succesful on LaB6_x0_y0.75
        Raw data files (11bmb_2386.h5 and 11bmb_2386.mda) saved to /data/oct25
===== Selected Functions =====
1. Slew Scan | {"sampleName": "LaB6_x0_y0.75"}
"""


def test_parse_autosave(tmp_path: Path):
    p = tmp_path / "11bmb__2386.pre"
    p.write_text(PRE_SAMPLE)
    res = aux_metadata.parse_autosave(p)
    assert res["11bmb:BeamOn"] == 1.0
    assert res["11bmb:m28.RBV"] == -6.0
    # Spaces in value -> kept as string.
    assert res["11bmb:AShutter:LastOnTime"] == "2025/10/21 11:46:58.162"
    assert res["11bmb:S:IOC:timeOfDayForm2SI"] == "10/22/25 19:20:55"
    # Commented line skipped.
    assert "11bmb:m45.RBV" not in res


def test_find_pre_post(tmp_path: Path):
    mda = tmp_path / "11bmb_2386.mda"
    mda.write_bytes(b"")
    pre = tmp_path / "11bmb__2386.pre"
    pre.write_text("a 1\n<END>\n")
    pre_p, post_p = aux_metadata.find_pre_post(mda)
    assert pre_p == pre
    assert post_p is None


def test_parse_staff_log(tmp_path: Path):
    p = tmp_path / "StaffLog_10_22_2025.log"
    p.write_text(STAFF_LOG_SAMPLE)
    blocks = aux_metadata.parse_staff_log(p)
    assert len(blocks) == 2
    b0 = blocks[0]
    assert b0["timestamp"] == "2025-10-22 17:21:45"
    assert b0["source_log"] == "StaffLog_10_22_2025.log"
    assert 2384 in b0["referenced_runs"]
    names = [f["name"] for f in b0["selected_functions"]]
    assert "Slew Scan" in names
    slew = next(f for f in b0["selected_functions"] if f["name"] == "Slew Scan")
    assert slew["params"]["sampleName"] == "Calib Al2O3"

    b1 = blocks[1]
    assert 2386 in b1["referenced_runs"]


def test_index_staff_logs_and_cache(tmp_path: Path, monkeypatch):
    p = tmp_path / "StaffLog_10_22_2025.log"
    p.write_text(STAFF_LOG_SAMPLE)

    # Reset cache.
    aux_metadata._STAFF_LOG_CACHE.clear()

    call_count = {"n": 0}
    real_parse = aux_metadata.parse_staff_log

    def counting_parse(path):
        call_count["n"] += 1
        return real_parse(path)

    monkeypatch.setattr(aux_metadata, "parse_staff_log", counting_parse)

    idx1 = aux_metadata.index_staff_logs([p])
    idx2 = aux_metadata.index_staff_logs([p])
    assert idx1 is idx2
    assert call_count["n"] == 1
    assert 2384 in idx1
    assert 2386 in idx1


def test_parse_run_number():
    assert aux_metadata.parse_run_number(Path("11bmb_2386.mda")) == 2386
    assert aux_metadata.parse_run_number(Path("11bmb__2386.pre")) == 2386
    assert aux_metadata.parse_run_number(Path("nonsense.mda")) is None


def test_staff_log_for_run_no_logs(tmp_path: Path):
    assert aux_metadata.staff_log_for_run(123, tmp_path) is None
