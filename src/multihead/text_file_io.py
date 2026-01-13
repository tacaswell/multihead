"""
Functions to parse beamline plan and log text files.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from multihead.file_io import HRDRawBase


@dataclass
class UR5LoadAction:
    """Represents a UR5 Load action in a plan"""

    action_number: int
    sample_pos: list[str]
    pre_load_scan: bool
    load_check_interfere: bool


@dataclass
class UR5LoadLogEntry:
    """Represents a UR5 Load action execution in log"""

    action_number: int
    start_time: datetime
    end_time: datetime
    result: str
    action: UR5LoadAction
    estimated_duration: float
    actual_duration: float


@dataclass
class UR5UnloadAction:
    """Represents a UR5 Unload action in a plan"""

    action_number: int
    sample_pos: list[str]
    unload_check_interfere: bool


@dataclass
class UR5UnloadLogEntry:
    """Represents a UR5 Unload action execution in log"""

    action_number: int
    start_time: datetime
    end_time: datetime
    result: str
    action: UR5UnloadAction
    estimated_duration: float
    actual_duration: float


@dataclass
class SlewScanAction:
    """Represents a Slew Scan action in a plan"""

    action_number: int
    startTTH: float
    endTTH: float
    stepTTH: float
    timeStep: float
    sampleName: str
    formula: str
    spinnerOn: bool
    proposal: str
    TempPV: str
    TempPVoffset: float
    barcode: str
    kx2_pos: str
    ur5_pos: str


@dataclass
class SlewScanLogEntry:
    """Represents a Slew Scan action execution in log"""

    action_number: int
    start_time: datetime
    end_time: datetime
    result: str
    action: SlewScanAction
    raw_files: tuple[str, str] | None = None
    save_path: str | None = None
    estimated_duration: float
    actual_duration: float

    def open_raw_data(
        self, version: int, basepath: str | Path | None = None
    ) -> "HRDRawBase":
        """
        Open the raw data files associated with this scan.

        Parameters
        ----------
        version : int
            Version of the raw data format (1 or 2)
        basepath : str | Path | None, optional
            Base directory where raw data files are stored. If None, uses
            the save_path from the log file (for on-line usage at beamline).

        Returns
        -------
        HRDRawBase
            Opened raw data object (HRDRawV1 or HRDRawV2)
        
        Raises
        ------
        ValueError
            If raw_files is not available, or if save_path is not available
            when basepath is None
        """
        from multihead.file_io import open_data

        if self.raw_files is None:
            raise ValueError(
                f"Cannot open raw data for scan {self.action_number}: "
                "raw_files not available"
            )

        # Determine the base path to use
        if basepath is None:
            if self.save_path is None:
                raise ValueError(
                    f"Cannot open raw data for scan {self.action_number}: "
                    "save_path not available and no basepath provided"
                )
            base_path = Path(self.save_path)
        else:
            base_path = Path(basepath)
        
        if version == 1:
            # For version 1, raw_files should be (h5_file, mda_file)
            # Extract the base name without extension
            h5_file, mda_file = self.raw_files
            base_name = Path(h5_file).stem
            file_path = base_path / base_name
        else:
            # For version 2, construct filename like: 11bmb_2384_mda_defROI.h5
            # from raw_files like (11bmb_2384.h5, 11bmb_2384.mda)
            h5_file, mda_file = self.raw_files
            # Extract the base name without extension
            base_name = Path(h5_file).stem
            # Construct the v2 filename with _mda_defROI suffix
            v2_filename = f"{base_name}_mda_defROI.h5"
            file_path = base_path / v2_filename
        
        return open_data(file_path, version)


@dataclass
class TranslateSampleAction:
    """Represents a Translate Sample action in a plan"""

    action_number: int
    target_position: float
    axis: str  # 'X' or 'Y'


@dataclass
class TranslateSampleLogEntry:
    """Represents a Translate Sample action execution in log"""

    action_number: int
    start_time: datetime
    end_time: datetime
    result: str
    action: TranslateSampleAction
    estimated_duration: float
    actual_duration: float


@dataclass
class ExecutionContext:
    """Groups all log entries from a single run execution"""

    run_time: datetime
    entries: list["LogActionType"]


PlanActionType = UR5LoadAction | UR5UnloadAction | SlewScanAction | TranslateSampleAction
LogActionType = UR5LoadLogEntry | UR5UnloadLogEntry | SlewScanLogEntry | TranslateSampleLogEntry


@dataclass
class BeamlinePlan:
    """Container for parsed beamline plan actions"""

    actions: list[PlanActionType]

    def get_scans(self) -> list[SlewScanAction]:
        """Return only the Slew Scan actions"""
        return [a for a in self.actions if isinstance(a, SlewScanAction)]

    def get_load_actions(self) -> list[UR5LoadAction]:
        """Return only the UR5 Load actions"""
        return [a for a in self.actions if isinstance(a, UR5LoadAction)]

    def get_unload_actions(self) -> list[UR5UnloadAction]:
        """Return only the UR5 Unload actions"""
        return [a for a in self.actions if isinstance(a, UR5UnloadAction)]

    def get_translate_actions(self) -> list[TranslateSampleAction]:
        """Return only the Translate Sample actions"""
        return [a for a in self.actions if isinstance(a, TranslateSampleAction)]


@dataclass
class BeamlineLog:
    """Container for parsed beamline log actions (from execution)"""

    runs: list[ExecutionContext]

    def get_scans(self) -> list[SlewScanLogEntry]:
        """Return only the Slew Scan log entries from all runs"""
        scans = []
        for run in self.runs:
            scans.extend([a for a in run.entries if isinstance(a, SlewScanLogEntry)])
        return scans

    def get_load_actions(self) -> list[UR5LoadLogEntry]:
        """Return only the UR5 Load log entries from all runs"""
        loads = []
        for run in self.runs:
            loads.extend([a for a in run.entries if isinstance(a, UR5LoadLogEntry)])
        return loads

    def get_unload_actions(self) -> list[UR5UnloadLogEntry]:
        """Return only the UR5 Unload log entries from all runs"""
        unloads = []
        for run in self.runs:
            unloads.extend([a for a in run.entries if isinstance(a, UR5UnloadLogEntry)])
        return unloads

    def get_translate_actions(self) -> list[TranslateSampleLogEntry]:
        """Return only the Translate Sample log entries from all runs"""
        translates = []
        for run in self.runs:
            translates.extend([a for a in run.entries if isinstance(a, TranslateSampleLogEntry)])
        return translates


def parse_plan_line(line: str) -> PlanActionType | None:
    """
    Parse a single line from the beamline plan file.

    Parameters
    ----------
    line : str
        A line from the plan file in format: "N. Action | {json_data}"

    Returns
    -------
    PlanActionType | None
        Parsed action object or None if line cannot be parsed
    """
    # Match pattern: "N. Action Type | {json}"
    match = re.match(r"(\d+)\.\s+(.+?)\s+\|\s+(\{.+\})", line.strip())
    if not match:
        return None

    action_number = int(match.group(1))
    action_type = match.group(2).strip()
    json_str = match.group(3)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if action_type == "UR5 Load":
        return UR5LoadAction(
            action_number=action_number,
            sample_pos=data["sample_pos"],
            pre_load_scan=data["pre_load_scan"],
            load_check_interfere=data["load_check_interfere"],
        )
    elif action_type == "UR5 Unload":
        return UR5UnloadAction(
            action_number=action_number,
            sample_pos=data["sample_pos"],
            unload_check_interfere=data["unload_check_interfere"],
        )
    elif action_type == "Translate Sample Y":
        return TranslateSampleAction(
            action_number=action_number,
            target_position=float(data["samy"]),
            axis="Y",
        )
    elif action_type == "Translate Sample X":
        return TranslateSampleAction(
            action_number=action_number,
            target_position=float(data["samx"]),
            axis="X",
        )
    elif action_type == "Slew Scan":
        return SlewScanAction(
            action_number=action_number,
            startTTH=float(data["startTTH"]),
            endTTH=float(data["endTTH"]),
            stepTTH=float(data["stepTTH"]),
            timeStep=float(data["timeStep"]),
            sampleName=data["sampleName"],
            formula=data["formula"],
            spinnerOn=data["spinnerOn"],
            proposal=data["proposal"],
            TempPV=data["TempPV"],
            TempPVoffset=data["TempPVoffset"],
            barcode=data["barcode"],
            kx2_pos=data["kx2_pos"],
            ur5_pos=data["ur5_pos"],
        )

    return None


def parse_log_line(line: str, context: dict[str, Any]) -> LogActionType | None:
    """
    Parse a single line from the beamline log file (format with run headers and execution details).

    Parameters
    ----------
    line : str
        A line from the log file
    context : dict
        Shared context dictionary for tracking run information across lines

    Returns
    -------
    LogActionType | None
        Parsed action object or None if line cannot be parsed
    """
    line = line.strip()

    # Match run header: *********** Run [YYYY-MM-DD HH:MM:SS] ***********
    run_match = re.match(r"\*+ Run \[([^\]]+)\] \*+", line)
    if run_match:
        context["run_time"] = datetime.fromisoformat(run_match.group(1))
        context["step_start_time"] = None
        context["step_end_time"] = None
        context["estimated_duration"] = None
        context["actual_duration"] = None
        context["raw_files"] = None
        context["save_path"] = None
        context["translate_position"] = None
        context["in_selected_functions"] = False
        context["selected_functions"] = {}  # Maps action_number to action
        return None

    # Match Selected Functions header
    if line.startswith("===== Selected Functions ====="):
        context["in_selected_functions"] = True
        return None

    # Match step start: [YYYY-MM-DD HH:MM:SS] Step N: Action (duration seconds):
    step_start_match = re.match(
        r"\[([^\]]+)\] Step (\d+): (.+?) \(([0-9.]+) seconds\):", line
    )
    if step_start_match:
        context["step_start_time"] = datetime.fromisoformat(step_start_match.group(1))
        context["estimated_duration"] = float(step_start_match.group(4))
        context["step_action_name"] = step_start_match.group(3)
        context["step_number"] = int(step_start_match.group(2))
        context["in_selected_functions"] = False
        return None

    # Match step end: Step N: Action (duration seconds): [YYYY-MM-DD HH:MM:SS]
    step_end_match = re.match(
        r"Step (\d+): (.+?) \(([0-9.]+) seconds\): \[([^\]]+)\]", line
    )
    if step_end_match:
        context["step_end_time"] = datetime.fromisoformat(step_end_match.group(4))
        context["actual_duration"] = float(step_end_match.group(3))
        step_number = int(step_end_match.group(1))

        # Check if this is a Translate Sample action
        action_name = context["step_action_name"]
        if action_name.startswith("Translate Sample"):
            # Create TranslateSampleLogEntry if we have position info
            if (
                context["translate_position"] is not None
                and context["step_start_time"] is not None
            ):
                # Determine axis from action name
                axis = "Y" if "Y" in action_name else "X"
                # Check if we have a corresponding action in selected_functions
                selected_functions = context["selected_functions"]
                if step_number in selected_functions:
                    action = selected_functions[step_number]
                    if not isinstance(action, TranslateSampleAction):
                        # Fallback: create action from translate_position
                        action = TranslateSampleAction(
                            action_number=step_number,
                            target_position=context["translate_position"],
                            axis=axis,
                        )
                else:
                    # Fallback: create action from translate_position
                    action = TranslateSampleAction(
                        action_number=step_number,
                        target_position=context["translate_position"],
                        axis=axis,
                    )
                log_entry = TranslateSampleLogEntry(
                    action_number=step_number,
                    start_time=context["step_start_time"],
                    end_time=context["step_end_time"],
                    result="success",
                    action=action,
                    estimated_duration=context["estimated_duration"],
                    actual_duration=context["actual_duration"],
                )
                context["translate_position"] = None
                return log_entry

        return None

    # Match translate result: Translated to X.XX
    translate_match = re.match(r"Translated to ([0-9.-]+)", line)
    if translate_match:
        context["translate_position"] = float(translate_match.group(1))
        return None

    # Match success message with raw files
    files_match = re.match(r"Raw data files \(([^ ]+) and ([^ ]+)\) saved to (.+)", line)
    if files_match:
        context["raw_files"] = (files_match.group(1), files_match.group(2))
        context["save_path"] = files_match.group(3)
        return None

    # Match action line: N. Action | {json}
    action_match = re.match(r"(\d+)\.\s+(.+?)\s+\|\s+(\{.+\})", line)
    if not action_match:
        return None

    action_number = int(action_match.group(1))
    action_type = action_match.group(2).strip()
    json_str = action_match.group(3)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    # If we're in the Selected Functions section, store the actions for later matching
    if context["in_selected_functions"]:
        if action_type == "UR5 Load":
            action = UR5LoadAction(
                action_number=action_number,
                sample_pos=data["sample_pos"],
                pre_load_scan=data["pre_load_scan"],
                load_check_interfere=data["load_check_interfere"],
            )
            context["selected_functions"][action_number] = action
        elif action_type == "UR5 Unload":
            action = UR5UnloadAction(
                action_number=action_number,
                sample_pos=data["sample_pos"],
                unload_check_interfere=data["unload_check_interfere"],
            )
            context["selected_functions"][action_number] = action
        elif action_type == "Translate Sample Y":
            action = TranslateSampleAction(
                action_number=action_number,
                target_position=float(data["samy"]),
                axis="Y",
            )
            context["selected_functions"][action_number] = action
        elif action_type == "Translate Sample X":
            action = TranslateSampleAction(
                action_number=action_number,
                target_position=float(data["samx"]),
                axis="X",
            )
            context["selected_functions"][action_number] = action
        elif action_type == "Slew Scan":
            action = SlewScanAction(
                action_number=action_number,
                startTTH=float(data["startTTH"]),
                endTTH=float(data["endTTH"]),
                stepTTH=float(data["stepTTH"]),
                timeStep=float(data["timeStep"]),
                sampleName=data["sampleName"],
                formula=data["formula"],
                spinnerOn=data["spinnerOn"],
                proposal=data["proposal"],
                TempPV=data["TempPV"],
                TempPVoffset=data["TempPVoffset"],
                barcode=data["barcode"],
                kx2_pos=data["kx2_pos"],
                ur5_pos=data["ur5_pos"],
            )
            context["selected_functions"][action_number] = action
        return None

    # Parse timestamps from data (fallback if not in context)
    start_time = context["step_start_time"]
    end_time = context["step_end_time"]
    if data.get("start_time"):
        start_time = datetime.fromisoformat(data["start_time"])
    if data.get("end_time"):
        end_time = datetime.fromisoformat(data["end_time"])

    # Get result, default to empty string if None
    result = data.get("result", "")
    if result is None:
        result = ""

    # Get the step number from context
    step_number = context["step_number"]

    if action_type == "UR5 Load":
        if start_time is None or end_time is None:
            raise ValueError(f"Missing timestamps for UR5 Load action {action_number}")
        # Try to get action from selected_functions, otherwise create from data
        selected_functions = context["selected_functions"]
        if step_number in selected_functions:
            action = selected_functions[step_number]
            if not isinstance(action, UR5LoadAction):
                action = UR5LoadAction(
                    action_number=action_number,
                    sample_pos=data["sample_pos"],
                    pre_load_scan=data["pre_load_scan"],
                    load_check_interfere=data["load_check_interfere"],
                )
        else:
            action = UR5LoadAction(
                action_number=action_number,
                sample_pos=data["sample_pos"],
                pre_load_scan=data["pre_load_scan"],
                load_check_interfere=data["load_check_interfere"],
            )
        return UR5LoadLogEntry(
            action_number=step_number,
            start_time=start_time,
            end_time=end_time,
            result=result,
            action=action,
            estimated_duration=context["estimated_duration"],
            actual_duration=context["actual_duration"],
        )
    elif action_type == "UR5 Unload":
        if start_time is None or end_time is None:
            raise ValueError(f"Missing timestamps for UR5 Unload action {action_number}")
        # Try to get action from selected_functions, otherwise create from data
        selected_functions = context["selected_functions"]
        if step_number in selected_functions:
            action = selected_functions[step_number]
            if not isinstance(action, UR5UnloadAction):
                action = UR5UnloadAction(
                    action_number=action_number,
                    sample_pos=data["sample_pos"],
                    unload_check_interfere=data["unload_check_interfere"],
                )
        else:
            action = UR5UnloadAction(
                action_number=action_number,
                sample_pos=data["sample_pos"],
                unload_check_interfere=data["unload_check_interfere"],
            )
        return UR5UnloadLogEntry(
            action_number=step_number,
            start_time=start_time,
            end_time=end_time,
            result=result,
            action=action,
            estimated_duration=context["estimated_duration"],
            actual_duration=context["actual_duration"],
        )
    elif action_type == "Slew Scan":
        if start_time is None or end_time is None:
            raise ValueError(f"Missing timestamps for Slew Scan action {action_number}")
        # Try to get action from selected_functions, otherwise create from data
        selected_functions = context["selected_functions"]
        if step_number in selected_functions:
            scan_action = selected_functions[step_number]
            if not isinstance(scan_action, SlewScanAction):
                scan_action = SlewScanAction(
                    action_number=action_number,
                    startTTH=float(data["startTTH"]),
                    endTTH=float(data["endTTH"]),
                    stepTTH=float(data["stepTTH"]),
                    timeStep=float(data["timeStep"]),
                    sampleName=data["sampleName"],
                    formula=data["formula"],
                    spinnerOn=data["spinnerOn"],
                    proposal=data["proposal"],
                    TempPV=data["TempPV"],
                    TempPVoffset=data["TempPVoffset"],
                    barcode=data["barcode"],
                    kx2_pos=data["kx2_pos"],
                    ur5_pos=data["ur5_pos"],
                )
        else:
            scan_action = SlewScanAction(
                action_number=action_number,
                startTTH=float(data["startTTH"]),
                endTTH=float(data["endTTH"]),
                stepTTH=float(data["stepTTH"]),
                timeStep=float(data["timeStep"]),
                sampleName=data["sampleName"],
                formula=data["formula"],
                spinnerOn=data["spinnerOn"],
                proposal=data["proposal"],
                TempPV=data["TempPV"],
                TempPVoffset=data["TempPVoffset"],
                barcode=data["barcode"],
                kx2_pos=data["kx2_pos"],
                ur5_pos=data["ur5_pos"],
            )

        # Create SlewScanLogEntry with raw file information from context
        return SlewScanLogEntry(
            action_number=step_number,
            start_time=start_time,
            end_time=end_time,
            result=result,
            action=scan_action,
            raw_files=context["raw_files"],
            save_path=context["save_path"],
            estimated_duration=context["estimated_duration"],
            actual_duration=context["actual_duration"],
        )

    return None


def parse_plan_file(path: str | Path) -> BeamlinePlan:
    """
    Parse a beamline plan file from 11BM.

    Plan files contain numbered action lines with JSON configuration,
    representing the intended data acquisition sequence.

    Parameters
    ----------
    path : str | Path
        Path to the plan file

    Returns
    -------
    BeamlinePlan
        Parsed planned actions from the file
    """
    actions: list[PlanActionType] = []

    with open(path, "r") as f:
        for line in f:
            action = parse_plan_line(line)
            if action is not None:
                actions.append(action)

    return BeamlinePlan(actions=actions)


def _parse_single_run(run_lines: list[str], run_time: datetime) -> ExecutionContext | None:
    """
    Parse a single run's worth of log lines.
    
    Parameters
    ----------
    run_lines : list[str]
        Lines belonging to a single run
    run_time : datetime
        Start time of the run
    
    Returns
    -------
    ExecutionContext | None
        Parsed execution context, or None if no valid entries found
    """
    # Parse execution context (step timing, results)
    exec_context: dict[int, dict[str, Any]] = {}  # Maps step_number to execution info
    in_selected_functions = False
    selected_functions: dict[int, PlanActionType] = {}
    
    # Track raw files per step (for scans)
    step_files: dict[int, tuple[tuple[str, str], str]] = {}  # step_number -> (raw_files, save_path)
    current_step: int | None = None
    
    for line in run_lines:
        line = line.strip()
        
        # Check for Selected Functions section
        if line.startswith("===== Selected Functions ====="):
            in_selected_functions = True
            continue
        
        if in_selected_functions:
            # Parse action lines in Selected Functions
            action = parse_plan_line(line)
            if action is not None:
                selected_functions[action.action_number] = action
            continue
        
        # Parse execution information
        # Match step start: [YYYY-MM-DD HH:MM:SS] Step N: Action (duration seconds):
        step_start_match = re.match(
            r"\[([^\]]+)\] Step (\d+): (.+?) \(([0-9.]+) seconds\):", line
        )
        if step_start_match:
            step_number = int(step_start_match.group(2))
            current_step = step_number
            if step_number not in exec_context:
                exec_context[step_number] = {}
            exec_context[step_number]["start_time"] = datetime.fromisoformat(
                step_start_match.group(1)
            )
            exec_context[step_number]["action_name"] = step_start_match.group(3)
            exec_context[step_number]["estimated_duration"] = float(step_start_match.group(4))
            continue

        # Match step end: Step N: Action (duration seconds): [YYYY-MM-DD HH:MM:SS]
        step_end_match = re.match(
            r"Step (\d+): (.+?) \(([0-9.]+) seconds\): \[([^\]]+)\]", line
        )
        if step_end_match:
            step_number = int(step_end_match.group(1))
            if step_number not in exec_context:
                exec_context[step_number] = {}
            exec_context[step_number]["end_time"] = datetime.fromisoformat(
                step_end_match.group(4)
            )
            exec_context[step_number]["actual_duration"] = float(step_end_match.group(3))
            exec_context[step_number]["result"] = "success"
            continue

        # Match translate result: Translated to X.XX
        translate_match = re.match(r"Translated to ([0-9.-]+)", line)
        if translate_match:
            if current_step is not None:
                if current_step not in exec_context:
                    exec_context[current_step] = {}
                exec_context[current_step]["translate_position"] = float(
                    translate_match.group(1)
                )
            continue

        # Match success message with raw files
        files_match = re.match(
            r"Raw data files \(([^ ]+) and ([^ ]+)\) saved to (.+)", line
        )
        if files_match:
            if current_step is not None:
                raw_files = (files_match.group(1), files_match.group(2))
                save_path = files_match.group(3)
                step_files[current_step] = (raw_files, save_path)
            continue

    # Now create log entries by combining execution context with selected functions
    entries: list[LogActionType] = []
    
    for step_number in sorted(exec_context.keys()):
        exec_info = exec_context[step_number]
        
        # Require matching selected function
        action = selected_functions[step_number]
        
        # Require complete timing information
        start_time = exec_info["start_time"]
        end_time = exec_info["end_time"]
        estimated_duration = exec_info["estimated_duration"]
        actual_duration = exec_info["actual_duration"]
        
        # Result - use empty string as default for missing result
        if "result" in exec_info:
            result = exec_info["result"]
        else:
            result = ""
        
        if isinstance(action, UR5LoadAction):
            entries.append(
                UR5LoadLogEntry(
                    action_number=step_number,
                    start_time=start_time,
                    end_time=end_time,
                    result=result,
                    action=action,
                    estimated_duration=estimated_duration,
                    actual_duration=actual_duration,
                )
            )
        elif isinstance(action, UR5UnloadAction):
            entries.append(
                UR5UnloadLogEntry(
                    action_number=step_number,
                    start_time=start_time,
                    end_time=end_time,
                    result=result,
                    action=action,
                    estimated_duration=estimated_duration,
                    actual_duration=actual_duration,
                )
            )
        elif isinstance(action, TranslateSampleAction):
            entries.append(
                TranslateSampleLogEntry(
                    action_number=step_number,
                    start_time=start_time,
                    end_time=end_time,
                    result=result,
                    action=action,
                    estimated_duration=estimated_duration,
                    actual_duration=actual_duration,
                )
            )
        elif isinstance(action, SlewScanAction):
            # Get raw files for this step - these are optional
            raw_files = None
            save_path = None
            if step_number in step_files:
                raw_files, save_path = step_files[step_number]
            
            entries.append(
                SlewScanLogEntry(
                    action_number=step_number,
                    start_time=start_time,
                    end_time=end_time,
                    result=result,
                    action=action,
                    raw_files=raw_files,
                    save_path=save_path,
                    estimated_duration=estimated_duration,
                    actual_duration=actual_duration,
                )
            )
    
    if not entries:
        return None
    
    return ExecutionContext(run_time=run_time, entries=entries)


def parse_log_files(paths: str | Path | Sequence[str | Path]) -> BeamlineLog:
    """
    Parse one or more beamline log files from 11BM.

    Log files contain run headers with timestamps and execution details,
    representing what actually happened during data acquisition.
    Multiple log files are concatenated together in order.

    This uses a two-pass approach:
    1. First pass: collect execution context (step timing, files saved)
    2. Second pass: parse Selected Functions and match to execution context

    Parameters
    ----------
    paths : str | Path | Sequence[str | Path]
        Path to a single log file, or paths to multiple log files to parse in order

    Returns
    -------
    BeamlineLog
        Parsed actions from all log files, grouped by execution run
    """
    # Normalize to sequence
    if isinstance(paths, (str, Path)):
        paths = [paths]

    # Read all lines from all files
    all_lines = []
    for path in paths:
        with open(path, "r") as f:
            all_lines.extend(f.readlines())

    runs: list[ExecutionContext] = []
    current_run_time: datetime | None = None
    current_run_lines: list[str] = []

    # Parse through all lines, processing each run when complete
    for line in all_lines:
        run_match = re.match(r"\*+ Run \[([^\]]+)\] \*+", line.strip())
        if run_match:
            # Found start of new run - process previous run if exists
            if current_run_time is not None and current_run_lines:
                execution_context = _parse_single_run(current_run_lines, current_run_time)
                if execution_context is not None:
                    runs.append(execution_context)
            
            # Start new run
            current_run_time = datetime.fromisoformat(run_match.group(1))
            current_run_lines = [line]
        else:
            # Accumulate lines for current run
            if current_run_time is not None:
                current_run_lines.append(line)

    # Handle final run
    if current_run_time is not None and current_run_lines:
        execution_context = _parse_single_run(current_run_lines, current_run_time)
        if execution_context is not None:
            runs.append(execution_context)

    return BeamlineLog(runs=runs)
