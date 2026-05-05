"""
Auxiliary metadata parsers for v1 raw data.

Handles:
- EPICS autosave .pre/.post sidecar files (PV snapshots).
- StaffLog_*.log run-block files (operator activity log).

The StaffLog parser is set-based and module-cached: parsing a directory's
worth of staff logs many times during a batch conversion only re-reads the
files when their mtime/size changes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

import pyarrow as pa


class RunBlock(TypedDict):
    """
    One run section from a StaffLog file.

    Attributes
    ----------
    timestamp : str
        ISO-ish timestamp from the run delimiter line.
    source_log : str
        Basename of the StaffLog file this block was parsed from.
    steps : list of str
        Raw step lines (``Step N: ...``) and associated status lines.
    selected_functions : list of dict
        Each entry has ``"name"`` (str) and ``"params"`` (dict) keys.
    referenced_runs : list of int
        Run numbers mentioned in ``Raw data files (...)`` lines.
    """

    timestamp: str
    source_log: str
    steps: list[str]
    selected_functions: list[dict[str, Any]]
    referenced_runs: list[int]


# ----- .pre / .post EPICS autosave -----------------------------------------


def parse_autosave(path: Path) -> dict[str, str | float]:
    """
    Parse an EPICS autosave ``.pre`` / ``.post`` snapshot.

    Lines starting with ``#``, ``!``, or ``<END>`` are skipped, as are blank
    lines.  Each remaining line is split on the first whitespace; values that
    parse as float are coerced, otherwise the raw string is kept.

    Parameters
    ----------
    path : Path
        Path to the autosave file.

    Returns
    -------
    dict of {str : str or float}
        Mapping of PV name to its snapshot value.
    """
    out: dict[str, str | float] = {}
    with path.open("r") as fin:
        for raw_line in fin:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("#", "!", "<END>")):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            pv, value_str = parts[0], parts[1].strip()
            try:
                out[pv] = float(value_str)
            except ValueError:
                out[pv] = value_str
    return out


def find_pre_post(data_path: Path) -> tuple[Path | None, Path | None]:
    """
    Locate the ``.pre`` / ``.post`` siblings for a data file.

    The autosave files use a double-underscore stem
    (e.g. ``11bmb__2386.pre``) while data files use a single
    underscore (``11bmb_2386.mda``, ``11bmb_2386_mda_defROI.h5``, etc.).
    The run number is extracted via :func:`parse_run_number` and the
    beamline prefix is taken as everything before ``_<run>``.

    Parameters
    ----------
    data_path : Path
        Path to the data file (e.g. ``.mda`` or ``.h5``).

    Returns
    -------
    pre : Path or None
        Path to the ``.pre`` file, or ``None`` if it does not exist.
    post : Path or None
        Path to the ``.post`` file, or ``None`` if it does not exist.
    """
    stem = data_path.stem
    # Match the first _<digits> group (the run number).
    m = re.match(r"^(.*?)_(\d+)", stem)
    if m is None:
        return None, None
    base, run = m.group(1), m.group(2)
    auto_stem = f"{base}__{run}"
    pre = data_path.with_name(f"{auto_stem}.pre")
    post = data_path.with_name(f"{auto_stem}.post")
    return (pre if pre.exists() else None, post if post.exists() else None)


_AUTOSAVE_HEADER_RE = re.compile(r"^#\s*autosave\s+\S+\s+.*?(\d{6})-(\d{6})\s*$")


def parse_autosave_timestamp(path: Path) -> str | None:
    """
    Extract the timestamp from the autosave header line.

    The header format is ``# autosave R5.3 ... YYMMDD-HHMMSS``.

    Parameters
    ----------
    path : Path
        Path to the autosave file.

    Returns
    -------
    str or None
        ISO-formatted timestamp (``YYYY-MM-DDTHH:MM:SS``), or ``None``
        if the header could not be parsed.
    """
    with path.open("r") as fin:
        for line in fin:
            m = _AUTOSAVE_HEADER_RE.match(line.rstrip("\n"))
            if m is not None:
                date_s, time_s = m.group(1), m.group(2)
                # YYMMDD -> 20YY-MM-DD
                yy, mm, dd = date_s[:2], date_s[2:4], date_s[4:6]
                hh, mi, ss = time_s[:2], time_s[2:4], time_s[4:6]
                return f"20{yy}-{mm}-{dd}T{hh}:{mi}:{ss}"
            # Only first line should be the header; stop early.
            break
    return None


def baseline_table(
    pre: dict[str, str | float] | None,
    post: dict[str, str | float] | None,
    *,
    pre_timestamp: str | None = None,
    post_timestamp: str | None = None,
) -> pa.Table | None:
    """
    Build a 2-row pyarrow Table from pre/post autosave snapshots.

    Row 0 corresponds to the pre-scan snapshot and row 1 to the
    post-scan snapshot.  Columns are sorted alphabetically by PV name.
    A ``timestamp`` column (string) is prepended if timestamps are
    available.

    Parameters
    ----------
    pre : dict of {str : str or float} or None
        Pre-scan PV snapshot (from :func:`parse_autosave`).
    post : dict of {str : str or float} or None
        Post-scan PV snapshot (from :func:`parse_autosave`).
    pre_timestamp : str or None
        Timestamp extracted from the pre autosave header.
    post_timestamp : str or None
        Timestamp extracted from the post autosave header.

    Returns
    -------
    pa.Table or None
        A 2-row Table, or ``None`` if both *pre* and *post* are ``None``.
    """
    if pre is None and post is None:
        return None

    pre = pre or {}
    post = post or {}

    # Union of keys, sorted for determinism.
    all_keys = sorted(set(pre) | set(post))

    columns: list[pa.Array] = []
    names: list[str] = []

    # Prepend timestamp column if we have any.
    if pre_timestamp is not None or post_timestamp is not None:
        names.append("timestamp")
        columns.append(pa.array([pre_timestamp, post_timestamp], type=pa.string()))

    for key in all_keys:
        pre_val = pre.get(key)
        post_val = post.get(key)

        # Determine type from whichever value is present.
        sample = pre_val if pre_val is not None else post_val
        if isinstance(sample, float):
            arr = pa.array([pre_val, post_val], type=pa.float64())
        else:
            # String or None — coerce both to str or null.
            arr = pa.array(
                [
                    str(pre_val) if pre_val is not None else None,
                    str(post_val) if post_val is not None else None,
                ],
                type=pa.string(),
            )
        columns.append(arr)
        names.append(key)

    return pa.table(columns, names=names)


def parse_run_number(path: Path) -> int | None:
    """
    Extract the trailing run number from a filename stem.

    Matches patterns like ``11bmb_2386`` or ``11bmb__2386``.

    Parameters
    ----------
    path : Path
        File path whose stem will be inspected.

    Returns
    -------
    int or None
        The run number, or ``None`` if the stem does not end with
        ``_<digits>``.
    """
    m = re.search(r"_(\d+)$", path.stem)
    if m is None:
        return None
    return int(m.group(1))


# ----- StaffLog parsing -----------------------------------------------------


_RUN_DELIM_RE = re.compile(r"^\*{5,}\s*Run\s*\[([^\]]+)\]\s*\*{5,}\s*$", re.MULTILINE)
_SEL_FUNC_HEADER_RE = re.compile(r"^=+\s*Selected Functions\s*=+\s*$", re.MULTILINE)
_SEL_FUNC_LINE_RE = re.compile(r"^\s*\d+\.\s*(.+?)\s*\|\s*(\{.*\})\s*$")
_STEP_LINE_RE = re.compile(r"^\s*(?:\[[^\]]+\]\s*)?Step\s+\d+:")
_RAW_REF_RE = re.compile(r"11bmb_(\d+)\.(?:h5|mda)")


def parse_staff_log(path: Path) -> list[RunBlock]:
    """
    Parse a single ``StaffLog_*.log`` file into a list of run blocks.

    The file is split on ``*********** Run [...] ***********`` delimiters.
    Each block's steps, selected functions, and referenced run numbers are
    extracted.

    Parameters
    ----------
    path : Path
        Path to the staff-log file.

    Returns
    -------
    list of RunBlock
        One entry per run delimiter found in the file.
    """
    text = path.read_text()
    matches = list(_RUN_DELIM_RE.finditer(text))
    blocks: list[RunBlock] = []
    for i, m in enumerate(matches):
        timestamp = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]

        # Split body at "===== Selected Functions =====" if present.
        sel_m = _SEL_FUNC_HEADER_RE.search(body)
        if sel_m is not None:
            steps_part = body[: sel_m.start()]
            funcs_part = body[sel_m.end() :]
        else:
            steps_part, funcs_part = body, ""

        steps: list[str] = []
        for line in steps_part.splitlines():
            s = line.rstrip()
            if not s.strip():
                continue
            if _STEP_LINE_RE.match(s) or s.lstrip().startswith(
                (
                    "Step ",
                    "Translated",
                    "Raw data",
                    "✅",
                    "❌",
                    "Pre-check",
                    "Post-check",
                )
            ):
                steps.append(s)

        selected_functions: list[dict[str, Any]] = []
        for line in funcs_part.splitlines():
            if not line.strip():
                continue
            fm = _SEL_FUNC_LINE_RE.match(line)
            if fm is None:
                continue
            name = fm.group(1).strip()
            try:
                params = json.loads(fm.group(2))
            except json.JSONDecodeError:
                params = {"_raw": fm.group(2)}
            selected_functions.append({"name": name, "params": params})

        referenced_runs = sorted({int(r) for r in _RAW_REF_RE.findall(body)})

        blocks.append(
            RunBlock(
                timestamp=timestamp,
                source_log=path.name,
                steps=steps,
                selected_functions=selected_functions,
                referenced_runs=referenced_runs,
            )
        )
    return blocks


def find_staff_logs(directory: Path) -> list[Path]:
    """
    Return sorted list of ``StaffLog_*.log`` files in *directory*.

    Parameters
    ----------
    directory : Path
        Directory to search.

    Returns
    -------
    list of Path
        Matching log files, sorted lexicographically.
    """
    return sorted(directory.glob("StaffLog_*.log"))


# Module-level cache. Key = frozenset of (path, mtime_ns, size) so the cache
# transparently invalidates when a log file changes on disk.
_STAFF_LOG_CACHE: dict[frozenset, dict[int, list[RunBlock]]] = {}


def _cache_key(paths: Sequence[Path]) -> frozenset:
    items = []
    for p in paths:
        try:
            st = p.stat()
            items.append((str(p.resolve()), st.st_mtime_ns, st.st_size))
        except FileNotFoundError:
            items.append((str(p.resolve()), 0, 0))
    return frozenset(items)


def index_staff_logs(paths: Sequence[Path]) -> dict[int, list[RunBlock]]:
    """
    Parse all staff-log files and build a run-number index.

    Results are cached at module scope, keyed by the set of
    ``(resolved_path, mtime_ns, size)`` tuples so that the cache
    transparently invalidates when any file changes on disk.

    Parameters
    ----------
    paths : Sequence of Path
        Staff-log files to parse (typically from :func:`find_staff_logs`).

    Returns
    -------
    dict of {int : list of RunBlock}
        Mapping from run number to every block that references it.
    """
    key = _cache_key(paths)
    cached = _STAFF_LOG_CACHE.get(key)
    if cached is not None:
        return cached

    index: dict[int, list[RunBlock]] = {}
    for p in paths:
        for block in parse_staff_log(p):
            for run in block["referenced_runs"]:
                index.setdefault(run, []).append(block)

    _STAFF_LOG_CACHE[key] = index
    return index


def staff_log_for_run(run_number: int, directory: Path) -> list[RunBlock] | None:
    """
    Look up staff-log blocks for a single run number.

    Finds all ``StaffLog_*.log`` files in *directory*, indexes them
    (using the module-level cache), and returns the blocks referencing
    *run_number*.

    Parameters
    ----------
    run_number : int
        The run number to look up.
    directory : Path
        Directory containing the staff-log files.

    Returns
    -------
    list of RunBlock or None
        Matching blocks, or ``None`` if no staff-log files exist or the
        run number is not referenced.
    """
    paths = find_staff_logs(directory)
    if not paths:
        return None
    index = index_staff_logs(paths)
    return index.get(run_number)
