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


def find_pre_post(mda_path: Path) -> tuple[Path | None, Path | None]:
    """
    Locate the ``.pre`` / ``.post`` siblings for an ``.mda`` file.

    The autosave files use a double-underscore stem
    (e.g. ``11bmb__2386.pre``) while the ``.mda`` file uses a single
    underscore (``11bmb_2386.mda``).

    Parameters
    ----------
    mda_path : Path
        Path to the ``.mda`` file.

    Returns
    -------
    pre : Path or None
        Path to the ``.pre`` file, or ``None`` if it does not exist.
    post : Path or None
        Path to the ``.post`` file, or ``None`` if it does not exist.
    """
    stem = mda_path.stem
    m = re.match(r"^(.*?)_(\d+)$", stem)
    if m is None:
        return None, None
    base, run = m.group(1), m.group(2)
    auto_stem = f"{base}__{run}"
    pre = mda_path.with_name(f"{auto_stem}.pre")
    post = mda_path.with_name(f"{auto_stem}.post")
    return (pre if pre.exists() else None, post if post.exists() else None)


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
                ("Step ", "Translated", "Raw data", "✅", "❌", "Pre-check", "Post-check")
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


def staff_log_for_run(
    run_number: int, directory: Path
) -> list[RunBlock] | None:
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
