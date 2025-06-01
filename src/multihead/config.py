"""
Configuration and data structure definitions for multihead.
"""

from dataclasses import dataclass
from typing import NamedTuple

__all__ = ["CrystalROI", "SimpleSliceTuple"]


class SimpleSliceTuple(NamedTuple):
    start: int
    stop: int


@dataclass
class CrystalROI:
    rslc: SimpleSliceTuple
    cslc: SimpleSliceTuple

    def to_slices(self) -> tuple[slice, slice]:
        return slice(*self.rslc), slice(*self.cslc)