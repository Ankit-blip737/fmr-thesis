"""Bounding-box / region geometry used by Signal B (attention grounding).

Regions are stored in normalized [0, 1] image coordinates so they are resolution
independent. This is the only geometry primitive the rest of the package needs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    """An axis-aligned box in normalized image coordinates (0..1)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        # Normalize ordering so x0<=x1, y0<=y1 without mutating a frozen field.
        object.__setattr__(self, "x0", float(min(self.x0, self.x1)))
        object.__setattr__(self, "x1", float(max(self.x0, self.x1)))
        object.__setattr__(self, "y0", float(min(self.y0, self.y1)))
        object.__setattr__(self, "y1", float(max(self.y0, self.y1)))

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def iou(self, other: "Region") -> float:
        """Intersection-over-union with another region (0..1)."""
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def from_grid_cell(cls, row: int, col: int, n_rows: int, n_cols: int) -> "Region":
        """Region covering one cell of an ``n_rows x n_cols`` grid over the image."""
        return cls(col / n_cols, row / n_rows, (col + 1) / n_cols, (row + 1) / n_rows)
