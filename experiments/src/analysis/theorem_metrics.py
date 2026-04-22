from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + (z * z / total)
    center = (p + z * z / (2 * total)) / denom
    margin = (z / denom) * sqrt((p * (1 - p) / total) + (z * z / (4 * total * total)))
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": mean(values), "std": pstdev(values)}
