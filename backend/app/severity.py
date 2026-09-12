"""
Severity is computed with plain arithmetic, not by asking the LLM to judge
it. This is a deliberate design choice: an LLM asked "how worried should
this patient be" will be inconsistent across runs, and for a health
product that inconsistency is a real risk. A fixed rule is boring but
reproducible, testable, and easy to explain and defend.

Rule:
  - Parse the printed reference range (handles "70-99", "<100", ">40",
    "0.4 - 4.0", etc).
  - If the value falls inside the range -> status = normal.
  - If outside, compute % deviation from the nearest boundary.
      <= 20% outside the boundary -> "mild"
      >  20% outside the boundary -> "concerning"
  - If the range can't be parsed, status/severity = "unknown" and we say
    so rather than guessing.
"""
import re
from typing import Optional, Tuple

from .schemas import LabValue, ScoredLabValue

MILD_THRESHOLD_PCT = 20.0


def _parse_range(range_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (low, high). Either may be None for open-ended ranges like
    '<100' (low=None, high=100) or '>40' (low=40, high=None).
    """
    s = range_str.strip()

    m = re.match(r"^[<≤]\s*([\d.]+)$", s)
    if m:
        return (None, float(m.group(1)))

    m = re.match(r"^[>≥]\s*([\d.]+)$", s)
    if m:
        return (float(m.group(1)), None)

    m = re.match(r"^([\d.]+)\s*[-–to]+\s*([\d.]+)$", s)
    if m:
        return (float(m.group(1)), float(m.group(2)))

    return (None, None)


def _parse_value(value_str: str) -> Optional[float]:
    m = re.search(r"-?\d+\.?\d*", value_str.replace(",", ""))
    return float(m.group()) if m else None


def score(lab_value: LabValue) -> ScoredLabValue:
    scored = ScoredLabValue(**lab_value.__dict__)

    v = _parse_value(lab_value.value)
    low, high = _parse_range(lab_value.reference_range)

    if v is None or (low is None and high is None):
        scored.status = "unknown"
        scored.severity = "unknown"
        return scored

    if low is not None and v < low:
        scored.status = "low"
        deviation = (low - v) / low * 100 if low != 0 else 100.0
    elif high is not None and v > high:
        scored.status = "high"
        deviation = (v - high) / high * 100 if high != 0 else 100.0
    else:
        scored.status = "normal"
        scored.severity = "normal"
        scored.deviation_pct = 0.0
        return scored

    scored.deviation_pct = round(deviation, 1)
    scored.severity = "mild" if deviation <= MILD_THRESHOLD_PCT else "concerning"
    return scored
