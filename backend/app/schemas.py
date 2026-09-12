"""
Shared data shapes. Keeping these as plain dataclasses (not pydantic v1/v2
locked-in) keeps the extraction -> severity -> retrieval -> generation
pipeline easy to unit test without spinning up FastAPI.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LabValue:
    """One row extracted from the uploaded report."""
    test: str
    value: str
    unit: str = ""
    reference_range: str = ""
    reported_flag: str = ""  # whatever the lab itself printed, if anything


@dataclass
class ScoredLabValue(LabValue):
    """A LabValue plus our own rule-based severity judgment."""
    status: str = "unknown"      # "low" | "high" | "normal" | "unknown"
    severity: str = "normal"     # "normal" | "mild" | "concerning"
    deviation_pct: Optional[float] = None


@dataclass
class Finding:
    """One fully-explained item in the final report."""
    test: str
    value: str
    unit: str
    status: str
    severity: str
    plain_meaning: str
    next_step: str
    sources: list = field(default_factory=list)


@dataclass
class ReportResult:
    overall_summary: str
    findings: list          # list[Finding]
    normal_findings: list   # list[str]
    bottom_line: str
    disclaimer: str
    insufficient_data: bool = False
    message: Optional[str] = None
