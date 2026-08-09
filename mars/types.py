"""Core data structures shared across the MARS pipeline.

These types define the contracts between agents. Each agent consumes and
produces these dataclasses, which keeps the hand-offs explicit and makes
every intermediate output inspectable -- the central design goal of the
architecture described in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Source(str, Enum):
    """Where a medication mention came from. Provenance is carried end to end."""
    ADMISSION_NOTE = "admission_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    ORDERS = "orders"


class DiscrepancyType(str, Enum):
    """The five typed discrepancy categories the Discrepancy Agent emits."""
    OMISSION = "omission"          # present at source, absent at discharge
    ADDITION = "addition"          # present at discharge, absent at source
    DOSE_CONFLICT = "dose_conflict"
    FREQUENCY_CONFLICT = "frequency_conflict"
    DUPLICATION = "duplication"    # same ingredient under two concepts


@dataclass
class MedicationMention:
    """A raw medication mention as produced by the Extraction Agent."""
    surface: str                       # raw text, e.g. "Tylenol 500 mg PO"
    source: Source
    dose: Optional[str] = None         # e.g. "500 mg"
    route: Optional[str] = None        # e.g. "PO"
    frequency: Optional[str] = None    # e.g. "BID"
    # Filled in by the Normalization Agent:
    rxnorm_cui: Optional[str] = None   # RxNorm concept unique identifier
    rxnorm_ingredient: Optional[str] = None  # normalized ingredient name

    @property
    def normalized(self) -> bool:
        return self.rxnorm_ingredient is not None


@dataclass
class Discrepancy:
    """A single typed discrepancy between the source and discharge lists."""
    dtype: DiscrepancyType
    ingredient: str
    severity: float                    # 0..1, higher = more clinically significant
    source_detail: Optional[str] = None
    discharge_detail: Optional[str] = None

    def describe(self) -> str:
        if self.dtype == DiscrepancyType.OMISSION:
            return f"{self.ingredient} present at admission, absent at discharge"
        if self.dtype == DiscrepancyType.ADDITION:
            return f"{self.ingredient} newly present at discharge"
        if self.dtype == DiscrepancyType.DOSE_CONFLICT:
            return (f"{self.ingredient} dose differs: "
                    f"{self.source_detail} -> {self.discharge_detail}")
        if self.dtype == DiscrepancyType.FREQUENCY_CONFLICT:
            return (f"{self.ingredient} frequency differs: "
                    f"{self.source_detail} -> {self.discharge_detail}")
        return f"{self.ingredient} appears duplicated"


@dataclass
class Interaction:
    """A flagged drug-drug interaction (a review flag, not a determination)."""
    ingredient_a: str
    ingredient_b: str
    severity: str                      # "minor" | "moderate" | "major"
    note: str = ""


@dataclass
class Admission:
    """The three input views for one admission, mirroring MIMIC-III structure."""
    hadm_id: str                       # hospital admission id
    subject_id: str                    # patient id (used for leakage-safe splits)
    admission_note: str = ""
    discharge_summary: str = ""
    # Structured order rows, each a dict with keys: drug, dose, route, frequency
    orders: list[dict] = field(default_factory=list)


@dataclass
class ReconciliationResult:
    """The full output of the pipeline for one admission."""
    hadm_id: str
    reconciled: list[MedicationMention]
    discrepancies: list[Discrepancy]
    interactions: list[Interaction]
    score: float                       # composite reconciliation score R
    score_components: dict             # {"A":.., "N":.., "C":..}
    rework_rounds: int = 0
    # Stage-level self-confidence, for error localization:
    agent_confidence: dict = field(default_factory=dict)
