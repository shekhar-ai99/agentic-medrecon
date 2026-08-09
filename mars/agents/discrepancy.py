"""Discrepancy Agent.

Aligns the normalized source-side list (admission note + orders) against the
normalized discharge list and emits a typed discrepancy set: omissions,
additions, dose conflicts, frequency conflicts, and duplications. Alignment is
exact at ingredient level after normalization -- which is why normalization
runs first. Each discrepancy carries a severity weight from the high-alert list.
"""
from __future__ import annotations

from collections import defaultdict

from ..knowledge import severity_for
from ..types import (MedicationMention, Discrepancy, DiscrepancyType, Source)


class DiscrepancyAgent:
    name = "discrepancy"

    def run(self, mentions: list[MedicationMention]) -> tuple[list[Discrepancy], float]:
        # Partition into source-side (admission note + orders) vs discharge.
        source = self._by_ingredient(
            [m for m in mentions
             if m.source in (Source.ADMISSION_NOTE, Source.ORDERS) and m.normalized])
        discharge = self._by_ingredient(
            [m for m in mentions
             if m.source == Source.DISCHARGE_SUMMARY and m.normalized])

        discrepancies: list[Discrepancy] = []
        all_ingredients = set(source) | set(discharge)

        for ing in sorted(all_ingredients):
            in_src = ing in source
            in_dis = ing in discharge

            if in_src and not in_dis:
                discrepancies.append(Discrepancy(
                    DiscrepancyType.OMISSION, ing, severity_for(ing),
                    source_detail=self._fmt(source[ing][0])))
            elif in_dis and not in_src:
                discrepancies.append(Discrepancy(
                    DiscrepancyType.ADDITION, ing, severity_for(ing),
                    discharge_detail=self._fmt(discharge[ing][0])))
            else:
                # present on both sides -> check dose / frequency / duplication
                s0 = source[ing][0]
                d0 = discharge[ing][0]
                # Duplication = same ingredient under TWO DISTINCT concepts
                # (RxCUIs) on the discharge list -- e.g. brand + generic both
                # ordered. Merely appearing in both the note and the orders
                # table (same concept) is NOT a duplication.
                distinct_discharge_cuis = {m.rxnorm_cui for m in discharge[ing]}
                if len(distinct_discharge_cuis) > 1:
                    discrepancies.append(Discrepancy(
                        DiscrepancyType.DUPLICATION, ing, severity_for(ing)))
                # A dose/frequency CONFLICT requires a value on BOTH sides that
                # differ. If one side is missing the field entirely (common in
                # real MIMIC-III: PRESCRIPTIONS has no frequency column), that
                # is missing data, not a conflict -- flagging it would inflate
                # discrepancy counts with false positives.
                if s0.dose and d0.dose and self._norm(s0.dose) != self._norm(d0.dose):
                    discrepancies.append(Discrepancy(
                        DiscrepancyType.DOSE_CONFLICT, ing, severity_for(ing),
                        source_detail=s0.dose, discharge_detail=d0.dose))
                if s0.frequency and d0.frequency and \
                        self._norm(s0.frequency) != self._norm(d0.frequency):
                    discrepancies.append(Discrepancy(
                        DiscrepancyType.FREQUENCY_CONFLICT, ing, severity_for(ing),
                        source_detail=s0.frequency, discharge_detail=d0.frequency))

        # Self-confidence: high when both sides had normalized content to align.
        confidence = 1.0 if (source and discharge) else 0.6
        return discrepancies, confidence

    @staticmethod
    def _by_ingredient(mentions: list[MedicationMention]) -> dict[str, list[MedicationMention]]:
        out: dict[str, list[MedicationMention]] = defaultdict(list)
        for m in mentions:
            out[m.rxnorm_ingredient].append(m)
        return out

    @staticmethod
    def _fmt(m: MedicationMention) -> str:
        parts = [m.rxnorm_ingredient]
        if m.dose:
            parts.append(m.dose)
        if m.frequency:
            parts.append(m.frequency)
        return " ".join(parts)

    @staticmethod
    def _norm(v: str | None) -> str:
        return (v or "").lower().replace(" ", "")
