"""Interaction Agent.

Screens the reconciled discharge list against the drug-drug interaction base
and emits review flags. Consistent with the alert-fatigue concern discussed in
the paper, these are flags for review, not determinations, and they feed the
composite score only weakly.
"""
from __future__ import annotations

from itertools import combinations

from ..knowledge import DDI_PAIRS
from ..types import MedicationMention, Interaction, Source


class InteractionAgent:
    name = "interaction"

    def run(self, mentions: list[MedicationMention]) -> tuple[list[Interaction], float]:
        # Unique normalized ingredients on the discharge list.
        ingredients = sorted({
            m.rxnorm_ingredient for m in mentions
            if m.source == Source.DISCHARGE_SUMMARY and m.normalized
        })
        flags: list[Interaction] = []
        for a, b in combinations(ingredients, 2):
            hit = DDI_PAIRS.get(frozenset({a, b}))
            if hit:
                severity, note = hit
                flags.append(Interaction(a, b, severity, note))

        # Confidence is high; screening is a deterministic lookup.
        return flags, 1.0
