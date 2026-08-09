"""Normalization Agent.

Maps each raw mention onto an RxNorm ingredient-level concept. Strategy, in
order: (1) direct lexical match against the RxNorm graph, (2) UMLS synonym
expansion then retry, (3) LLM disambiguation for residual ambiguity. Here (1)
and (2) run against the synthetic tables in `knowledge.py`; step (3) is a
deterministic stub you can replace with a real LLM call. Coverage N (the
fraction of mentions mapped) is reported and feeds the composite score.
"""
from __future__ import annotations

import re

from ..knowledge import RXNORM, SYNONYMS
from ..types import MedicationMention


class NormalizationAgent:
    name = "normalization"

    def run(self, mentions: list[MedicationMention],
            widen: bool = False) -> tuple[list[MedicationMention], float]:
        """Map mentions to RxNorm ingredient level.

        widen=True is the "widened instruction" the orchestrator issues on
        re-work: it enables the more aggressive fuzzy disambiguation pass that
        the first pass skips (to keep first-pass precision high). This lets
        re-work recover a fraction of otherwise-unmappable mentions, which is
        why orchestration improves coverage on noisy admissions.
        """
        mapped = 0
        for m in mentions:
            if m.normalized:                 # already resolved on a prior pass
                mapped += 1
                continue

            key = self._clean(m.surface)

            # (1) direct lexical match
            hit = RXNORM.get(key)

            # (2) synonym expansion then retry
            if hit is None and key in SYNONYMS:
                hit = RXNORM.get(SYNONYMS[key])

            # (2b) try dropping a trailing salt/strength token
            if hit is None:
                base = key.split()[0] if key else ""
                hit = RXNORM.get(base)
                if hit is None and base in SYNONYMS:
                    hit = RXNORM.get(SYNONYMS[base])

            # (3) fuzzy disambiguation -- only on widened (re-work) passes
            if hit is None and widen:
                hit = self._disambiguate(key)

            if hit is not None:
                m.rxnorm_cui, m.rxnorm_ingredient = hit
                mapped += 1

        coverage = round(mapped / len(mentions), 3) if mentions else 1.0
        return mentions, coverage

    @staticmethod
    def _clean(surface: str) -> str:
        s = surface.lower().strip()
        # strip dose/route/frequency fragments that ride along on the surface
        s = re.sub(r"\d+(\.\d+)?\s?(mg|mcg|g|units?|ml)", "", s)
        s = re.sub(r"\b(po|iv|im|sc|sl|pr|bid|tid|qid|qd|qhs|prn|daily)\b", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # An extended ingredient table the FUZZY (widened) pass can reach but the
    # fast first pass does not. Mirrors an LLM disambiguation step that resolves
    # harder mentions at higher cost -- which is why re-work recovers coverage.
    _EXTENDED = {
        "metformin": ("6809", "metformin"),
        "hydrochlorothiazide": ("5487", "hydrochlorothiazide"),
        "gabapentin": ("25480", "gabapentin"),
        "levothyroxine": ("10582", "levothyroxine"),
        "pantoprazole": ("40790", "pantoprazole"),
        "carvedilol": ("20352", "carvedilol"),
        "sertraline": ("36437", "sertraline"),
        "montelukast": ("88249", "montelukast"),
    }

    def _disambiguate(self, key: str) -> tuple[str, str] | None:
        """LLM-disambiguation stub for the widened (re-work) pass.

        Replace with a real model call conditioned on dose/route. Here it
        reaches an extended ingredient table via prefix match, recovering
        mentions the fast pass could not map.
        """
        if not key:
            return None
        base = key.split()[0]
        for name, val in self._EXTENDED.items():
            if base and (base in name or name in base):
                return val
        for surface, val in RXNORM.items():
            if surface in key or key in surface:
                return val
        return None
