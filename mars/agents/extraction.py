"""Extraction Agent.

Identifies medication mentions across the admission note, discharge summary,
and structured order table, attaching dose/route/frequency and a provenance
tag. In the paper the free-text pass is a fine-tuned BioBERT span tagger
followed by an LLM attribute/filter pass; here we use a transparent regex +
lexicon matcher over the synthetic notes so the demo runs with no model
weights. Swap `_extract_from_text` for a real tagger to use MIMIC-III notes.
"""
from __future__ import annotations

import re

from ..knowledge import RXNORM, SYNONYMS
from ..types import MedicationMention, Source, Admission

# Build a lexicon of known surface forms from the knowledge base.
_LEXICON = sorted(set(RXNORM.keys()) | set(SYNONYMS.keys()), key=len, reverse=True)

_DOSE = re.compile(r"(\d+(?:\.\d+)?\s?(?:mg|mcg|g|units?|ml))", re.I)
_ROUTE = re.compile(r"\b(PO|IV|IM|SC|SL|PR|topical)\b", re.I)
_FREQ = re.compile(r"\b(QD|BID|TID|QID|QHS|Q\d+H|PRN|daily|once daily|twice daily)\b", re.I)


class ExtractionAgent:
    name = "extraction"

    def run(self, adm: Admission) -> tuple[list[MedicationMention], float]:
        mentions: list[MedicationMention] = []

        # Free-text sources.
        for text, src in (
            (adm.admission_note, Source.ADMISSION_NOTE),
            (adm.discharge_summary, Source.DISCHARGE_SUMMARY),
        ):
            mentions.extend(self._extract_from_text(text, src))

        # Structured orders parse directly (no NER needed).
        proposed_from_text = len(mentions)
        for row in adm.orders:
            surface = str(row.get("drug", "")).strip()
            if not surface:
                continue
            mentions.append(MedicationMention(
                surface=surface,
                source=Source.ORDERS,
                dose=row.get("dose"),
                route=row.get("route"),
                frequency=row.get("frequency"),
            ))

        # Self-confidence: fraction of text-proposed spans that matched the
        # lexicon (a stand-in for the LLM filter survival rate in the paper).
        # Orders are treated as fully trusted.
        confidence = 1.0
        if proposed_from_text:
            matched = sum(1 for m in mentions
                          if m.source != Source.ORDERS)
            # every text mention we kept did match the lexicon, so approximate
            # confidence by coverage vs. a light heuristic token count.
            token_est = self._estimate_drug_tokens(adm)
            confidence = min(1.0, matched / token_est) if token_est else 1.0
        return mentions, round(confidence, 3)

    def _extract_from_text(self, text: str, src: Source) -> list[MedicationMention]:
        found: list[MedicationMention] = []
        low = text.lower()
        used_spans: list[tuple[int, int]] = []
        for surface in _LEXICON:
            start = 0
            while True:
                idx = low.find(surface, start)
                if idx == -1:
                    break
                span = (idx, idx + len(surface))
                if not any(s <= span[0] < e or s < span[1] <= e for s, e in used_spans):
                    window = text[idx: idx + len(surface) + 30]
                    found.append(MedicationMention(
                        surface=surface,
                        source=src,
                        dose=self._first(_DOSE, window),
                        route=self._first(_ROUTE, window),
                        frequency=self._first(_FREQ, window),
                    ))
                    used_spans.append(span)
                start = idx + len(surface)
        return found

    @staticmethod
    def _first(pattern: re.Pattern, text: str) -> str | None:
        m = pattern.search(text)
        return m.group(1) if m else None

    @staticmethod
    def _estimate_drug_tokens(adm: Admission) -> int:
        # crude: count capitalized medical-looking tokens as candidate drugs
        text = adm.admission_note + " " + adm.discharge_summary
        caps = re.findall(r"\b[A-Z][a-z]{3,}\b", text)
        return max(1, len(set(w.lower() for w in caps)))
