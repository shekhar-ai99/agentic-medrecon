"""FHIR serialization.

Serializes a ReconciliationResult into FHIR MedicationStatement and
DetectedIssue resources, with the composite reconciliation score exposed as a
resource extension. This mirrors Appendix C of the paper. No resource produced
here is transmitted to any clinical system; this is a serialization format only.
"""
from __future__ import annotations

import json

from .types import ReconciliationResult

RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"
_EXT = "http://example.org/fhir/StructureDefinition"


def to_fhir_bundle(result: ReconciliationResult) -> dict:
    entries: list[dict] = []

    for m in result.reconciled:
        entries.append({
            "resourceType": "MedicationStatement",
            "status": "active",
            "medicationCodeableConcept": {
                "coding": [{
                    "system": RXNORM_SYSTEM,
                    "code": m.rxnorm_cui,
                    "display": m.rxnorm_ingredient,
                }]
            },
            "extension": [
                {"url": f"{_EXT}/recon-score", "valueDecimal": result.score},
                {"url": f"{_EXT}/source-provenance",
                 "valueString": m.source.value},
            ],
        })

    for d in result.discrepancies:
        entries.append({
            "resourceType": "DetectedIssue",
            "status": "preliminary",
            "severity": _severity_band(d.severity),
            "code": {"text": d.dtype.value},
            "detail": d.describe(),
        })

    for it in result.interactions:
        entries.append({
            "resourceType": "DetectedIssue",
            "status": "preliminary",
            "severity": it.severity,
            "code": {"text": "drug-drug-interaction"},
            "detail": f"{it.ingredient_a} + {it.ingredient_b}: {it.note}",
        })

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "id": f"recon-{result.hadm_id}",
        "entry": [{"resource": r} for r in entries],
    }


def _severity_band(sev: float) -> str:
    if sev >= 0.75:
        return "high"
    if sev >= 0.45:
        return "moderate"
    return "low"


def to_json(result: ReconciliationResult, indent: int = 2) -> str:
    return json.dumps(to_fhir_bundle(result), indent=indent)
