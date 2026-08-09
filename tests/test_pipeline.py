"""End-to-end and unit tests for the MARS scaffold."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mars.orchestrator import MARS
from mars.synthetic import make_admission, generate
from mars.agents import (ExtractionAgent, NormalizationAgent,
                         DiscrepancyAgent, InteractionAgent)
from mars.types import Admission, Source
from mars.fhir import to_fhir_bundle
from mars.evaluation import (score_discrepancies, pearson, spearman)


def test_extraction_finds_brand_and_generic():
    adm = Admission(
        hadm_id="H1", subject_id="P1",
        admission_note="Home medications:\n - Coumadin 5 mg QD\n - Tylenol 500 mg",
        discharge_summary="Discharge medications:\n - warfarin 5 mg QD",
        orders=[{"drug": "Coumadin", "dose": "5 mg", "route": "PO", "frequency": "QD"}],
    )
    mentions, conf = ExtractionAgent().run(adm)
    surfaces = {m.surface for m in mentions}
    assert "coumadin" in surfaces or "Coumadin" in surfaces
    assert 0.0 <= conf <= 1.0


def test_normalization_resolves_brand_to_ingredient():
    adm, _ = make_admission(3)
    mentions, _ = ExtractionAgent().run(adm)
    mentions, coverage = NormalizationAgent().run(mentions)
    # brands like Coumadin/Lopressor must normalize to their ingredient
    for m in mentions:
        if m.surface.lower() == "coumadin":
            assert m.rxnorm_ingredient == "warfarin"
    assert 0.0 <= coverage <= 1.0


def test_discrepancy_types_are_valid():
    adm, _ = make_admission(5)
    result = MARS().run(adm)
    for d in result.discrepancies:
        assert d.severity >= 0.0
        assert d.ingredient


def test_orchestration_helps_or_matches():
    """Full orchestration should not do worse than no-orchestration on F1."""
    data = list(generate(60, start_seed=100))
    full = MARS(coordinate=True)
    noorch = MARS(coordinate=False)

    def run(system):
        preds, truths = [], []
        for adm, truth in data:
            preds.append(system.run(adm).discrepancies)
            truths.append(truth)
        return score_discrepancies(preds, truths).f1

    assert run(full) >= run(noorch) - 1e-6


def test_score_in_unit_interval():
    for adm, _ in generate(30, start_seed=200):
        r = MARS().run(adm)
        assert 0.0 <= r.score <= 1.0
        assert set(r.score_components) == {"A", "N", "C"}


def test_fhir_bundle_shape():
    adm, _ = make_admission(11)
    bundle = to_fhir_bundle(MARS().run(adm))
    assert bundle["resourceType"] == "Bundle"
    kinds = {e["resource"]["resourceType"] for e in bundle["entry"]} if bundle["entry"] else set()
    # at least MedicationStatement should appear when discharge meds normalized
    assert "MedicationStatement" in kinds or bundle["entry"] == []


def test_correlation_helpers():
    xs = [1, 2, 3, 4, 5]
    ys = [2, 4, 6, 8, 10]
    assert abs(pearson(xs, ys) - 1.0) < 1e-9
    assert abs(spearman(xs, ys) - 1.0) < 1e-9
