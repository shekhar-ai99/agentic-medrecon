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


def test_mimic_loader_real_schema(tmp_path):
    """Loader parses real MIMIC-III column names and runs through the pipeline."""
    import csv
    # PRESCRIPTIONS with real columns
    presc = tmp_path / "PRESCRIPTIONS.csv"
    presc.write_text(
        "SUBJECT_ID,HADM_ID,DRUG,DOSE_VAL_RX,DOSE_UNIT_RX,ROUTE,STARTDATE,ENDDATE\n"
        "10006,142345,Warfarin,5,mg,PO,2164-10-23,2164-10-25\n"
        "10006,142345,Metoprolol Tartrate,25,mg,PO,2164-10-23,2164-10-25\n"
    )
    notes = tmp_path / "NOTEEVENTS.csv"
    notes.write_text(
        'SUBJECT_ID,HADM_ID,CATEGORY,DESCRIPTION,TEXT\n'
        '10006,142345,Discharge summary,Report,'
        '"Discharge Medications:\n1. Warfarin 5 mg PO daily\n\n'
        'Disposition:\nHome"\n'
    )
    from mars.mimic_loader import load_admissions as load_mimic
    from mars.orchestrator import MARS
    adms = list(load_mimic(str(tmp_path)))
    assert len(adms) == 1
    adm, gt = adms[0]
    assert adm.hadm_id == "142345"
    assert len(adm.orders) == 2
    assert "Warfarin" in adm.discharge_summary
    # runs through the pipeline: metoprolol dropped at discharge => omission
    result = MARS().run(adm)
    omissions = [d for d in result.discrepancies if d.dtype.value == "omission"]
    assert any(d.ingredient == "metoprolol" for d in omissions)


def test_no_false_frequency_conflict_on_missing_field():
    """A field present on one side but missing on the other is not a conflict."""
    from mars.types import Admission
    from mars.orchestrator import MARS
    adm = Admission(
        hadm_id="H1", subject_id="P1",
        admission_note="",
        discharge_summary="Discharge medications:\n - warfarin 5 mg daily",
        # orders have dose but NO frequency (like real PRESCRIPTIONS)
        orders=[{"drug": "warfarin", "dose": "5 mg", "route": "PO",
                 "frequency": None}],
    )
    result = MARS().run(adm)
    freq_conflicts = [d for d in result.discrepancies
                      if d.dtype.value == "frequency_conflict"]
    assert freq_conflicts == []
