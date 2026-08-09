# MARS — Multi-Agent Reconciliation System

[![tests](https://github.com/shekhar-ai99/agentic-medrecon/actions/workflows/tests.yml/badge.svg)](https://github.com/shekhar-ai99/agentic-medrecon/actions/workflows/tests.yml)

A reference scaffold for the paper **"Agentic Clinical Decision Support: A
Multi-Agent LLM Architecture for Medication Reconciliation on MIMIC-III."**

MARS decomposes medication reconciliation across five specialized agents —
Extraction, Normalization, Discrepancy, Interaction, and a Reconciliation
Orchestrator — coordinated by an explicit protocol, so that every detected
discrepancy is traceable to the stage and source that produced it.

This repository **runs end to end on synthetic, MIMIC-III-shaped data with no
credentialed downloads and no model weights**, so the coordination design and
the evaluation methodology can be inspected and reproduced immediately. Real
MIMIC-III data and real model backbones drop in behind the same interfaces.

---

## Quick start

```bash
git clone https://github.com/shekhar-ai99/agentic-medrecon.git
cd agentic-medrecon
pip install -r requirements.txt        # only numpy-free stdlib + pytest

# Inspect one admission end to end (all intermediate agent outputs + FHIR):
python scripts/demo.py --seed 7

# Run the full evaluation on synthetic data:
python scripts/run_eval.py --n 750 --seed 0 --noise 0.7

# Run the tests:
python -m pytest tests/ -q
```

---

## Architecture

```
Admission (note + discharge summary + structured orders)
        │
        ▼
  Extraction Agent      → medication mentions + provenance
        │
        ▼
  Normalization Agent   → RxNorm ingredient mapping, coverage N
        │
        ▼
  Discrepancy Agent     → typed discrepancy set (omission / addition /
        │                  dose / frequency conflict / duplication)
        ▼
  Interaction Agent     → drug–drug interaction review flags
        │
        ▼
  Reconciliation Orchestrator
        │   • sequences agents
        │   • measures inter-agent consistency  C̄
        │   • triggers bounded targeted re-work when C̄ < τ
        │   • composite score  R = 0.4·A + 0.3·N + 0.3·C̄
        ▼
  FHIR MedicationStatement + DetectedIssue  (+ recon-score extension)
```

The orchestrator is an **ablatable** component. Construct `MARS(coordinate=False)`
to run the same agents in fixed sequence with no consistency check and no
re-work — this is the "Pipeline (no orchestration)" baseline in the paper.

---

## Repository layout

```
mars/
  types.py           Shared dataclasses (the contracts between agents)
  knowledge.py       Synthetic RxNorm-like graph, DDI base, high-alert list
  agents/
    extraction.py    Mention extraction from notes + structured orders
    normalization.py RxNorm ingredient mapping (+ widened re-work pass)
    discrepancy.py   Ingredient-level alignment → typed discrepancies
    interaction.py   DDI screening (review flags, not determinations)
  orchestrator.py    MARS — coordination protocol + composite score
  fhir.py            FHIR Bundle / MedicationStatement / DetectedIssue
  synthetic.py       MIMIC-III-shaped admission generator (+ ground truth)
  evaluation.py      P/R/F1, normalization acc, score correlation (stdlib only)
scripts/
  demo.py            Single-admission walkthrough with full inspectability
  run_eval.py        Full evaluation: MARS vs no-orch vs rule-based baseline
  build_reference.py Reference-standard builder STUB for real MIMIC-III
tests/
  test_pipeline.py   End-to-end + unit tests
configs/
  default.yaml       Hyperparameters mirroring the paper's Table
```

---

## Using real MIMIC-III

The scaffold is written so real data drops in behind the existing interfaces:

1. **Get credentialed access.** MIMIC-III requires CITI human-subjects training
   and a signed PhysioNet Data Use Agreement:
   https://physionet.org/content/mimiciii/1.4/
2. **Load real admissions.** Replace the body of `load_admissions()` in
   `scripts/run_eval.py` with a loader over `PRESCRIPTIONS.csv.gz` (structured
   orders) and `NOTEEVENTS.csv.gz` (admission + discharge summaries).
3. **Build the reference standard.** Fill in `scripts/build_reference.py`,
   which documents the exact heuristic procedure from the paper (independent
   offline RxNorm normalization of both sides, typed set-difference, severity
   weighting, patient-level 70/15/15 splits).
4. **Swap in real components** (optional): a fine-tuned BioBERT span tagger in
   `extraction.py`, a real RxNorm RRF load + LLM disambiguation in
   `normalization.py`, and your DDI source in `knowledge.py`.

None of these changes touch the orchestration logic or the evaluation harness.

---

## Known limitations (read before citing numbers)

- **The synthetic numbers are NOT the paper's numbers.** They come from the
  synthetic generator, not MIMIC-III, and are illustrative of the *harness*,
  not clinical performance. Run on real MIMIC-III to get reportable results.
- **On clean synthetic data, the orchestration ablation shows equal discrepancy
  F1** for full vs. no-orchestration. This is *by construction*: the generator
  only injects discrepancies on mappable drugs, so the hard-to-normalize
  mentions that re-work recovers never hide a discrepancy. The orchestration
  benefit is therefore visible here in **normalization coverage and score
  calibration** (run `run_eval.py` and compare the coverage / re-work lines),
  not in F1. On real MIMIC-III, where hard-to-normalize mentions genuinely hide
  discrepancies, orchestration is expected to move F1 as well — that is an
  empirical claim to be validated on real data, not something this scaffold
  can demonstrate.
- **The reference standard is heuristic and structural**, not prospective
  clinician chart review. It is adequate only for comparative, computational
  claims — never a substitute for clinical validation.
- **No clinical deployment.** This is a computational study. Nothing here has
  been tested with clinicians or patients or connected to a live EHR.

---

## Citation

If you use this code, please cite the paper (details to be finalized on
publication). DOI to be assigned.

## License

MIT — see `LICENSE`.
