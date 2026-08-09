"""Single-admission demo.

Runs MARS on one synthetic admission and prints every intermediate output --
the stage-level confidences, the typed discrepancies, the interaction flags,
the composite score with its components, and the FHIR bundle. This is the
"inspectability" the paper argues an agentic decomposition buys you.

Usage:
    python scripts/demo.py --seed 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mars.orchestrator import MARS
from mars.synthetic import make_admission
from mars.fhir import to_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    adm, truth = make_admission(args.seed)
    print("\n--- ADMISSION", adm.hadm_id, "(patient", adm.subject_id, ") ---")
    print("\n[Admission note]\n" + adm.admission_note)
    print("\n[Discharge summary]\n" + adm.discharge_summary)

    result = MARS(coordinate=True).run(adm)

    print("\n--- STAGE CONFIDENCES ---")
    for stage, c in result.agent_confidence.items():
        print(f"  {stage:<14} {c}")
    print(f"  rework rounds: {result.rework_rounds}")

    print("\n--- DETECTED DISCREPANCIES ---")
    for d in result.discrepancies:
        print(f"  [{d.dtype.value:<18} sev={d.severity:.2f}] {d.describe()}")

    print("\n--- GROUND TRUTH (synthetic) ---")
    for d in truth:
        print(f"  [{d.dtype.value:<18}] {d.ingredient}")

    print("\n--- INTERACTION FLAGS ---")
    for it in result.interactions:
        print(f"  [{it.severity:<8}] {it.ingredient_a} + {it.ingredient_b}: {it.note}")
    if not result.interactions:
        print("  (none)")

    print("\n--- RECONCILIATION SCORE ---")
    print(f"  R = {result.score}   components {result.score_components}")

    print("\n--- FHIR BUNDLE (truncated) ---")
    js = to_json(result)
    print("\n".join(js.splitlines()[:24]) + "\n  ...")


if __name__ == "__main__":
    main()
