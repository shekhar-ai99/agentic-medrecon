"""Reference-standard construction for REAL MIMIC-III.

This is the procedure described in the paper's "Reference Standard
Construction" subsection. It is a STUB: it documents the exact steps and
provides the skeleton, but the actual PhysioNet-credentialed data loading is
left to you (the data cannot be redistributed). The reference standard is
heuristic and structural -- NOT prospective clinician chart review -- and is
adequate only for the comparative, computational claims in the paper.

Steps:
  1. For each HADM_ID, pull the structured source list from PRESCRIPTIONS and
     the discharge medication list parsed from the discharge summary in
     NOTEEVENTS (CATEGORY = 'Discharge summary').
  2. Normalize BOTH lists to RxNorm ingredient level with an OFFLINE,
     high-precision mapper that is INDEPENDENT of the Normalization Agent under
     test (to avoid circularity). The NLM RxNorm REST API or a local RRF load
     both work.
  3. Label discrepancies by typed set-difference at ingredient level:
     omission / addition / dose_conflict / frequency_conflict.
  4. Weight each discrepancy by severity from a high-alert list + the DDI base.
  5. Enforce PATIENT-LEVEL disjoint 70/15/15 splits on SUBJECT_ID.

Outputs a JSONL file, one record per admission:
    {"hadm_id": ..., "subject_id": ..., "discrepancies": [...]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build(mimic_dir: Path, out_path: Path):
    raise NotImplementedError(
        "Fill in MIMIC-III loading here.\n"
        "  - PRESCRIPTIONS.csv.gz  -> structured source medication list\n"
        "  - NOTEEVENTS.csv.gz     -> discharge summaries (CATEGORY filter)\n"
        "Then normalize both sides independently to RxNorm and diff.\n"
        "See the module docstring for the full procedure. This stub exists so\n"
        "the repository documents the reference-standard method exactly, "
        "without redistributing credentialed data."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-dir", type=Path, required=True,
                    help="Directory with MIMIC-III CSVs (PhysioNet credentialed).")
    ap.add_argument("--out", type=Path, default=Path("data/reference.jsonl"))
    args = ap.parse_args()
    build(args.mimic_dir, args.out)


if __name__ == "__main__":
    main()
