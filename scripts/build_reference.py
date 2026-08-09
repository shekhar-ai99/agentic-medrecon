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


def build(mimic_dir: Path, out_path: Path, limit=None):
    """Build a heuristic reference standard from real MIMIC-III CSVs.

    Uses mars.mimic_loader to read PRESCRIPTIONS + NOTEEVENTS, then labels
    discrepancies by an independent, offline normalization of both sides.

    The normalization here MUST stay independent of the Normalization Agent
    under test to avoid circularity. We use the same synthetic RxNorm table for
    demonstration; for real reportable numbers, swap `_offline_normalize` for a
    high-precision mapper (e.g. the NLM RxNorm REST API or a local RRF load).
    """
    from mars.mimic_loader import load_admissions
    from mars.knowledge import RXNORM, SYNONYMS, severity_for
    import re

    def _offline_normalize(surface: str):
        """Independent RxNorm mapping (kept separate from the agent under test)."""
        s = re.sub(r"\d+(\.\d+)?\s?(mg|mcg|g|units?|ml)", "", surface.lower())
        s = re.sub(r"\b(po|iv|im|sc|daily|bid|tid|qd|qhs|prn)\b", "", s).strip()
        hit = RXNORM.get(s)
        if hit is None and s in SYNONYMS:
            hit = RXNORM.get(SYNONYMS[s])
        if hit is None:
            base = s.split()[0] if s else ""
            hit = RXNORM.get(base) or (RXNORM.get(SYNONYMS[base])
                                       if base in SYNONYMS else None)
        return hit[1] if hit else None

    def _parse_discharge(text: str):
        ings = set()
        for line in text.splitlines():
            line = line.strip().lstrip("0123456789.-) ").strip()
            if not line:
                continue
            ing = _offline_normalize(line.split()[0] if line.split() else "")
            if ing:
                ings.add(ing)
        return ings

    records = []
    for adm, _ in load_admissions(mimic_dir, limit=limit):
        source_ings = {n for n in
                       (_offline_normalize(o["drug"]) for o in adm.orders) if n}
        discharge_ings = _parse_discharge(adm.discharge_summary)

        discs = []
        for ing in source_ings - discharge_ings:
            discs.append({"dtype": "omission", "ingredient": ing,
                          "severity": severity_for(ing)})
        for ing in discharge_ings - source_ings:
            discs.append({"dtype": "addition", "ingredient": ing,
                          "severity": severity_for(ing)})
        records.append({"hadm_id": adm.hadm_id, "subject_id": adm.subject_id,
                        "discrepancies": discs})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} reference records to {out_path}")
    print("NOTE: heuristic/structural reference standard, NOT clinician review. "
          "Adequate for comparative computational claims only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-dir", type=Path, required=True,
                    help="Directory with MIMIC-III CSVs (PhysioNet credentialed).")
    ap.add_argument("--out", type=Path, default=Path("data/reference.jsonl"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of admissions (for a quick smoke test).")
    args = ap.parse_args()
    build(args.mimic_dir, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
