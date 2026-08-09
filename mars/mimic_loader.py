"""Real MIMIC-III loader.

Reads genuine MIMIC-III CSVs (PRESCRIPTIONS + NOTEEVENTS) into the same
`Admission` objects the synthetic generator produces, so the rest of the
pipeline runs unchanged on real data.

IMPORTANT SCHEMA NOTE
---------------------
The OPEN MIMIC-III demo (mimiciii-demo/1.4) has the NOTEEVENTS table REMOVED
-- every row deleted. Since reconciliation reads the discharge medication list
from discharge summaries in NOTEEVENTS, the open demo can exercise only the
PRESCRIPTIONS (structured orders) half of this loader, not the notes half.
Full, credentialed MIMIC-III (or MIMIC-IV) is required for the complete
pipeline and for any reportable numbers. This loader targets the full-DB
schema; on the demo it will find zero discharge summaries and say so.

Real column names (verified against the MIMIC-III schema):
  PRESCRIPTIONS: SUBJECT_ID, HADM_ID, DRUG, DOSE_VAL_RX, DOSE_UNIT_RX,
                 ROUTE, STARTDATE, ENDDATE
  NOTEEVENTS:    SUBJECT_ID, HADM_ID, CATEGORY, DESCRIPTION, TEXT
                 (filter CATEGORY == 'Discharge summary')

No MIMIC data is bundled with this repo (see .gitignore). Point --mimic-dir at
your local, credentialed copy.
"""
from __future__ import annotations

import csv
import gzip
import io
import re
from pathlib import Path

from .types import Admission

# Larger CSV fields (clinical notes) exceed the default limit.
csv.field_size_limit(10 * 1024 * 1024)


def _open(path: Path):
    """Open a MIMIC CSV whether it is plain .csv or gzipped .csv.gz."""
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def _find(mimic_dir: Path, stem: str) -> Path | None:
    """MIMIC files appear as UPPER/lower case and .csv or .csv.gz."""
    for name in (f"{stem}.csv", f"{stem}.csv.gz",
                 f"{stem.lower()}.csv", f"{stem.lower()}.csv.gz"):
        p = mimic_dir / name
        if p.exists():
            return p
    return None


def load_prescriptions(mimic_dir: Path) -> dict[str, list[dict]]:
    """HADM_ID -> list of structured order rows (drug/dose/route/frequency)."""
    path = _find(mimic_dir, "PRESCRIPTIONS")
    if path is None:
        raise FileNotFoundError(
            f"PRESCRIPTIONS.csv[.gz] not found in {mimic_dir}")

    by_hadm: dict[str, list[dict]] = {}
    with _open(path) as fh:
        reader = csv.DictReader(fh)
        # normalize header case
        fields = {k.upper(): k for k in (reader.fieldnames or [])}
        for row in reader:
            def g(col):
                key = fields.get(col)
                return (row.get(key) or "").strip() if key else ""

            hadm = g("HADM_ID")
            drug = g("DRUG")
            if not hadm or not drug:
                continue
            dose = g("DOSE_VAL_RX")
            unit = g("DOSE_UNIT_RX")
            dose_str = f"{dose} {unit}".strip() if dose else None
            by_hadm.setdefault(hadm, []).append({
                "drug": drug,
                "dose": dose_str,
                "route": g("ROUTE") or None,
                # PRESCRIPTIONS has no explicit frequency column; left None.
                "frequency": None,
                "subject_id": g("SUBJECT_ID"),
            })
    return by_hadm


def load_discharge_notes(mimic_dir: Path) -> dict[str, str]:
    """HADM_ID -> discharge-summary text (first discharge summary per admission).

    Returns an empty dict on the open demo, where NOTEEVENTS has no rows.
    """
    path = _find(mimic_dir, "NOTEEVENTS")
    if path is None:
        return {}   # demo: table absent entirely

    notes: dict[str, str] = {}
    with _open(path) as fh:
        reader = csv.DictReader(fh)
        fields = {k.upper(): k for k in (reader.fieldnames or [])}
        for row in reader:
            def g(col):
                key = fields.get(col)
                return (row.get(key) or "").strip() if key else ""

            if g("CATEGORY") != "Discharge summary":
                continue
            hadm = g("HADM_ID")
            if not hadm or hadm in notes:
                continue          # keep first discharge summary per admission
            notes[hadm] = g("TEXT")
    return notes


# Discharge summaries contain a "Discharge Medications:" section; we slice it
# out so the Extraction Agent sees the discharge list, not the whole note.
_DISCHARGE_MED_HEADER = re.compile(
    r"discharge\s+medications?\s*:?", re.IGNORECASE)
_NEXT_SECTION = re.compile(
    r"\n\s*[A-Z][A-Za-z ]{2,40}:\s*\n")


def extract_discharge_med_section(text: str) -> str:
    """Return the Discharge Medications section, or the whole note if absent."""
    m = _DISCHARGE_MED_HEADER.search(text)
    if not m:
        return text
    tail = text[m.end():]
    nxt = _NEXT_SECTION.search(tail)
    section = tail[:nxt.start()] if nxt else tail
    return "Discharge medications:\n" + section.strip()


def load_admissions(mimic_dir: str | Path, limit: int | None = None):
    """Yield (Admission, None) pairs from real MIMIC-III CSVs.

    Ground truth is None here -- build it separately with
    scripts/build_reference.py, which normalizes both sides independently. This
    keeps the loader (data plumbing) separate from label construction.
    """
    mimic_dir = Path(mimic_dir)
    prescriptions = load_prescriptions(mimic_dir)
    notes = load_discharge_notes(mimic_dir)

    if not notes:
        print("WARNING: no discharge summaries found (NOTEEVENTS empty or "
              "absent). This is expected on the OPEN MIMIC-III DEMO, which "
              "ships PRESCRIPTIONS but not NOTEEVENTS. The discharge side of "
              "each admission will be empty; only orders-parsing is exercised. "
              "Use full credentialed MIMIC-III for the complete pipeline.")

    hadms = list(prescriptions.keys())
    if limit:
        hadms = hadms[:limit]

    for hadm in hadms:
        orders = prescriptions[hadm]
        subject = orders[0].get("subject_id", "") if orders else ""
        discharge_text = notes.get(hadm, "")
        discharge_summary = (extract_discharge_med_section(discharge_text)
                             if discharge_text else "")
        # The admission note is approximated by the structured orders here;
        # on full MIMIC-III you may also pull an admission-note category.
        adm = Admission(
            hadm_id=str(hadm),
            subject_id=str(subject),
            admission_note="",            # optional: add a note category
            discharge_summary=discharge_summary,
            orders=[{k: v for k, v in o.items() if k != "subject_id"}
                    for o in orders],
        )
        yield adm, None
