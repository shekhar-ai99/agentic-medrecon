"""Synthetic admission generator.

Produces admissions that mirror the *shape* of MIMIC-III medication data: a
free-text admission note, a discharge summary, and a structured order table,
with brand/generic mixing, dose/frequency variation, and deliberately injected
discrepancies. This lets the full pipeline run and be evaluated with no
credentialed download. Swap `load_admissions` in `scripts/run_eval.py` for a
real MIMIC-III loader (PRESCRIPTIONS + NOTEEVENTS) to use the actual data.

Each admission is generated together with its GROUND-TRUTH discrepancy set, so
the synthetic generator doubles as the reference standard for the demo. On real
MIMIC-III you instead build the reference standard with
`scripts/build_reference.py`.
"""
from __future__ import annotations

import random

from .knowledge import RXNORM, severity_for
from .types import (Admission, Discrepancy, DiscrepancyType)

# A pool of (surface, ingredient) drawn from the knowledge base, favouring
# brand/generic pairs so normalization has real work to do.
_DRUG_POOL = [
    ("metoprolol", "metoprolol"), ("Lopressor", "metoprolol"),
    ("warfarin", "warfarin"), ("Coumadin", "warfarin"),
    ("aspirin", "aspirin"), ("ASA", "aspirin"),
    ("atorvastatin", "atorvastatin"), ("Lipitor", "atorvastatin"),
    ("furosemide", "furosemide"), ("Lasix", "furosemide"),
    ("lisinopril", "lisinopril"),
    ("amlodipine", "amlodipine"), ("Norvasc", "amlodipine"),
    ("omeprazole", "omeprazole"), ("Prilosec", "omeprazole"),
    ("clopidogrel", "clopidogrel"), ("Plavix", "clopidogrel"),
    ("Tylenol", "acetaminophen"), ("acetaminophen", "acetaminophen"),
    ("insulin glargine", "insulin glargine"), ("Lantus", "insulin glargine"),
]
_DOSES = ["5 mg", "10 mg", "25 mg", "40 mg", "500 mg", "81 mg", "20 units"]
_FREQS = ["QD", "BID", "TID", "QHS", "daily"]


# Surface forms that DON'T appear in the knowledge base, to simulate the
# fraction of real MIMIC-III mentions that resist normalization (misspellings,
# compound formulations, abbreviations). These lower coverage and, with
# orchestration on, trigger re-work.
_UNMAPPABLE = [
    "metformin XR", "hydrochlorothiazide", "gabapentin", "levothyroxine",
    "pantoprazole", "carvedilol", "sertraline", "montelukast",
]


def make_admission(seed: int, noise: float = 0.0) -> tuple[Admission, list[Discrepancy]]:
    """Generate one admission.

    noise in [0,1] controls the fraction of admissions that include mentions
    which resist normalization, mirroring the messiness of real MIMIC-III drug
    text. Higher noise -> lower normalization coverage -> orchestration re-work
    fires and reconciliation scores spread out.
    """
    rng = random.Random(seed)
    hadm_id = f"H{100000 + seed}"
    subject_id = f"P{200000 + seed // 3}"   # ~3 admissions per patient

    # Sample DISTINCT ingredients, then pick one surface form (brand or
    # generic) per ingredient. This prevents accidental same-ingredient
    # collisions (e.g. warfarin + Coumadin) that would create spurious
    # duplication discrepancies not present in the ground truth.
    by_ingredient: dict[str, list[str]] = {}
    for surface, ing in _DRUG_POOL:
        by_ingredient.setdefault(ing, []).append(surface)
    n_home = rng.randint(4, min(8, len(by_ingredient)))
    chosen_ings = rng.sample(list(by_ingredient), n_home)
    home_regimen = [(rng.choice(by_ingredient[ing]), ing,
                     rng.choice(_DOSES), rng.choice(_FREQS))
                    for ing in chosen_ings]

    truth: list[Discrepancy] = []
    discharge_regimen = list(home_regimen)

    # Inject an omission: drop one home med at discharge.
    if rng.random() < 0.7 and discharge_regimen:
        dropped = rng.randrange(len(discharge_regimen))
        d = discharge_regimen.pop(dropped)
        truth.append(Discrepancy(DiscrepancyType.OMISSION, d[1],
                                 severity_for(d[1])))

    # Inject an addition: add a new med at discharge.
    if rng.random() < 0.5:
        candidates = [p for p in _DRUG_POOL
                      if p[1] not in {r[1] for r in discharge_regimen}]
        if candidates:
            surface, ing = rng.choice(candidates)
            discharge_regimen.append((surface, ing, rng.choice(_DOSES),
                                      rng.choice(_FREQS)))
            truth.append(Discrepancy(DiscrepancyType.ADDITION, ing,
                                     severity_for(ing)))

    # Inject a dose conflict on a surviving med.
    if rng.random() < 0.4 and discharge_regimen:
        i = rng.randrange(len(discharge_regimen))
        surface, ing, dose, freq = discharge_regimen[i]
        new_dose = rng.choice([d for d in _DOSES if d != dose])
        discharge_regimen[i] = (surface, ing, new_dose, freq)
        truth.append(Discrepancy(DiscrepancyType.DOSE_CONFLICT, ing,
                                 severity_for(ing),
                                 source_detail=dose, discharge_detail=new_dose))

    # Noise: with probability `noise`, add SEVERAL unmappable mentions to BOTH
    # the source and discharge sides. They are not discrepancies (present on
    # both), but they drag first-pass normalization coverage down far enough to
    # drop inter-agent consistency below threshold, which triggers the
    # orchestrator's widened re-normalization (re-work). This is how the demo
    # exercises the coordination protocol the paper evaluates.
    if rng.random() < noise:
        k = rng.randint(2, 4)
        for junk in rng.sample(_UNMAPPABLE, min(k, len(_UNMAPPABLE))):
            jdose, jfreq = rng.choice(_DOSES), rng.choice(_FREQS)
            home_regimen.append((junk, junk, jdose, jfreq))
            discharge_regimen.append((junk, junk, jdose, jfreq))

    admission_note = _render_note("Home medications on admission:", home_regimen)
    discharge_summary = _render_note("Discharge medications:", discharge_regimen)
    orders = [{"drug": s, "dose": dose, "route": "PO", "frequency": f}
              for (s, ing, dose, f) in home_regimen]

    adm = Admission(hadm_id=hadm_id, subject_id=subject_id,
                    admission_note=admission_note,
                    discharge_summary=discharge_summary, orders=orders)
    return adm, _dedup(truth)


def _render_note(header: str, regimen) -> str:
    lines = [header]
    for surface, ing, dose, freq in regimen:
        lines.append(f"  - {surface} {dose} {freq}")
    lines.append("Patient stable. Continue as directed.")
    return "\n".join(lines)


def _dedup(truth: list[Discrepancy]) -> list[Discrepancy]:
    seen = set()
    out = []
    for d in truth:
        key = (d.dtype, d.ingredient)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def generate(n: int, start_seed: int = 0, noise: float = 0.0):
    """Yield (Admission, ground_truth_discrepancies) pairs."""
    for i in range(start_seed, start_seed + n):
        yield make_admission(i, noise=noise)
