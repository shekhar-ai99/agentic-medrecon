"""Knowledge resources for normalization and interaction screening.

In the paper these are RxNorm (a ~120K-concept graph), UMLS synonyms, and a
curated ~25K-pair drug-drug interaction base. Here we ship a small, fully
synthetic stand-in so the pipeline runs end to end with no credentialed
downloads. To use the real resources, replace `RXNORM`, `SYNONYMS`, and
`DDI_PAIRS` below with loaders over the actual RxNorm RRF files and your DDI
source. The rest of the pipeline is unchanged.
"""
from __future__ import annotations

# --- Synthetic RxNorm-like table: brand/surface -> (CUI, ingredient) -------
# Real RxNorm keys on RxCUI and resolves brand->ingredient via the concept
# graph; this flat map is a deliberate simplification for the demo.
RXNORM: dict[str, tuple[str, str]] = {
    # ingredient forms
    "acetaminophen": ("161", "acetaminophen"),
    "metoprolol": ("6918", "metoprolol"),
    "metoprolol tartrate": ("6918", "metoprolol"),
    "warfarin": ("11289", "warfarin"),
    "warfarin sodium": ("11289", "warfarin"),
    "aspirin": ("1191", "aspirin"),
    "lisinopril": ("29046", "lisinopril"),
    "atorvastatin": ("83367", "atorvastatin"),
    "furosemide": ("4603", "furosemide"),
    "insulin glargine": ("274783", "insulin glargine"),
    "omeprazole": ("7646", "omeprazole"),
    "amlodipine": ("17767", "amlodipine"),
    "clopidogrel": ("32968", "clopidogrel"),
    "heparin": ("5224", "heparin"),
    "ibuprofen": ("5640", "ibuprofen"),
    # common brand names -> ingredient
    "tylenol": ("161", "acetaminophen"),
    "coumadin": ("11289", "warfarin"),
    "lopressor": ("6918", "metoprolol"),
    "lipitor": ("83367", "atorvastatin"),
    "lasix": ("4603", "furosemide"),
    "lantus": ("274783", "insulin glargine"),
    "prilosec": ("7646", "omeprazole"),
    "norvasc": ("17767", "amlodipine"),
    "plavix": ("32968", "clopidogrel"),
    "motrin": ("5640", "ibuprofen"),
}

# --- UMLS-style synonym expansion: misc surface -> canonical key -----------
SYNONYMS: dict[str, str] = {
    "apap": "acetaminophen",
    "paracetamol": "acetaminophen",
    "asa": "aspirin",
    "acetylsalicylic acid": "aspirin",
    "hctz": "hydrochlorothiazide",
    "metop": "metoprolol",
}

# --- Curated DDI base: frozenset({ingredient,ingredient}) -> (severity,note)
DDI_PAIRS: dict[frozenset, tuple[str, str]] = {
    frozenset({"warfarin", "aspirin"}):
        ("major", "Increased bleeding risk (additive anticoagulation)"),
    frozenset({"warfarin", "clopidogrel"}):
        ("major", "Increased bleeding risk"),
    frozenset({"warfarin", "ibuprofen"}):
        ("major", "NSAID increases warfarin bleeding risk"),
    frozenset({"aspirin", "clopidogrel"}):
        ("moderate", "Dual antiplatelet therapy; monitor bleeding"),
    frozenset({"aspirin", "ibuprofen"}):
        ("moderate", "NSAID may blunt aspirin cardioprotection"),
    frozenset({"lisinopril", "furosemide"}):
        ("minor", "Monitor for hypotension and renal function"),
    frozenset({"heparin", "aspirin"}):
        ("major", "Additive bleeding risk"),
}

# --- High-alert medications: drive discrepancy severity weighting ---------
# Loosely modeled on ISMP high-alert medication classes.
HIGH_ALERT: dict[str, float] = {
    "warfarin": 0.95,
    "heparin": 0.95,
    "insulin glargine": 0.90,
    "clopidogrel": 0.75,
    "aspirin": 0.55,
    "metoprolol": 0.55,
    "furosemide": 0.50,
    "lisinopril": 0.45,
    "amlodipine": 0.40,
    "atorvastatin": 0.35,
    "omeprazole": 0.25,
    "acetaminophen": 0.20,
    "ibuprofen": 0.30,
}
DEFAULT_SEVERITY = 0.40


def severity_for(ingredient: str) -> float:
    """Severity weight for a discrepancy on this ingredient."""
    return HIGH_ALERT.get(ingredient, DEFAULT_SEVERITY)
