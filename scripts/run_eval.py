"""Run the full evaluation on synthetic data and print a results table.

This reproduces the *structure* of Table (Overall performance) and the
ablation in the paper: it runs the full MARS system and the two LLM-free
baselines that this scaffold can express without model weights --

  - MARS (full)            : five agents + orchestration
  - Pipeline (no orch.)    : same agents, coordinate=False
  - Rule-based             : string-match normalization + raw set-difference

The absolute numbers here come from the SYNTHETIC generator, not MIMIC-III, so
they are illustrative of the harness, not the paper's reported results. Point
`load_admissions` at real MIMIC-III to get real numbers.

Usage:
    python scripts/run_eval.py --n 750 --seed 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mars.orchestrator import MARS
from mars.synthetic import generate
from mars.types import (Admission, Discrepancy, DiscrepancyType, Source)
from mars.evaluation import (score_discrepancies, normalization_accuracy,
                             score_severity_correlation)


def load_admissions(n: int, seed: int, noise: float = 0.0):
    """Load (Admission, ground_truth) pairs.

    Replace this function body with a real MIMIC-III loader:
      - read PRESCRIPTIONS for structured orders,
      - read NOTEEVENTS for admission/discharge notes,
      - build ground truth with scripts/build_reference.py.
    """
    return list(generate(n, start_seed=seed, noise=noise))


def rule_based_baseline(adm: Admission):
    """Non-LLM baseline: fuzzy string-match normalization + set-difference.

    Deliberately weaker than the agents: it does not resolve brand->generic,
    so brand/generic pairs look like distinct drugs and inflate false diffs.
    """
    def crude_norm(surface: str) -> str:
        return surface.lower().split()[0] if surface else ""

    src, dis = {}, {}
    for row in adm.orders:
        src[crude_norm(row.get("drug", ""))] = row
    # parse discharge lines
    for line in adm.discharge_summary.splitlines():
        line = line.strip()
        if line.startswith("- "):
            tok = crude_norm(line[2:])
            dis[tok] = line
    preds = []
    for k in src:
        if k and k not in dis:
            preds.append(Discrepancy(DiscrepancyType.OMISSION, k, 0.4))
    for k in dis:
        if k and k not in src:
            preds.append(Discrepancy(DiscrepancyType.ADDITION, k, 0.4))
    return preds


def evaluate(system_name: str, run_one, admissions, collect_results=False):
    preds, truths, results = [], [], []
    t0 = time.perf_counter()
    for adm, truth in admissions:
        out = run_one(adm)
        # `run_one` may return a ReconciliationResult (MARS) or a bare list of
        # discrepancies (rule-based baseline). Normalize to discrepancy lists.
        if hasattr(out, "discrepancies"):
            if collect_results:
                results.append(out)
            preds.append(out.discrepancies)
        else:
            preds.append(out)
        truths.append(truth)
    elapsed = time.perf_counter() - t0

    m = score_discrepancies(preds, truths)
    line = f"{system_name:<24} {m}  [{elapsed*1000/len(admissions):.1f} ms/adm]"
    return line, results, truths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=750)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.4,
                    help="Fraction of admissions with hard-to-normalize meds "
                         "(mirrors MIMIC-III messiness; exercises orchestration).")
    args = ap.parse_args()

    admissions = load_admissions(args.n, args.seed, noise=args.noise)
    print(f"\nLoaded {len(admissions)} admissions "
          f"({len({a.subject_id for a, _ in admissions})} patients)\n")
    print("=" * 78)

    mars_full = MARS(coordinate=True)
    line, results_full, truths = evaluate("MARS (full)", mars_full.run,
                                          admissions, collect_results=True)
    print(line)

    mars_noorch = MARS(coordinate=False)
    line2, results_noorch, _ = evaluate("Pipeline (no orch.)", mars_noorch.run,
                                        admissions, collect_results=True)
    print(line2)

    line3, _, _ = evaluate("Rule-based", rule_based_baseline, admissions)
    print(line3)

    print("=" * 78)

    # Where orchestration genuinely helps on this data: normalization coverage
    # (via widened re-work) and the resulting score. Discrepancy F1 is equal
    # here BY CONSTRUCTION -- the synthetic generator only injects discrepancies
    # on mappable drugs, so unmappable mentions (which re-work recovers) never
    # hide a discrepancy. On real MIMIC-III, where hard-to-normalize mentions DO
    # hide discrepancies, orchestration is expected to move F1 as well.
    def mean_cov(rs):
        return sum(r.score_components["N"] for r in rs) / len(rs)

    def mean_rework(rs):
        return sum(r.rework_rounds for r in rs) / len(rs)

    print(f"\nMean normalization coverage  full={mean_cov(results_full):.3f}  "
          f"no-orch={mean_cov(results_noorch):.3f}")
    print(f"Mean re-work rounds          full={mean_rework(results_full):.2f}  "
          f"no-orch={mean_rework(results_noorch):.2f}")

    norm_acc = normalization_accuracy(results_full)
    r, rho = score_severity_correlation(results_full, truths)
    print(f"MARS normalization accuracy  {norm_acc:.3f}")
    print(f"Score vs truth-severity      Pearson r={r:.3f}, Spearman rho={rho:.3f}")
    print("\nNote: on clean synthetic data the orchestration ablation is NOT")
    print("expected to favour the full system on discrepancy F1 -- the generator")
    print("only injects discrepancies on mappable drugs, so re-work (which")
    print("recovers UN-mappable mentions) cannot change the discrepancy set in")
    print("the intended direction, and can even add spurious diffs on some seeds.")
    print("The orchestration benefit here is in coverage; the paper's F1 gain is")
    print("a claim about real MIMIC-III. See README 'Known limitations'.\n")


if __name__ == "__main__":
    main()
