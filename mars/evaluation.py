"""Evaluation harness.

Scores the pipeline against a reference standard: discrepancy-detection
precision/recall/F1, normalization accuracy, severity-weighted recall, and the
Pearson/Spearman correlation between the composite reconciliation score and
reference-standard discrepancy severity. Pure-Python (no numpy dependency) so
it runs anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .types import Discrepancy, ReconciliationResult


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    severity_weighted_recall: float
    n_admissions: int
    n_true: int
    n_pred: int

    def __str__(self) -> str:
        return (f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} "
                f"SevWtdRecall={self.severity_weighted_recall:.3f} "
                f"(N={self.n_admissions}, true={self.n_true}, pred={self.n_pred})")


def _key(d: Discrepancy):
    return (d.dtype, d.ingredient)


def score_discrepancies(pred: list[list[Discrepancy]],
                        truth: list[list[Discrepancy]]) -> Metrics:
    tp = fp = fn = 0
    sev_caught = sev_total = 0.0
    n_true = n_pred = 0

    for p_list, t_list in zip(pred, truth):
        p_keys = {_key(d) for d in p_list}
        t_map = {_key(d): d for d in t_list}
        t_keys = set(t_map)
        n_true += len(t_keys)
        n_pred += len(p_keys)

        tp += len(p_keys & t_keys)
        fp += len(p_keys - t_keys)
        fn += len(t_keys - p_keys)

        for k, d in t_map.items():
            sev_total += d.severity
            if k in p_keys:
                sev_caught += d.severity

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    swr = sev_caught / sev_total if sev_total else 0.0
    return Metrics(precision, recall, f1, swr, len(pred), n_true, n_pred)


def normalization_accuracy(results: list[ReconciliationResult]) -> float:
    """Fraction of reconciled mentions that carry a normalized ingredient."""
    total = mapped = 0
    for r in results:
        for m in r.reconciled:
            total += 1
            if m.normalized:
                mapped += 1
    return mapped / total if total else 0.0


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if (vx and vy) else 0.0


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks
    return pearson(rank(xs), rank(ys))


def score_severity_correlation(results: list[ReconciliationResult],
                               truth: list[list[Discrepancy]]):
    """Correlate composite score with (inverse) reference-standard severity.

    A well-behaved score should be LOWER when an admission carries more severe
    unresolved discrepancies. We correlate the score against total truth
    severity and expect a negative Pearson r.
    """
    scores = [r.score for r in results]
    severities = [sum(d.severity for d in t) for t in truth]
    return pearson(scores, severities), spearman(scores, severities)
