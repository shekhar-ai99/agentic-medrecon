"""Reconciliation Orchestrator.

The coordination protocol. It sequences the five agents, computes inter-agent
consistency, triggers bounded targeted re-work when confidence is low, and
folds the signals into the composite reconciliation score

    R = w_A * A + w_N * N + w_C * C_bar

with default weights (0.4, 0.3, 0.3). The orchestrator is deliberately an
*ablatable* component: constructing MARS with `coordinate=False` runs the same
agents in fixed sequence with no consistency check and no re-work, which is the
"Pipeline (no orchestration)" baseline in the paper.
"""
from __future__ import annotations

from .agents import (ExtractionAgent, NormalizationAgent,
                     DiscrepancyAgent, InteractionAgent)
from .types import Admission, ReconciliationResult, MedicationMention, Source


class MARS:
    def __init__(self,
                 w_a: float = 0.4, w_n: float = 0.3, w_c: float = 0.3,
                 consistency_threshold: float = 0.8,
                 max_rework: int = 2,
                 coordinate: bool = True):
        assert abs((w_a + w_n + w_c) - 1.0) < 1e-6, "weights must sum to 1"
        self.w_a, self.w_n, self.w_c = w_a, w_n, w_c
        self.tau = consistency_threshold
        self.max_rework = max_rework
        self.coordinate = coordinate

        self.extraction = ExtractionAgent()
        self.normalization = NormalizationAgent()
        self.discrepancy = DiscrepancyAgent()
        self.interaction = InteractionAgent()

    def run(self, adm: Admission) -> ReconciliationResult:
        conf: dict[str, float] = {}

        mentions, conf["extraction"] = self.extraction.run(adm)
        mentions, coverage = self.normalization.run(mentions)
        conf["normalization"] = coverage
        discrepancies, conf["discrepancy"] = self.discrepancy.run(mentions)
        interactions, conf["interaction"] = self.interaction.run(mentions)

        rework = 0
        if self.coordinate:
            c_bar = self._consistency(mentions, discrepancies)
            # Targeted re-work: re-invoke the lowest-confidence stage with a
            # widened instruction. In this scaffold "widening" = retry
            # normalization with synonym+base fallback already enabled, which
            # can pick up a few more mappings on a second pass over residuals.
            while c_bar < self.tau and rework < self.max_rework:
                rework += 1
                # widened re-normalization recovers some residual mentions
                mentions, coverage = self.normalization.run(mentions, widen=True)
                conf["normalization"] = coverage
                discrepancies, conf["discrepancy"] = self.discrepancy.run(mentions)
                c_bar = self._consistency(mentions, discrepancies)
        else:
            # No orchestration: consistency still measured for the score, but
            # no re-work and no conflict resolution.
            c_bar = self._consistency(mentions, discrepancies)

        a = self._mean_agent_agreement(conf)
        n = conf["normalization"]
        score = self.w_a * a + self.w_n * n + self.w_c * c_bar
        if c_bar < self.tau:
            score *= 0.8   # penalty for residual inconsistency

        reconciled = [m for m in mentions
                      if m.source == Source.DISCHARGE_SUMMARY and m.normalized]

        return ReconciliationResult(
            hadm_id=adm.hadm_id,
            reconciled=reconciled,
            discrepancies=discrepancies,
            interactions=interactions,
            score=round(score, 3),
            score_components={"A": round(a, 3), "N": round(n, 3),
                              "C": round(c_bar, 3)},
            rework_rounds=rework,
            agent_confidence={k: round(v, 3) for k, v in conf.items()},
        )

    # --- scoring helpers ----------------------------------------------------

    def _consistency(self, mentions: list[MedicationMention],
                     discrepancies: list) -> float:
        """Mean pairwise agreement between agents on shared entities.

        We approximate the paper's inter-agent consistency by three checks:
          - extraction<->normalization: fraction of mentions that normalized
          - normalization<->discrepancy: every discrepancy references a
            normalized ingredient (should always hold; measures leakage)
          - internal: fraction of mentions carrying provenance (always 1 here)
        """
        if not mentions:
            return 1.0
        norm_rate = sum(1 for m in mentions if m.normalized) / len(mentions)

        known = {m.rxnorm_ingredient for m in mentions if m.normalized}
        if discrepancies:
            grounded = sum(1 for d in discrepancies if d.ingredient in known)
            ground_rate = grounded / len(discrepancies)
        else:
            ground_rate = 1.0

        prov_rate = sum(1 for m in mentions if m.source is not None) / len(mentions)
        # Un-normalized mentions are the primary signal that the agents lack a
        # shared, grounded view of the medication list. We make normalization
        # coverage the dominant term so that genuinely low-coverage admissions
        # fall below threshold and trigger targeted re-normalization; the other
        # two terms are secondary corroboration.
        return 0.8 * norm_rate + 0.15 * ground_rate + 0.05 * prov_rate

    @staticmethod
    def _mean_agent_agreement(conf: dict[str, float]) -> float:
        vals = list(conf.values())
        return sum(vals) / len(vals) if vals else 1.0
