"""MARS: Multi-Agent Reconciliation System.

A scaffold implementation of the agentic medication-reconciliation architecture
described in "Agentic Clinical Decision Support: A Multi-Agent LLM Architecture
for Medication Reconciliation on MIMIC-III".

The pipeline runs end to end on synthetic data (see mars.synthetic) with no
credentialed downloads and no model weights, so the coordination design and the
evaluation methodology can be inspected and reproduced immediately. Real
MIMIC-III data and real model backbones drop in behind the same interfaces.
"""
from .orchestrator import MARS
from .types import Admission, ReconciliationResult

__version__ = "0.1.0"
__all__ = ["MARS", "Admission", "ReconciliationResult"]
