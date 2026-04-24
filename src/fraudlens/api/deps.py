"""Shared application-level singletons.

Centralises scorer and extractor so both main.py and routers can import
them without creating a circular dependency.
"""

from __future__ import annotations

from fraudlens.ml.feature_extractor import InferenceExtractor
from fraudlens.ml.model import FraudScorer

scorer: FraudScorer = FraudScorer()
extractor: InferenceExtractor = InferenceExtractor()