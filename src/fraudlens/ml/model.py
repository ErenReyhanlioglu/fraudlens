"""XGBoost fraud scorer — loads the trained artifact and computes scores + SHAP."""

from __future__ import annotations

import asyncio
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from fraudlens.core.exceptions import ModelNotLoadedError
from fraudlens.core.logging import get_logger
from fraudlens.schemas.transaction import ShapFeature

logger = get_logger(__name__)

# Number of top SHAP contributors to surface in the API response.
_TOP_N_SHAP = 10


class FraudScorer:
    """Wraps the trained XGBoost artifact for online fraud scoring.

    Usage:
        scorer = FraudScorer()
        scorer.load(Path("data/processed/xgb_tuned_v1.joblib"))
        prob, shap_features = scorer.score(feature_df)
    """

    def __init__(self) -> None:
        self._model: xgb.XGBClassifier | xgb.Booster | None = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: Path) -> None:
        """Load the joblib artifact from *path*.

        Accepts either a bare XGBoost `Booster` or an `XGBClassifier`
        (scikit-learn API). Feature names are extracted from the booster
        so the scorer can validate incoming DataFrames.
        """
        if not path.exists():
            raise ModelNotLoadedError(
                f"Model artifact not found: {path}",
                details={"path": str(path)},
            )
        raw = joblib.load(path)

        if isinstance(raw, xgb.Booster):
            self._model = raw
            self._feature_names = raw.feature_names or []
        elif isinstance(raw, xgb.XGBClassifier):
            self._model = raw
            booster: xgb.Booster = raw.get_booster()
            self._feature_names = booster.feature_names or []
        else:
            raise ModelNotLoadedError(
                f"Unexpected artifact type: {type(raw).__name__}",
                details={"type": type(raw).__name__},
            )

        logger.info(
            "fraud_scorer_loaded",
            path=str(path),
            n_features=len(self._feature_names),
        )

    # ------------------------------------------------------------------
    # Scoring (sync)
    # ------------------------------------------------------------------

    def score(self, features: pd.DataFrame) -> tuple[float, list[ShapFeature]]:
        """Score a single transaction.

        Args:
            features: DataFrame with exactly one row containing model features.

        Returns:
            Tuple of (fraud_probability, top_shap_features).

        Raises:
            ModelNotLoadedError: If `load()` has not been called.
        """
        if self._model is None:
            raise ModelNotLoadedError("FraudScorer has not been loaded yet.")

        # Align columns to model's expected feature order.
        if self._feature_names:
            features = features.reindex(columns=self._feature_names, fill_value=np.nan)

        # XGBoost rejects object-dtype columns (e.g. un-encoded strings that
        # slipped through).  Coerce everything to float; unparseable → NaN.
        if features.select_dtypes(include="object").shape[1] > 0:
            features = features.apply(pd.to_numeric, errors="coerce")

        if isinstance(self._model, xgb.XGBClassifier):
            prob: float = float(self._model.predict_proba(features)[0, 1])
        else:
            dmatrix = xgb.DMatrix(features)
            prob = float(self._model.predict(dmatrix)[0])

        shap_features = self._compute_shap(features)
        return prob, shap_features

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    @cached_property
    def _explainer(self) -> shap.TreeExplainer:
        if self._model is None:
            raise ModelNotLoadedError("FraudScorer has not been loaded yet.")
        booster = self._model.get_booster() if isinstance(self._model, xgb.XGBClassifier) else self._model
        return shap.TreeExplainer(booster)

    def _compute_shap(self, features: pd.DataFrame) -> list[ShapFeature]:
        """Return the top-N SHAP contributors for a single row."""
        try:
            shap_vals: np.ndarray = self._explainer.shap_values(features)
            # shap_values returns shape (1, n_features) for a single row.
            row_shap = shap_vals[0] if shap_vals.ndim == 2 else shap_vals  # type: ignore[union-attr]
            feature_values = features.iloc[0].values

            top_idx = np.argsort(np.abs(row_shap))[::-1][:_TOP_N_SHAP]
            col_names = list(features.columns)

            return [
                ShapFeature(
                    feature=col_names[i],
                    value=float(feature_values[i]),
                    contribution=float(row_shap[i]),
                )
                for i in top_idx
            ]
        except Exception as exc:
            logger.warning("shap_computation_failed", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Async wrapper for use inside FastAPI
    # ------------------------------------------------------------------

    async def score_async(self, features: pd.DataFrame) -> tuple[float, list[ShapFeature]]:
        """Run `score` in the default thread executor (CPU-bound work)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.score, features)

    # ------------------------------------------------------------------
    # Raw scoring — bypasses InferenceExtractor (demo / test use)
    # ------------------------------------------------------------------

    def score_raw(self, features: dict) -> tuple[float, list[ShapFeature]]:
        """Score using a raw IEEE-CIS feature dict, skipping InferenceExtractor.

        Useful when test.parquet features are passed directly (e.g. demo mode
        or integration tests) to verify the full triage range is reachable.

        Args:
            features: Flat dict mapping feature names to numeric values.
                      Missing features are filled with NaN.

        Returns:
            Tuple of (fraud_probability, top_shap_features).
        """
        if self._model is None:
            raise ModelNotLoadedError("FraudScorer has not been loaded yet.")

        df = pd.DataFrame([features])
        if self._feature_names:
            df = df.reindex(columns=self._feature_names, fill_value=np.nan)
        if df.select_dtypes(include="object").shape[1] > 0:
            df = df.apply(pd.to_numeric, errors="coerce")

        if isinstance(self._model, xgb.XGBClassifier):
            prob: float = float(self._model.predict_proba(df)[0, 1])
        else:
            dmatrix = xgb.DMatrix(df)
            prob = float(self._model.predict(dmatrix)[0])

        shap_features = self._compute_shap(df)
        return prob, shap_features

    async def score_raw_async(self, features: dict) -> tuple[float, list[ShapFeature]]:
        """Run `score_raw` in the default thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.score_raw, features)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def build_feature_row(self, values: dict[str, Any]) -> pd.DataFrame:
        """Construct a single-row DataFrame from a plain dict.

        Missing features are filled with NaN so the model handles them
        via its trained missing-value strategy.
        """
        return pd.DataFrame([values])
