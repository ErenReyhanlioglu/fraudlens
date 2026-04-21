"""Preprocessing utilities for tabular fraud data."""

import numpy as np
import pandas as pd

from fraudlens.core.logging import get_logger

logger = get_logger(__name__)


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns in place to shrink a DataFrame's memory footprint.

    For every column:
        - Integer columns are downcast to the smallest signed type that fits the
          observed range (``int8`` / ``int16`` / ``int32``; left as-is if a wider
          type is needed).
        - Float columns are cast to ``float32``.
        - ``object`` columns are left untouched (category conversion is a
          separate decision that affects joins and downstream encoders).

    The transformation preserves all values — downcasting is skipped whenever
    the observed min/max would overflow the candidate type.

    Args:
        df: Input DataFrame. The operation mutates a copy; the original frame
            is not modified.
        verbose: When True, log memory usage before/after and percent saved via
            the project's structlog logger.

    Returns:
        A new DataFrame with downcast numeric dtypes and the same shape, index,
        and column order as ``df``.
    """
    out = df.copy()
    start_mb = out.memory_usage(deep=True).sum() / 1024**2

    int_types: list[np.dtype] = [np.dtype(t) for t in (np.int8, np.int16, np.int32)]

    for col in out.columns:
        dtype = out[col].dtype

        if pd.api.types.is_object_dtype(dtype):
            continue

        if pd.api.types.is_integer_dtype(dtype):
            col_min = out[col].min()
            col_max = out[col].max()
            for candidate in int_types:
                info = np.iinfo(candidate)
                if col_min >= info.min and col_max <= info.max:
                    out[col] = out[col].astype(candidate)
                    break
            continue

        if pd.api.types.is_float_dtype(dtype):
            out[col] = out[col].astype(np.float32)

    end_mb = out.memory_usage(deep=True).sum() / 1024**2
    saved_pct = (1 - end_mb / start_mb) * 100 if start_mb > 0 else 0.0

    if verbose:
        logger.info(
            "reduce_mem_usage",
            before_mb=round(start_mb, 2),
            after_mb=round(end_mb, 2),
            saved_mb=round(start_mb - end_mb, 2),
            saved_pct=round(saved_pct, 2),
            n_rows=len(out),
            n_cols=out.shape[1],
        )

    return out
