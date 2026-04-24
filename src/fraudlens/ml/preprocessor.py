"""Preprocessing utilities for tabular fraud data.

Pipeline ordering:

    Pre-split (only row-local or deterministic operations):
        1. reduce_mem_usage
        2. join_transaction_identity
        3. normalize_d_features          # row-local arithmetic
        4. create_uid_feature            # row-local string concat

    Temporal split:
        5. apply_temporal_split          # returns (train, val, test)

    Post-split — every fit-on-train, apply-to-all-three:
        6.  compute_missing_strategy(train)        → apply_missing_strategy(df)
        7.  compute_low_variance_drop_list(train)  → apply_drop_columns(df)
        8.  compute_v_feature_drop_list(train)     → apply_drop_columns(df)
        9.  compute_aggregation_mappings(train)    → returns (train_with_oof, mappings)
                                                     apply_aggregation_mappings(val/test, mappings)
        10. compute_domain_mappings(train)         → apply_domain_mappings(df)
        11. apply_time_features                    # row-local; call per split
        12. compute_encoding_mappings(train)       → apply_encoding_mappings(df)
        13. compute_correlation_drop_list(train)   → apply_drop_columns(df)

Every ``compute_*`` returns a JSON-serializable mapping AND writes it to disk
so the same mapping can be reapplied at inference. Every ``apply_*`` is a pure
transformation. ``groupby`` / ``value_counts`` / ``corr`` NEVER runs on val or
test.

Step 9 combines two leakage / overfit guards that run together:

* **Bayesian smoothing** on every per-group mean (``alpha=50`` by default).
  Groups with few rows are shrunk toward the target's global mean, so tiny
  groups (``count=1..3``) no longer produce razor-sharp, overfit-prone
  local means. The same formula also defines the fallback for unseen keys
  in val/test — at ``count=0`` it collapses to ``global_mean``.
* **Expanding-window time-block OOF** on the listed ``oof_keys`` (default:
  all group keys). Train is sorted by ``TransactionDT`` and cut into
  ``n_blocks`` contiguous blocks; each block's aggregates are built from
  the blocks strictly before it. Block 0 has no history — mean columns
  fall back to ``global_mean``, which is consistent with how val treats
  unseen keys on its first days. No ``StratifiedKFold`` shuffle, so
  train's aggregates never peek at temporally-future rows within train.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from fraudlens.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_EXCLUDES: tuple[str, ...] = ("TransactionID", "isFraud", "uid")


# ---------------------------------------------------------------------------
# JSON / dtype helpers
# ---------------------------------------------------------------------------


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars to plain Python types.

    ``json.dumps`` does not accept ``np.float64``, ``np.int64``, or ``NaN``.
    This walks dicts / lists and coerces leaf values so any mapping produced
    by a ``compute_*`` function can be serialized without surprises.
    """
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _write_json(payload: Any, export_path: str | Path) -> None:
    """Write ``payload`` as indented UTF-8 JSON. Parents are created if missing."""
    path = Path(export_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_json_safe(payload), f, indent=2, ensure_ascii=False)


def _safe_str_fill(s: pd.Series, missing_token: str = "Unknown") -> pd.Series:
    """Stringify a series, mapping NaN to ``missing_token`` (not the literal ``'nan'``).

    ``s.astype(str)`` converts NaN to the string ``'nan'`` first, which then
    bypasses any subsequent ``fillna`` call. Doing the NaN-replacement first
    via ``.where`` avoids that trap and gives a true "Unknown" sentinel for
    label encoding.
    """
    return s.where(pd.notna(s), missing_token).astype(str)


# ===========================================================================
# Step 1 — memory optimization (row-local; pre-split safe)
# ===========================================================================


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to shrink the DataFrame's memory footprint.

    Integer columns are downcast to the smallest signed type (``int8`` /
    ``int16`` / ``int32``); float columns to ``float32``. Object columns
    are left untouched. Pure dtype manipulation, no statistics.
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
            saved_pct=round(saved_pct, 2),
            n_rows=len(out),
            n_cols=out.shape[1],
        )

    return out


# ===========================================================================
# Step 2 — identity join (row-local; pre-split safe)
# ===========================================================================


def join_transaction_identity(
    transaction: pd.DataFrame,
    identity: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Left-join the identity table onto the transaction table on ``TransactionID``."""
    for name, frame in (("transaction", transaction), ("identity", identity)):
        if "TransactionID" not in frame.columns:
            raise KeyError(f"{name} frame is missing required column 'TransactionID'")

    joined = transaction.merge(identity, on="TransactionID", how="left")

    if verbose:
        matched = identity["TransactionID"].isin(transaction["TransactionID"]).sum()
        match_rate = matched / len(transaction) if len(transaction) > 0 else 0.0
        fraud_rate = float(joined["isFraud"].mean()) if "isFraud" in joined.columns else float("nan")
        logger.info(
            "join_transaction_identity",
            shape=list(joined.shape),
            fraud_rate=round(fraud_rate, 6),
            identity_match_rate=round(match_rate, 6),
            identity_matched=int(matched),
        )

    return joined


# ===========================================================================
# Step 3 — D-feature normalization (row-local arithmetic; pre-split safe)
# ===========================================================================


def normalize_d_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Convert ``D1..D15`` deltas to absolute day references via ``TransactionDT``.

    For each row, ``D[i]_norm = TransactionDT // 86400 - D[i]``. Pure row-local
    arithmetic — no statistics, no leakage.
    """
    if "TransactionDT" not in df.columns:
        if verbose:
            logger.warning("normalize_d_features: TransactionDT missing; skipping")
        return df.copy()

    out = df.copy()
    transaction_day = out["TransactionDT"] // 86400

    new_cols: dict[str, pd.Series] = {}
    for i in range(1, 16):
        col = f"D{i}"
        if col in out.columns:
            new_cols[f"{col}_norm"] = transaction_day - out[col]

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1).copy()

    if verbose:
        logger.info("normalize_d_features", normalized_count=len(new_cols))

    return out


# ===========================================================================
# Step 4 — UID creation (row-local string concat; pre-split safe)
# ===========================================================================


def create_uid_feature(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Build a synthetic unique-identifier column from card / address / email / D1.

    UID approximates "a single card/account/user" and is used as a groupby key
    for behavioral aggregations in step 9. It is a feature-engineering artifact
    only — ``uid`` must be dropped from the final feature matrix before training.
    """
    required = ["card1", "addr1", "P_emaildomain", "TransactionDT", "D1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        if verbose:
            logger.error("create_uid_feature: missing required columns", missing=missing)
        return df.copy()

    out = df.copy()
    transaction_day = out["TransactionDT"] // 86400
    d1_norm = transaction_day - out["D1"]

    parts = [
        out["card1"].fillna("NaN").astype(str),
        out["addr1"].fillna("NaN").astype(str),
        out["P_emaildomain"].fillna("NaN").astype(str),
        d1_norm.fillna(-999).astype(str),
    ]
    out["uid"] = parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3]

    if verbose:
        logger.info(
            "create_uid_feature",
            n_rows=len(out),
            unique_uids=int(out["uid"].nunique()),
        )
    return out


# ===========================================================================
# Step 5 — temporal split (chronological; writes parquet files)
# ===========================================================================


def apply_temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    save_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by ``TransactionDT`` and split chronologically into train / val / test.

    Test ratio is implicitly ``1 - train_ratio - val_ratio``. If ``save_dir`` is
    provided, parquet files are written immediately. Defaults to ``None`` because
    the canonical save point is *after* all post-split feature engineering.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio + val_ratio must be < 1 (got {train_ratio + val_ratio:.3f})")
    if "TransactionDT" not in df.columns:
        raise KeyError("apply_temporal_split requires 'TransactionDT'")

    sorted_df = df.sort_values("TransactionDT").reset_index(drop=True)
    total_len = len(sorted_df)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    train = sorted_df.iloc[:train_end].copy()
    val = sorted_df.iloc[train_end:val_end].copy()
    test = sorted_df.iloc[val_end:].copy()

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for name, frame in (("train", train), ("val", val), ("test", test)):
            path = save_path / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            if verbose:
                logger.info(
                    "apply_temporal_split.saved",
                    split=name,
                    path=str(path),
                    shape=list(frame.shape),
                )

    if verbose:
        logger.info(
            "apply_temporal_split",
            train_shape=list(train.shape),
            val_shape=list(val.shape),
            test_shape=list(test.shape),
        )

    return train, val, test


# ===========================================================================
# Step 6 — missing-value handling (fit on TRAIN, apply to all)
# ===========================================================================


def compute_missing_strategy(
    train: pd.DataFrame,
    drop_threshold: float = 0.9,
    flag_threshold: float = 0.05,
    exclude_cols: Sequence[str] | None = None,
    export_path: str = "data/processed/missing_strategy.json",
    verbose: bool = True,
) -> dict[str, Any]:
    """Compute which columns to drop or flag, **using train missing-rates only**.

    Returns:
        ``{"flag_columns": [...], "drop_columns": [...], "thresholds": {...}}``
    """
    if not 0.0 <= drop_threshold <= 1.0:
        raise ValueError(f"drop_threshold must be in [0, 1], got {drop_threshold}")
    if not 0.0 <= flag_threshold <= 1.0:
        raise ValueError(f"flag_threshold must be in [0, 1], got {flag_threshold}")

    excluded = set(exclude_cols) if exclude_cols is not None else set(DEFAULT_EXCLUDES)
    missing_rates = train.isna().mean()

    flag_columns: list[str] = []
    drop_columns: list[str] = []

    for col in train.columns:
        if col in excluded:
            continue
        rate = float(missing_rates.get(col, 0.0))
        if rate > flag_threshold:
            flag_columns.append(col)
        if rate > drop_threshold:
            drop_columns.append(col)

    payload = {
        "flag_columns": flag_columns,
        "drop_columns": drop_columns,
        "thresholds": {"drop": drop_threshold, "flag": flag_threshold},
    }
    _write_json(payload, export_path)

    if verbose:
        logger.info(
            "compute_missing_strategy",
            n_flags=len(flag_columns),
            n_drops=len(drop_columns),
            drop_threshold=drop_threshold,
            flag_threshold=flag_threshold,
            export_path=str(export_path),
        )

    return payload


def apply_missing_strategy(
    df: pd.DataFrame,
    strategy: Mapping[str, Any],
    verbose: bool = True,
) -> pd.DataFrame:
    """Append ``{col}_is_null`` flag columns and drop high-missing columns.

    Both lists are pre-computed from train; the same lists are applied to
    val and test so the three splits stay feature-aligned.
    """
    flag_columns: list[str] = list(strategy.get("flag_columns", []))
    drop_columns: list[str] = list(strategy.get("drop_columns", []))

    flag_series: dict[str, pd.Series] = {}
    for col in flag_columns:
        if col in df.columns:
            flag_series[f"{col}_is_null"] = df[col].isna().astype(np.int8)

    drop_present = [c for c in drop_columns if c in df.columns]
    base = df.drop(columns=drop_present) if drop_present else df

    if flag_series:
        flags_df = pd.DataFrame(flag_series, index=df.index)
        out = pd.concat([base, flags_df], axis=1).copy()
    else:
        out = base.copy()

    if verbose:
        logger.info(
            "apply_missing_strategy",
            input_cols=df.shape[1],
            output_cols=out.shape[1],
            flags_added=len(flag_series),
            cols_dropped=len(drop_present),
        )
    return out


# ===========================================================================
# Step 7 — low-variance column drop (fit on TRAIN, drop from all)
# ===========================================================================


def compute_low_variance_drop_list(
    train: pd.DataFrame,
    frequency_threshold: float = 0.99,
    exclude_cols: Sequence[str] | None = None,
    export_path: str = "data/processed/drop_low_variance.json",
    verbose: bool = True,
) -> list[str]:
    """List columns where a single value covers ``frequency_threshold`` of train.

    The list is computed only from train and then handed to
    :func:`apply_drop_columns` for val and test. NaN values count as a category
    via ``dropna=False``.
    """
    excluded = set(exclude_cols) if exclude_cols is not None else set(DEFAULT_EXCLUDES)
    to_drop: list[str] = []
    for col in train.columns:
        if col in excluded:
            continue
        top_ratio = train[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_ratio >= frequency_threshold:
            to_drop.append(col)

    _write_json(to_drop, export_path)
    if verbose:
        logger.info(
            "compute_low_variance_drop_list",
            threshold=frequency_threshold,
            n_dropped=len(to_drop),
            export_path=str(export_path),
        )
    return to_drop


# ===========================================================================
# Step 8 — V-feature pruning (fit on TRAIN, drop from all)
# ===========================================================================


def compute_v_feature_drop_list(
    train: pd.DataFrame,
    keep_n: int = 2,
    export_path: str = "data/processed/drop_v_features.json",
    verbose: bool = True,
) -> list[str]:
    """List V* columns to drop based on TRAIN NaN-pattern groups + variance.

    IEEE-CIS V* columns come in blocks that share an identical NaN pattern.
    Within each block, keep the ``keep_n`` columns with the highest variance
    (in train); drop the rest. NaN-grouping and variance ranking are both
    train-only.
    """
    v_cols = [c for c in train.columns if c.startswith("V") and not c.endswith("_is_null")]
    if not v_cols:
        if verbose:
            logger.warning("compute_v_feature_drop_list: no V* columns found")
        return []

    nan_groups: dict[int, list[str]] = {}
    for col in v_cols:
        nan_count = int(train[col].isna().sum())
        nan_groups.setdefault(nan_count, []).append(col)

    v_variances = train[v_cols].var()

    dropped: list[str] = []
    for cols in nan_groups.values():
        if len(cols) <= keep_n:
            continue
        ranked = v_variances[cols].sort_values(ascending=False)
        top = ranked.head(keep_n).index.tolist()
        dropped.extend(c for c in cols if c not in top)

    _write_json(
        {"groups_found": len(nan_groups), "dropped_v_cols": dropped},
        export_path,
    )
    if verbose:
        logger.info(
            "compute_v_feature_drop_list",
            initial_v_cols=len(v_cols),
            groups=len(nan_groups),
            dropped=len(dropped),
            export_path=str(export_path),
        )
    return dropped


# ===========================================================================
# Step 9 — groupby aggregations with OOF for high-cardinality keys
# ===========================================================================


DEFAULT_GROUP_KEYS: tuple[str, ...] = ("card1", "card2", "card3", "card5", "uid", "addr1")
DEFAULT_AGGS: dict[str, tuple[str, ...]] = {
    "TransactionAmt": ("mean", "std", "max", "min", "median"),
    "D1": ("mean", "std", "nunique"),
    "C1": ("mean", "max"),
    "dist1": ("mean", "std"),
}
DEFAULT_NUNIQUE_PAIRS: tuple[tuple[str, str], ...] = (
    ("card1", "addr1"),
    ("card1", "P_emaildomain"),
)
DEFAULT_ALPHA: float = 50.0
DEFAULT_N_BLOCKS: int = 5


def _smooth(count: pd.Series, local_mean: pd.Series, global_mean: float, alpha: float) -> pd.Series:
    """Bayesian shrinkage of per-group means toward a global prior.

    ``smoothed = (count * local_mean + alpha * global_mean) / (count + alpha)``

    Vectorized on groupby outputs. Groups with ``count >> alpha`` keep their
    local mean; groups with ``count << alpha`` are pulled toward the prior.
    A universal formula — no hard-threshold branches on small groups.
    """
    return (count * local_mean + alpha * global_mean) / (count + alpha)


def _build_rules_from_pool(
    pool: pd.DataFrame,
    group_keys: Sequence[str],
    agg_spec: Mapping[str, Sequence[str]],
    nunique_pairs: Sequence[tuple[str, str]],
    global_means: Mapping[str, float],
    alpha: float,
) -> dict[str, dict[str, dict[str, dict[Any, Any]]]]:
    """Build per-key aggregation rules on the given data pool.

    ``mean`` metrics are smoothed via :func:`_smooth` with ``global_means`` as
    the prior. ``max / min / std / median / nunique`` are stored raw — the
    Bayesian formula is only well-defined for means.

    An empty pool still produces a rule skeleton with empty dicts, so the
    apply step can fall back to ``global_mean`` on mean columns (matches the
    ``count=0`` limit of the smoothing formula).
    """
    rules: dict[str, dict[str, dict[str, dict[Any, Any]]]] = {}
    for key in group_keys:
        if key not in pool.columns:
            continue
        rules.setdefault(key, {})

        for target, metrics in agg_spec.items():
            if target not in pool.columns or target == key:
                continue
            rules[key][target] = {}
            if len(pool) == 0:
                for metric in metrics:
                    rules[key][target][metric] = {}
                continue
            needed = list(dict.fromkeys(("count", "mean", *metrics)))
            grouped = pool.groupby(key)[target].agg(needed)
            for metric in metrics:
                if metric == "mean":
                    smoothed = _smooth(
                        grouped["count"],
                        grouped["mean"],
                        global_means.get(target, 0.0),
                        alpha,
                    )
                    rules[key][target]["mean"] = smoothed.to_dict()
                else:
                    rules[key][target][metric] = grouped[metric].to_dict()

        for k_pair, t_pair in nunique_pairs:
            if k_pair != key or t_pair not in pool.columns:
                continue
            feature = f"{t_pair}_nunique"
            nunique_dict = pool.groupby(key)[t_pair].nunique().to_dict() if len(pool) > 0 else {}
            rules[key][feature] = {feature: nunique_dict}

    return rules


def _map_rules_to_series(
    rules: Mapping[str, Any],
    df: pd.DataFrame,
    global_means: Mapping[str, float],
) -> dict[str, pd.Series]:
    """Map ``rules`` onto ``df`` rows and return the aggregate columns as Series.

    On ``mean`` columns, unseen keys fall back to the target's ``global_mean``
    (Bayesian formula's ``count=0`` limit). Other metrics leave unseen as NaN.
    """
    new_cols: dict[str, pd.Series] = {}
    for key, target_dict in rules.items():
        if key not in df.columns:
            continue
        key_series = df[key]
        for target, metric_dict in target_dict.items():
            for metric, mapping in metric_dict.items():
                col_name = f"{key}_{target}" if target.endswith("_nunique") else f"{key}_{target}_{metric}"
                mapped = key_series.map(mapping)
                if metric == "mean" and target in global_means:
                    mapped = mapped.fillna(global_means[target])
                new_cols[col_name] = mapped.astype("float32")
    return new_cols


def _added_agg_column_names(
    train: pd.DataFrame,
    group_keys: Sequence[str],
    agg_spec: Mapping[str, Sequence[str]],
    nunique_pairs: Sequence[tuple[str, str]],
) -> list[str]:
    """Deterministic list of aggregate column names this pipeline will add."""
    names: list[str] = []
    for key in group_keys:
        if key not in train.columns:
            continue
        for target, metrics in agg_spec.items():
            if target not in train.columns or target == key:
                continue
            names.extend(f"{key}_{target}_{m}" for m in metrics)
        for k_pair, t_pair in nunique_pairs:
            if k_pair != key or t_pair not in train.columns:
                continue
            names.append(f"{key}_{t_pair}_nunique")
    return names


def _apply_aggregations_to_train(
    train: pd.DataFrame,
    full_rules: Mapping[str, Any],
    group_keys: Sequence[str],
    agg_spec: Mapping[str, Sequence[str]],
    nunique_pairs: Sequence[tuple[str, str]],
    oof_keys: Sequence[str],
    global_means: Mapping[str, float],
    alpha: float,
    n_blocks: int,
    time_col: str,
) -> pd.DataFrame:
    """Materialize aggregate features on train.

    Keys in ``oof_keys`` get expanding-window time-block OOF: train is sorted
    by ``time_col`` and cut into ``n_blocks`` contiguous blocks; for block
    ``i`` the rules are built only from blocks ``0..i-1`` (strictly past).
    Block 0 has no history — mean columns fall back to the target's
    ``global_mean`` (same behaviour val sees on its unseen keys), other
    metrics stay NaN.

    Keys not in ``oof_keys`` are mapped directly from ``full_rules`` — the
    Bayesian smoothing already dampens any low-count self-leak.
    """
    n = len(train)
    oof_set = {k for k in oof_keys if k in train.columns}
    non_oof_set = {k for k in group_keys if k in train.columns and k not in oof_set}

    new_cols: dict[str, np.ndarray] = {}

    if non_oof_set:
        non_oof_rules = {k: v for k, v in full_rules.items() if k in non_oof_set}
        for col_name, series in _map_rules_to_series(non_oof_rules, train, global_means).items():
            new_cols[col_name] = series.to_numpy(dtype="float32", copy=True)

    if oof_set:
        if time_col not in train.columns:
            raise KeyError(f"time-block OOF requires column {time_col!r} in train")

        oof_group_keys = [k for k in group_keys if k in oof_set]
        oof_nunique_pairs = [p for p in nunique_pairs if p[0] in oof_set]

        oof_buffers: dict[str, np.ndarray] = {col_name: np.full(n, np.nan, dtype="float32") for col_name in _added_agg_column_names(train, oof_group_keys, agg_spec, oof_nunique_pairs)}

        sort_pos = np.argsort(train[time_col].to_numpy(), kind="stable")
        sorted_train = train.iloc[sort_pos]

        bounds = [int(n * i / n_blocks) for i in range(n_blocks + 1)]
        for i in range(n_blocks):
            start, end = bounds[i], bounds[i + 1]
            if start == end:
                continue
            pool = sorted_train.iloc[:start]
            fold_df = sorted_train.iloc[start:end]

            fold_rules = _build_rules_from_pool(
                pool=pool,
                group_keys=oof_group_keys,
                agg_spec=agg_spec,
                nunique_pairs=oof_nunique_pairs,
                global_means=global_means,
                alpha=alpha,
            )
            fold_cols = _map_rules_to_series(fold_rules, fold_df, global_means)
            for col_name, series in fold_cols.items():
                oof_buffers[col_name][start:end] = series.to_numpy(dtype="float32", copy=True)
            logger.debug(
                "aggregation_oof.block_done",
                block=i,
                n_blocks=n_blocks,
                pool_rows=int(start),
                fold_rows=int(end - start),
            )

        inv_sort = np.argsort(sort_pos, kind="stable")
        for col_name, arr in oof_buffers.items():
            new_cols[col_name] = arr[inv_sort]

    if not new_cols:
        return train.copy()

    new_df = pd.DataFrame(new_cols, index=train.index)
    return pd.concat([train, new_df], axis=1).copy()


def compute_aggregation_mappings(
    train: pd.DataFrame,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    aggs: Mapping[str, Sequence[str]] | None = None,
    nunique_pairs: Sequence[tuple[str, str]] = DEFAULT_NUNIQUE_PAIRS,
    oof_keys: Sequence[str] | None = None,
    alpha: float = DEFAULT_ALPHA,
    n_blocks: int = DEFAULT_N_BLOCKS,
    time_col: str = "TransactionDT",
    export_path: str = "data/processed/aggregation_mappings.json",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit aggregation mappings on train with Bayesian smoothing + time-block OOF.

    Every ``mean`` aggregate is shrunk toward the target's global prior via
    :func:`_smooth` (prior weight ``alpha``). Other metrics are stored raw.
    For each key in ``oof_keys`` (default: all ``group_keys``), train-side
    features come from an expanding-window time-block OOF so that no train
    aggregate peeks at temporally-future train rows.

    Args:
        train: Train-only DataFrame.
        group_keys: Columns used as groupby keys.
        aggs: ``{target_col: (metric, ...)}``. Defaults to :data:`DEFAULT_AGGS`.
        nunique_pairs: Extra ``(group_key, target)`` pairs for nunique counts.
        oof_keys: Subset of ``group_keys`` that receive expanding-window
            time-block OOF on train. ``None`` (default) means all group keys.
        alpha: Prior weight for Bayesian smoothing. Default 50.
        n_blocks: Number of chronological blocks used for train OOF.
        time_col: Column used to order train before block-cutting.
        export_path: Where to write the JSON mapping snapshot.
        verbose: Emit a structlog summary when True.

    Returns:
        ``(train_with_features, mappings)`` — ``train_with_features`` is the
        train DataFrame extended with the new aggregate columns. ``mappings``
        is the payload fed into :func:`apply_aggregation_mappings` for val/test.
    """
    agg_spec: Mapping[str, Sequence[str]] = DEFAULT_AGGS if aggs is None else aggs
    oof_keys_effective: tuple[str, ...] = tuple(group_keys) if oof_keys is None else tuple(oof_keys)

    global_means: dict[str, float] = {}
    for target in agg_spec:
        if target in train.columns:
            global_means[target] = float(train[target].mean())

    full_rules = _build_rules_from_pool(
        pool=train,
        group_keys=group_keys,
        agg_spec=agg_spec,
        nunique_pairs=nunique_pairs,
        global_means=global_means,
        alpha=alpha,
    )
    added_columns = _added_agg_column_names(train, group_keys, agg_spec, nunique_pairs)

    payload: dict[str, Any] = {
        "rules": full_rules,
        "columns": added_columns,
        "oof_keys": list(oof_keys_effective),
        "n_blocks": n_blocks,
        "alpha": alpha,
        "global_means": global_means,
        "time_col": time_col,
    }
    _write_json(payload, export_path)

    train_with_features = _apply_aggregations_to_train(
        train=train,
        full_rules=full_rules,
        group_keys=group_keys,
        agg_spec=agg_spec,
        nunique_pairs=nunique_pairs,
        oof_keys=oof_keys_effective,
        global_means=global_means,
        alpha=alpha,
        n_blocks=n_blocks,
        time_col=time_col,
    )

    if verbose:
        logger.info(
            "compute_aggregation_mappings",
            n_rules=sum(len(v) for v in full_rules.values()),
            added_columns=len(added_columns),
            oof_keys=list(oof_keys_effective),
            n_blocks=n_blocks,
            alpha=alpha,
            train_shape_after=list(train_with_features.shape),
            export_path=str(export_path),
        )

    return train_with_features, payload


def apply_aggregation_mappings(
    df: pd.DataFrame,
    mappings: Mapping[str, Any],
    verbose: bool = True,
) -> pd.DataFrame:
    """Map the fitted (smoothed) aggregations onto ``df`` (val or test).

    On ``mean`` columns, unseen keys fall back to ``global_mean`` of the
    target — which is exactly the ``count=0`` limit of :func:`_smooth`. Other
    metrics leave unseen values as NaN (XGBoost handles natively).
    """
    rules = mappings.get("rules", {})
    global_means: Mapping[str, float] = mappings.get("global_means", {})

    new_cols = _map_rules_to_series(rules, df, global_means)
    out = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1).copy() if new_cols else df.copy()

    if verbose:
        logger.info(
            "apply_aggregation_mappings",
            n_rows=len(out),
            n_new_cols=len(new_cols),
            output_cols=out.shape[1],
        )
    return out


# ===========================================================================
# Step 10 — relative / domain features (fit entity stats on TRAIN)
# ===========================================================================


DEFAULT_ENTITY_COLS: tuple[str, ...] = ("card1", "card2", "addr1", "ProductCD", "P_emaildomain")


def compute_domain_mappings(
    train: pd.DataFrame,
    entity_cols: Sequence[str] = DEFAULT_ENTITY_COLS,
    alpha: float = DEFAULT_ALPHA,
    export_path: str = "data/processed/domain_mappings.json",
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit per-entity ``TransactionAmt`` mean/median lookups on train.

    Means are shrunk toward ``global_mean`` via :func:`_smooth` (prior weight
    ``alpha``) so low-count entities do not inject razor-sharp local averages
    into the downstream ratio features. Medians are stored raw.

    Behavioural Z-scores in :func:`apply_domain_mappings` read UID-aggregated
    columns produced by step 9, so nothing is recomputed there.
    """
    entities: dict[str, dict[str, dict[Any, float]]] = {}
    if "TransactionAmt" not in train.columns:
        payload: dict[str, Any] = {
            "entities": entities,
            "global_mean": None,
            "global_median": None,
            "alpha": alpha,
        }
        _write_json(payload, export_path)
        if verbose:
            logger.warning("compute_domain_mappings: TransactionAmt missing; empty payload")
        return payload

    global_mean = float(train["TransactionAmt"].mean())
    global_median = float(train["TransactionAmt"].median())

    for col in entity_cols:
        if col not in train.columns:
            continue
        grouped = train.groupby(col)["TransactionAmt"].agg(["count", "mean", "median"])
        smoothed_mean = _smooth(grouped["count"], grouped["mean"], global_mean, alpha)
        entities[col] = {
            "mean": smoothed_mean.to_dict(),
            "median": grouped["median"].to_dict(),
        }

    payload = {
        "entities": entities,
        "global_mean": global_mean,
        "global_median": global_median,
        "alpha": alpha,
    }
    _write_json(payload, export_path)

    if verbose:
        logger.info(
            "compute_domain_mappings",
            entities=list(entities.keys()),
            alpha=alpha,
            export_path=str(export_path),
        )
    return payload


def apply_domain_mappings(
    df: pd.DataFrame,
    mappings: Mapping[str, Any],
    verbose: bool = True,
) -> pd.DataFrame:
    """Append relative / domain features using pre-fitted entity mappings.

    Entity ratios use the **smoothed** entity mean (with fallback to
    ``global_mean`` for unseen entities) and the raw entity median (with
    fallback to ``global_median``). UID-based Z-scores read columns produced
    by :func:`compute_aggregation_mappings` — call that first. Row-local
    features (``log1p``, decimal split, ``email_match``) are computed here.
    """
    out = df.copy()
    eps = 1e-6
    new_cols: dict[str, pd.Series] = {}

    uid_mean = out.get("uid_TransactionAmt_mean")
    uid_std = out.get("uid_TransactionAmt_std")
    uid_median = out.get("uid_TransactionAmt_median")
    amt = out.get("TransactionAmt")

    if amt is not None and uid_mean is not None and uid_std is not None and uid_median is not None:
        new_cols["amt_zscore_uid"] = (amt - uid_mean) / (uid_std + eps)
        new_cols["amt_to_mean_uid"] = amt / (uid_mean + eps)
        new_cols["amt_to_median_uid"] = amt / (uid_median + eps)
    elif verbose:
        logger.warning("apply_domain_mappings: UID aggregates missing; z-scores skipped")

    entities = mappings.get("entities", {})
    global_mean = mappings.get("global_mean")
    global_median = mappings.get("global_median")
    if amt is not None:
        for entity, stats in entities.items():
            if entity not in out.columns:
                continue
            mean_map = out[entity].map(stats["mean"])
            if global_mean is not None:
                mean_map = mean_map.fillna(global_mean)
            median_map = out[entity].map(stats["median"])
            if global_median is not None:
                median_map = median_map.fillna(global_median)
            new_cols[f"amt_to_mean_{entity}"] = amt / (mean_map + eps)
            new_cols[f"amt_to_median_{entity}"] = amt / (median_map + eps)

    card1_c1_mean = out.get("card1_C1_mean")
    c1 = out.get("C1")
    if c1 is not None and card1_c1_mean is not None:
        new_cols["c1_to_mean_card1"] = c1 / (card1_c1_mean + eps)

    if amt is not None:
        new_cols["TransactionAmt_log1p"] = np.log1p(amt)
        decimal_part = (amt - amt.fillna(0).astype(int)) * 1000
        new_cols["TransactionAmt_decimal"] = decimal_part.fillna(0).astype(np.int32)

    p_email = out.get("P_emaildomain")
    r_email = out.get("R_emaildomain")
    if p_email is not None and r_email is not None:
        match = (p_email.astype(str) == r_email.astype(str)).astype(np.int8)
        match[p_email.isna() | r_email.isna()] = -1
        new_cols["email_match"] = match

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1).copy()

    if verbose:
        logger.info(
            "apply_domain_mappings",
            n_rows=len(out),
            n_new_cols=len(new_cols),
            output_cols=out.shape[1],
        )
    return out


# ===========================================================================
# Step 11 — time features (row-local; call per split)
# ===========================================================================


DEFAULT_TIME_RULES: dict[str, Any] = {
    "hour_seconds": 3600,
    "day_seconds": 86400,
    "week_days": 7,
    "night_range": [0, 6],
    "weekend_start_day": 5,
}


def apply_time_features(
    df: pd.DataFrame,
    time_rules: Mapping[str, Any] | None = None,
    export_path: str = "data/processed/time_rules.json",
    verbose: bool = True,
) -> pd.DataFrame:
    """Derive cyclic and calendar features from ``TransactionDT``. Row-local."""
    if "TransactionDT" not in df.columns:
        if verbose:
            logger.warning("apply_time_features: TransactionDT missing; skipping")
        return df.copy()

    rules: dict[str, Any] = dict(DEFAULT_TIME_RULES if time_rules is None else time_rules)

    dt = df["TransactionDT"]
    hour = (dt // rules["hour_seconds"]) % 24
    dow = (dt // rules["day_seconds"]) % rules["week_days"]
    night_lo, night_hi = rules["night_range"]

    new_cols = {
        "dt_hour": hour,
        "dt_day_week": dow,
        "dt_hour_sin": np.sin(2 * np.pi * hour / 24),
        "dt_hour_cos": np.cos(2 * np.pi * hour / 24),
        "is_night": ((hour >= night_lo) & (hour <= night_hi)).astype(np.int8),
        "is_weekend": (dow >= rules["weekend_start_day"]).astype(np.int8),
    }

    out = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1).copy()
    _write_json(rules, export_path)

    if verbose:
        logger.info(
            "apply_time_features",
            n_rows=len(out),
            n_new_cols=len(new_cols),
            export_path=str(export_path),
        )
    return out


# ===========================================================================
# Step 12 — frequency + label encoding (fit on TRAIN, apply to all)
# ===========================================================================


def compute_encoding_mappings(
    train: pd.DataFrame,
    cat_cols: Sequence[str],
    missing_token: str = "Unknown",
    export_path: str = "data/processed/encoding_mappings.json",
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit frequency + label encodings on train only.

    NaN handling: all values are first stringified via :func:`_safe_str_fill`,
    so missing values become a real ``"Unknown"`` token (not the literal string
    ``"nan"``). Both frequency counts and label classes are computed on this
    cleaned representation, so train and val/test stay perfectly consistent.

    Returns:
        ``{"frequency": {col: {token: count}}, "label_classes": {col: [token,...]},
           "missing_token": "Unknown"}``.
    """
    frequency: dict[str, dict[str, int]] = {}
    label_classes: dict[str, list[str]] = {}

    for col in cat_cols:
        if col not in train.columns:
            continue
        cleaned = _safe_str_fill(train[col], missing_token=missing_token)
        frequency[col] = {str(k): int(v) for k, v in cleaned.value_counts().items()}
        label_classes[col] = sorted(cleaned.unique().tolist())

    payload = {
        "frequency": frequency,
        "label_classes": label_classes,
        "missing_token": missing_token,
    }
    _write_json(payload, export_path)

    if verbose:
        logger.info(
            "compute_encoding_mappings",
            n_cols=len(label_classes),
            missing_token=missing_token,
            export_path=str(export_path),
        )
    return payload


def apply_encoding_mappings(
    df: pd.DataFrame,
    mappings: Mapping[str, Any],
    unknown_value: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply fitted frequency + label encodings to ``df``.

    Unseen categories (present in val/test but not in train):
        - Frequency feature: ``0``.
        - Label feature: ``unknown_value`` (default ``-1``).

    NaN values map to the ``missing_token`` from the mapping (default
    ``"Unknown"``), so a NaN in val gets the same code as a NaN in train.
    """
    out = df.copy()
    freq_maps: Mapping[str, Mapping[str, int]] = mappings.get("frequency", {})
    class_maps: Mapping[str, list[str]] = mappings.get("label_classes", {})
    missing_token = mappings.get("missing_token", "Unknown")

    freq_new: dict[str, pd.Series] = {}
    for col, freq_map in freq_maps.items():
        if col not in out.columns:
            continue
        cleaned = _safe_str_fill(out[col], missing_token=missing_token)
        freq_new[f"{col}_freq"] = cleaned.map(freq_map).fillna(0).astype(np.int32)
    if freq_new:
        out = pd.concat([out, pd.DataFrame(freq_new, index=out.index)], axis=1).copy()

    for col, classes in class_maps.items():
        if col not in out.columns:
            continue
        index = {cls: i for i, cls in enumerate(classes)}
        cleaned = _safe_str_fill(out[col], missing_token=missing_token)
        values = cleaned.map(index)
        out[col] = values.fillna(unknown_value).astype(np.int32)

    if verbose:
        logger.info(
            "apply_encoding_mappings",
            n_rows=len(out),
            n_frequency_cols=len(freq_new),
            n_label_cols=len(class_maps),
            output_cols=out.shape[1],
        )
    return out


# ===========================================================================
# Step 13 — correlation drop (fit list on TRAIN, drop from all splits)
# ===========================================================================


def compute_correlation_drop_list(
    train: pd.DataFrame,
    threshold: float = 0.80,
    sample_ratio: float = 0.20,
    protected_cols: Sequence[str] = DEFAULT_EXCLUDES,
    export_path: str = "data/processed/drop_corr.json",
    random_state: int = 42,
    verbose: bool = True,
) -> list[str]:
    """Compute the list of highly-correlated columns to drop, **from train only**.

    A sampled correlation matrix keeps the computation tractable. The list is
    then handed to :func:`apply_drop_columns` for every split so train and
    val/test stay feature-aligned.
    """
    numeric_df = train.select_dtypes(include=[np.number])
    sample_n = max(1, int(len(numeric_df) * sample_ratio))
    sample = numeric_df.sample(n=sample_n, random_state=random_state)

    corr = sample.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    protected = set(protected_cols)
    to_drop = [col for col in upper.columns if col not in protected and (upper[col] > threshold).any()]

    _write_json(to_drop, export_path)
    if verbose:
        logger.info(
            "compute_correlation_drop_list",
            threshold=threshold,
            sample_rows=sample_n,
            n_dropped=len(to_drop),
            export_path=str(export_path),
        )
    return to_drop


# ===========================================================================
# Generic — drop a pre-computed column list (used by steps 7, 8, 13)
# ===========================================================================


def apply_drop_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """Drop a pre-computed column list (missing names are silently ignored)."""
    present = [c for c in columns if c in df.columns]
    out = df.drop(columns=present).copy()
    if verbose:
        logger.info(
            "apply_drop_columns",
            requested=len(columns),
            dropped=len(present),
            output_cols=out.shape[1],
        )
    return out


# ===========================================================================
# Optional diagnostic — adversarial validation (post feature-engineering)
# ===========================================================================


def apply_adversarial_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    auc_threshold: float = 0.80,
    ignore_cols: Sequence[str] = DEFAULT_EXCLUDES,
    export_path: str = "data/processed/drop_adversarial.json",
    random_state: int = 42,
    verbose: bool = True,
) -> list[str]:
    """Identify features whose distribution shifts between ``train`` and ``test``.

    For each candidate feature, train a shallow XGBoost classifier to predict
    the split label (0 = train, 1 = test). Features whose per-feature AUC
    exceeds ``auc_threshold`` are drift suspects and their names are returned.

    Diagnostic only — drop with :func:`apply_drop_columns` if you decide to act.
    """
    cols = [c for c in train.columns if c in test.columns and c not in set(ignore_cols) and c != "TransactionDT"]
    if not cols:
        return []

    combined = pd.concat([train[cols], test[cols]], axis=0).reset_index(drop=True)
    labels = np.concatenate([np.zeros(len(train)), np.ones(len(test))])

    drift: list[str] = []
    for col in cols:
        dmat = xgb.DMatrix(combined[[col]].values, label=labels)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "verbosity": 0,
            "tree_method": "hist",
            "seed": random_state,
        }
        cv_results = xgb.cv(params, dmat, num_boost_round=10, nfold=3, seed=random_state)
        auc = float(cv_results["test-auc-mean"].iloc[-1])  # type: ignore[union-attr]
        if auc > auc_threshold:
            drift.append(col)
            if verbose:
                logger.warning("adversarial.drift_detected", column=col, auc=round(auc, 4))

    _write_json(drift, export_path)
    if verbose:
        logger.info(
            "apply_adversarial_validation",
            n_dropped=len(drift),
            export_path=str(export_path),
        )
    return drift


def _calculate_single_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Computes the Population Stability Index for a single feature.
    """
    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return 999.0

    expected_percents, _ = np.histogram(expected, bins=buckets)
    actual_percents, _ = np.histogram(actual, bins=buckets)

    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)

    expected_percents = np.clip(expected_percents, 1e-6, 1)
    actual_percents = np.clip(actual_percents, 1e-6, 1)

    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return float(psi_value)


def apply_psi_validation(train_df: pd.DataFrame, val_df: pd.DataFrame, psi_threshold: float = 0.10, export_path: str | None = None) -> list[str]:
    """
    Identifies and returns columns with a PSI value greater than the threshold.
    Exports the list to a JSON file if export_path is provided.
    """
    logger.info("Starting PSI validation...", threshold=psi_threshold)

    drift_cols = []
    ignore_cols = ["isFraud", "TransactionID", "TransactionDT", "uid"]
    features = [c for c in train_df.columns if c not in ignore_cols]

    for col in features:
        psi_val = _calculate_single_psi(train_df[col], val_df[col])
        if psi_val > psi_threshold:
            logger.warning("psi.drift_detected", column=col, psi=round(psi_val, 4))
            drift_cols.append(col)

    logger.info("PSI validation finished.", dropped_count=len(drift_cols))

    if export_path:
        with open(export_path, "w") as f:
            json.dump({"psi_drift_columns": drift_cols}, f, indent=4)
        logger.info("PSI drops exported.", path=export_path)

    return drift_cols
