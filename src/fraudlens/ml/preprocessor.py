"""Preprocessing utilities for tabular fraud data."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import json
import structlog
from typing import Tuple, List, Dict

from fraudlens.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MissingAnalysis:
    """Result of a missing-value pass over a DataFrame.

    Attributes:
        dropped_columns: Column names dropped because their missing rate
            exceeded ``drop_threshold``.
        flag_columns: Names of ``{col}_is_null`` flag columns that were added
            to the frame.
        missing_rates: Per-column missing rate (``NaN`` ratio) computed before
            any transformation. Useful for later inspection / plotting.
    """

    dropped_columns: list[str] = field(default_factory=list)
    flag_columns: list[str] = field(default_factory=list)
    missing_rates: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))


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


def join_transaction_identity(
    transaction: pd.DataFrame,
    identity: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Left-join the identity table onto the transaction table.

    IEEE-CIS splits each record across two files: ``train_transaction`` holds
    one row per transaction and ``train_identity`` holds optional device/session
    attributes keyed by ``TransactionID``. Only ~25% of transactions have a
    matching identity row, so a left join is the right shape — legit rows
    without identity context keep their features and gain NaNs for the
    identity columns.

    Args:
        transaction: Transaction frame; must contain ``TransactionID`` and the
            target column ``isFraud``.
        identity: Identity frame; must contain ``TransactionID``.
        verbose: When True, log joined shape, fraud rate, and identity match
            rate via the project's structlog logger.

    Returns:
        A new DataFrame with every row from ``transaction`` plus the identity
        columns, joined on ``TransactionID``. Rows without an identity match
        carry NaN in the identity columns.

    Raises:
        KeyError: If ``TransactionID`` is missing from either input.
    """
    for name, frame in (("transaction", transaction), ("identity", identity)):
        if "TransactionID" not in frame.columns:
            raise KeyError(f"{name} frame is missing required column 'TransactionID'")

    joined = transaction.merge(identity, on="TransactionID", how="left")

    if verbose:
        matched = identity["TransactionID"].isin(transaction["TransactionID"]).sum()
        match_rate = matched / len(transaction) if len(transaction) > 0 else 0.0
        fraud_rate = (
            float(joined["isFraud"].mean()) if "isFraud" in joined.columns else float("nan")
        )
        logger.info(
            "join_transaction_identity",
            shape=list(joined.shape),
            n_rows=joined.shape[0],
            n_cols=joined.shape[1],
            fraud_rate=round(fraud_rate, 6),
            identity_match_rate=round(match_rate, 6),
            identity_matched=int(matched),
        )

    return joined


def handle_missing_values(
    df: pd.DataFrame,
    drop_threshold: float = 0.9,
    flag_threshold: float = 0.05,
    exclude_cols: Sequence[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, MissingAnalysis]:
    """Flag and prune columns based on missing-value rates.

    The transformation runs in two passes over every non-excluded column:

        1. **Flag**: if the missing rate is strictly greater than
           ``flag_threshold``, append an ``{col}_is_null`` column of ``int8``
           indicating which rows were NaN in the original. Flags are created
           *before* pruning so the missingness signal of a dropped column is
           preserved as a binary feature.
        2. **Drop**: if the missing rate is strictly greater than
           ``drop_threshold``, remove the original column. The corresponding
           ``{col}_is_null`` flag (if any) stays in the returned frame.

    The input DataFrame is copied, never mutated.

    Args:
        df: Input DataFrame. Expected to already be memory-optimized / joined.
        drop_threshold: Missing-rate cutoff above which a column is dropped
            (default ``0.9`` → columns with more than 90% NaN are removed).
        flag_threshold: Missing-rate cutoff above which a flag column is
            created (default ``0.05`` → columns with more than 5% NaN get an
            ``_is_null`` companion).
        exclude_cols: Column names to skip entirely (identifiers, target, etc.).
            Defaults to ``("TransactionID", "isFraud")``.
        verbose: When True, log counts and examples via the project logger.

    Returns:
        A tuple ``(df_out, report)`` where ``df_out`` is the transformed
        frame and ``report`` holds the dropped/flag column lists and the
        per-column missing-rate series (computed before any change).

    Raises:
        ValueError: If either threshold is outside ``[0.0, 1.0]``.
    """
    if not 0.0 <= drop_threshold <= 1.0:
        raise ValueError(f"drop_threshold must be in [0, 1], got {drop_threshold}")
    if not 0.0 <= flag_threshold <= 1.0:
        raise ValueError(f"flag_threshold must be in [0, 1], got {flag_threshold}")

    default_excludes = ("TransactionID", "isFraud")
    excluded = set(exclude_cols) if exclude_cols is not None else set(default_excludes)

    missing_rates = df.isna().mean().sort_values(ascending=False)

    dropped: list[str] = []
    flags: list[str] = []
    flag_series: dict[str, pd.Series] = {}

    candidate_cols = [c for c in df.columns if c not in excluded]

    for col in candidate_cols:
        rate = float(missing_rates.get(col, 0.0))
        if rate > flag_threshold:
            flag_name = f"{col}_is_null"
            flag_series[flag_name] = df[col].isna().astype(np.int8)
            flags.append(flag_name)
        if rate > drop_threshold:
            dropped.append(col)

    base = df.drop(columns=dropped) if dropped else df
    if flag_series:
        flags_df = pd.DataFrame(flag_series, index=df.index)
        out = pd.concat([base, flags_df], axis=1).copy()
    else:
        out = base.copy()

    report = MissingAnalysis(
        dropped_columns=dropped,
        flag_columns=flags,
        missing_rates=missing_rates,
    )

    if verbose:
        logger.info(
            "handle_missing_values",
            input_cols=df.shape[1],
            output_cols=out.shape[1],
            n_dropped=len(dropped),
            n_flags=len(flags),
            drop_threshold=drop_threshold,
            flag_threshold=flag_threshold,
            top_missing=[
                {"col": c, "rate": round(float(r), 4)} for c, r in missing_rates.head(5).items()
            ],
        )

    return out, report

def drop_low_variance_features(
    df: pd.DataFrame, 
    frequency_threshold: float = 0.99, 
    export_path: str = "data/processed/drop_variance.json"
) -> Tuple[pd.DataFrame, List[str]]:
    
    logger.info("Starting variance analysis...", initial_cols=df.shape[1])
    
    to_drop = []
    
    for col in df.columns:
        # Calculate the proportion of the single most frequent value (including NaNs)
        most_frequent_ratio = df[col].value_counts(normalize=True, dropna=False).values[0]
        
        if most_frequent_ratio >= frequency_threshold:
            to_drop.append(col)
            
    df_reduced = df.drop(columns=to_drop)
    
    with open(export_path, 'w') as f:
        json.dump(to_drop, f, indent=4)
        
    logger.info(
        "Variance analysis completed",
        threshold=frequency_threshold,
        dropped_count=len(to_drop),
        remaining_cols=df_reduced.shape[1]
    )
    
    return df_reduced, to_drop

def reduce_v_features(
    df: pd.DataFrame, 
    keep_n: int = 2,
    export_path: str = "data/processed/v_feature_reduction.json"
) -> Tuple[pd.DataFrame, List[str]]:
    
    logger.info("Starting V-feature reduction...", keep_per_group=keep_n)
    
    v_cols = [col for col in df.columns if col.startswith('V') and not col.endswith('_is_null')]
    
    if not v_cols:
        logger.warning("No V-features found in the DataFrame.")
        return df, []

    nan_groups: Dict[int, List[str]] = {}
    
    # Group V-features by their exact NaN counts
    for col in v_cols:
        nan_count = df[col].isnull().sum()
        if nan_count not in nan_groups:
            nan_groups[nan_count] = []
        nan_groups[nan_count].append(col)
        
    # Calculate variances once for efficiency
    v_variances = df[v_cols].var()
    
    kept_cols = []
    dropped_cols = []
    
    # Process each group
    for nan_count, cols in nan_groups.items():
        if len(cols) <= keep_n:
            kept_cols.extend(cols)
        else:
            # Sort columns in the group by variance (descending)
            group_variances = v_variances[cols].sort_values(ascending=False)
            top_cols = group_variances.head(keep_n).index.tolist()
            
            kept_cols.extend(top_cols)
            
            # The rest are dropped
            dropped = [c for c in cols if c not in top_cols]
            dropped_cols.extend(dropped)

    df_reduced = df.drop(columns=dropped_cols)
    
    # Export state for inference API
    state = {
        "groups_found": len(nan_groups),
        "kept_v_cols": kept_cols,
        "dropped_v_cols": dropped_cols
    }
    
    with open(export_path, 'w') as f:
        json.dump(state, f, indent=4)
        
    logger.info(
        "V-feature reduction completed",
        initial_v_cols=len(v_cols),
        groups_identified=len(nan_groups),
        kept_count=len(kept_cols),
        dropped_count=len(dropped_cols)
    )
    
    return df_reduced, dropped_cols

def normalize_d_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes time-delta D features (D1-D15) by converting them into 
    static reference points using the TransactionDT.
    
    Calculation:
    Transaction Day = TransactionDT // (24 * 60 * 60)
    Normalized_D = Transaction Day - D_feature
    """
    logger.info("Starting D-feature normalization...")
    
    if 'TransactionDT' not in df.columns:
        logger.warning("TransactionDT not found! Cannot normalize D features.")
        return df
        
    # TransactionDT is in seconds. Convert to Days 
    transaction_day = df['TransactionDT'] // 86400
    
    d_cols = [f'D{i}' for i in range(1, 16)]
    
    normalized_count = 0
    for col in d_cols:
        if col in df.columns:
            new_col_name = f'{col}_norm'
            # Calculate the absolute past day reference
            df[new_col_name] = transaction_day - df[col]
            normalized_count += 1
            
    logger.info(
        "D-feature normalization completed", 
        features_normalized=normalized_count
    )
    
    return df

def create_uid_feature(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting UID creation...")
    
    required_cols = ['card1', 'addr1', 'P_emaildomain', 'TransactionDT', 'D1']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing columns for UID creation: {missing_cols}")
        return df

    # Re-calculate normalized D1 for the UID string 
    # (in case D1_norm was dropped or altered)
    transaction_day = df['TransactionDT'] // 86400
    d1_norm = transaction_day - df['D1']
    
    c_card = df['card1'].fillna('NaN').astype(str)
    c_addr = df['addr1'].fillna('NaN').astype(str)
    c_email = df['P_emaildomain'].fillna('NaN').astype(str)
    c_d1 = d1_norm.fillna(-999).astype(str)
    
    # Vectorized string concatenation
    df['uid'] = c_card + "_" + c_addr + "_" + c_email + "_" + c_d1
    
    unique_uids = df['uid'].nunique()
    
    logger.info(
        "UID creation completed", 
        total_rows=df.shape[0],
        unique_uids=unique_uids
    )
    
    return df

def apply_scaled_aggregations(
    df: pd.DataFrame, 
    export_path: str = "data/processed/aggregation_rules.json"
) -> Tuple[pd.DataFrame, List[str]]:
    logger.info("Starting aggregation matrix")
    
    new_features_data = {} 
    all_rules = {}
    
    group_keys = ['card1', 'card2', 'card3', 'card5', 'uid', 'addr1']
    aggs = {
        'TransactionAmt': ['mean', 'std', 'max', 'min', 'median'],
        'D1': ['mean', 'std', 'nunique'],
        'C1': ['mean', 'max'],
        'dist1': ['mean', 'std']
    }
    
    for key in group_keys:
        if key not in df.columns:
            continue
            
        all_rules[key] = {}
        for target, metrics in aggs.items():
            if target not in df.columns or key == target:
                continue
                
            grouped_res = df.groupby(key)[target].agg(metrics)
            grouped_dict = grouped_res.to_dict()
            all_rules[key][target] = grouped_dict
            
            for metric in metrics:
                col_name = f'{key}_{target}_{metric}'
                # Store in dict instead of adding to df immediately
                new_features_data[col_name] = df[key].map(grouped_dict[metric])

    # 3. Specific unique counts
    for target_nunique in ['addr1', 'P_emaildomain']:
        col_name = f'card1_{target_nunique}_nunique'
        nunique_map = df.groupby('card1')[target_nunique].nunique().to_dict()
        new_features_data[col_name] = df['card1'].map(nunique_map)
        
        if 'card1' not in all_rules: all_rules['card1'] = {}
        all_rules['card1'][f'{target_nunique}_nunique'] = nunique_map

    # 4. Single-pass concatenation to avoid fragmentation
    if new_features_data:
        new_df = pd.DataFrame(new_features_data, index=df.index)
        df = pd.concat([df, new_df], axis=1)
        new_features = list(new_features_data.keys())
    else:
        new_features = []

    def sanitize_dict(d):
        if not isinstance(d, dict):
            return None if pd.isna(d) else d
        return {str(k): sanitize_dict(v) for k, v in d.items()}

    with open(export_path, 'w') as f:
        json.dump(sanitize_dict(all_rules), f)
        
    logger.info("Aggregation completed", added_features=len(new_features))
    
    return df, new_features

def apply_relative_and_domain_features(
    df: pd.DataFrame, 
    export_path: str = "data/processed/domain_rules.json"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Computes relative features
    """
    logger.info("Starting feature engineering (defragmented)...")
    
    new_features_dict = {} 
    rules = {}
    epsilon = 1e-6

    # 1. Behavioral Anomalies (Personal Norms)
    uid_mean_col = 'uid_TransactionAmt_mean'
    uid_std_col = 'uid_TransactionAmt_std'
    uid_median_col = 'uid_TransactionAmt_median'

    if all(col in df.columns for col in [uid_mean_col, uid_std_col, uid_median_col]):
        new_features_dict['amt_zscore_uid'] = (df['TransactionAmt'] - df[uid_mean_col]) / (df[uid_std_col] + epsilon)
        new_features_dict['amt_to_mean_uid'] = df['TransactionAmt'] / (df[uid_mean_col] + epsilon)
        new_features_dict['amt_to_median_uid'] = df['TransactionAmt'] / (df[uid_median_col] + epsilon)
    else:
        logger.warning("Step 9 UID aggregations not found! Skipping behavioral Z-scores.")

    # 2. Entity-Based Ratios (Segment Norms)
    target_entities = ['card1', 'card2', 'addr1', 'ProductCD', 'P_emaildomain']
    
    for entity in target_entities:
        if entity in df.columns:
            group_stats = df.groupby(entity)['TransactionAmt'].agg(['mean', 'median']).to_dict()
            rules[f'{entity}_stats'] = group_stats
            
            new_features_dict[f'amt_to_mean_{entity}'] = df['TransactionAmt'] / (df[entity].map(group_stats['mean']) + epsilon)
            new_features_dict[f'amt_to_median_{entity}'] = df['TransactionAmt'] / (df[entity].map(group_stats['median']) + epsilon)

    # 3. Frequency & Velocity Ratios
    card1_c1_mean_col = 'card1_C1_mean'
    if 'C1' in df.columns and 'card1' in df.columns:
        if card1_c1_mean_col in df.columns:
            new_features_dict['c1_to_mean_card1'] = df['C1'] / (df[card1_c1_mean_col] + epsilon)
        else:
            c1_mean = df.groupby('card1')['C1'].transform('mean')
            new_features_dict['c1_to_mean_card1'] = df['C1'] / (c1_mean + epsilon)

    # 4. Domain Transformation & Identification
    new_features_dict['TransactionAmt_log1p'] = np.log1p(df['TransactionAmt'])
    new_features_dict['TransactionAmt_decimal'] = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype(int)
    
    # Email matching logic (Optimized for Series)
    email_match = (df['P_emaildomain'].astype(str) == df['R_emaildomain'].astype(str)).astype(int)
    mask_missing = df['P_emaildomain'].isna() | df['R_emaildomain'].isna()
    email_match[mask_missing] = -1
    new_features_dict['email_match'] = email_match

    if new_features_dict:
        new_df = pd.DataFrame(new_features_dict, index=df.index)
        df = pd.concat([df, new_df], axis=1)
        new_features = list(new_features_dict.keys())
    else:
        new_features = []

    serialized_rules = json.loads(pd.Series(rules).to_json())
    with open(export_path, 'w') as f:
        json.dump(serialized_rules, f, indent=4)
        
    logger.info(
        "Industry-standard features completed",
        added_count=len(new_features),
        final_cols=df.shape[1]
    )
    
    return df, new_features

def apply_time_features(
    df: pd.DataFrame, 
    export_path: str = "data/processed/time_rules.json"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extracts temporal features and exports constants to JSON for inference consistency.
    Returns (DataFrame, new_features_list).
    """
    logger.info("Starting time-based feature engineering")
    
    new_features_dict = {}
    
    # 1. Define Time Rules (Constants that might change in the future)
    time_rules = {
        'hour_seconds': 3600,
        'day_seconds': 86400,
        'week_days': 7,
        'night_range': [0, 6],    # 00:00 to 06:00
        'weekend_start_day': 5    # Friday or Saturday depending on business logic
    }
    
    # 2. Extract Basic Units
    new_features_dict['dt_hour'] = (df['TransactionDT'] // time_rules['hour_seconds']) % 24
    new_features_dict['dt_day_week'] = (df['TransactionDT'] // time_rules['day_seconds']) % time_rules['week_days']
    
    # 3. Cyclical Encoding (Sin-Cos)
    # Formula: $$x_{sin} = \sin\left(\frac{2\pi \cdot x}{24}\right)$$
    new_features_dict['dt_hour_sin'] = np.sin(2 * np.pi * new_features_dict['dt_hour'] / 24)
    new_features_dict['dt_hour_cos'] = np.cos(2 * np.pi * new_features_dict['dt_hour'] / 24)
    
    new_features_dict['is_night'] = (
        (new_features_dict['dt_hour'] >= time_rules['night_range'][0]) & 
        (new_features_dict['dt_hour'] <= time_rules['night_range'][1])
    ).astype(int)
    
    new_features_dict['is_weekend'] = (
        new_features_dict['dt_day_week'] >= time_rules['weekend_start_day']
    ).astype(int)

    new_df = pd.DataFrame(new_features_dict, index=df.index)
    df = pd.concat([df, new_df], axis=1)
    
    new_features = list(new_features_dict.keys())

    # 6. Export Rules to JSON
    with open(export_path, 'w') as f:
        json.dump(time_rules, f, indent=4)
    
    logger.info(
        "Time features and rules exported",
        added_count=len(new_features),
        export_path=export_path
    )
    
    return df, new_features
