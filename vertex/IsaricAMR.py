"""
IsaricAMR.py — Antimicrobial Resistance Analytics Library

Provides analytical functions for AMR surveillance including resistance rate
calculations, antibiogram generation, MDR classification, prescribing metrics,
temporal trends, outbreak detection, and GLASS-aligned summaries.
"""

import numpy as np
import pandas as pd


def resistance_rates(
    df_micro: pd.DataFrame,
    organism_col: str,
    antibiotic_cols: list,
    groupby: str = None,
) -> pd.DataFrame:
    """Calculates % resistance for each organism x antibiotic combination.

    Resistant = 'R', Intermediate = 'I', Susceptible = 'S'.

    Returns tidy DataFrame: organism, antibiotic, n_tested, n_resistant,
    pct_resistant. Optionally grouped by an additional column.
    """
    if df_micro.empty:
        return pd.DataFrame(
            columns=["organism", "antibiotic", "n_tested", "n_resistant", "pct_resistant"]
        )

    records = []
    group_cols = [organism_col]
    if groupby is not None:
        group_cols = [groupby, organism_col]

    for abx in antibiotic_cols:
        df_valid = df_micro.loc[df_micro[abx].isin(["R", "I", "S"])].copy()
        if df_valid.empty:
            continue
        grouped = df_valid.groupby(group_cols)
        for group_key, grp in grouped:
            n_tested = len(grp)
            n_resistant = (grp[abx] == "R").sum()
            pct_resistant = (n_resistant / n_tested * 100) if n_tested > 0 else 0.0
            row = {
                "antibiotic": abx,
                "n_tested": n_tested,
                "n_resistant": int(n_resistant),
                "pct_resistant": round(pct_resistant, 1),
            }
            if groupby is not None:
                row[groupby] = group_key[0]
                row["organism"] = group_key[1]
            else:
                row["organism"] = group_key if isinstance(group_key, str) else group_key[0]
            records.append(row)

    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(
            columns=["organism", "antibiotic", "n_tested", "n_resistant", "pct_resistant"]
        )
    col_order = ["organism", "antibiotic", "n_tested", "n_resistant", "pct_resistant"]
    if groupby is not None:
        col_order = [groupby] + col_order
    return result[col_order].reset_index(drop=True)


def antibiogram_matrix(
    df_micro: pd.DataFrame,
    organism_col: str,
    antibiotic_cols: list,
) -> pd.DataFrame:
    """Builds antibiogram matrix: rows=organisms, cols=antibiotics,
    values=% susceptibility (for heatmap).

    Returns wide DataFrame with organism as index column.
    """
    if df_micro.empty:
        return pd.DataFrame()

    records = []
    for organism, grp in df_micro.groupby(organism_col):
        row = {"index": organism}
        for abx in antibiotic_cols:
            valid = grp[abx].isin(["R", "I", "S"])
            n_valid = valid.sum()
            if n_valid > 0:
                n_susceptible = (grp.loc[valid, abx] == "S").sum()
                row[abx] = round(n_susceptible / n_valid * 100, 1)
            else:
                row[abx] = np.nan
        records.append(row)

    result = pd.DataFrame(records)
    return result


def mdr_classification(
    df_micro: pd.DataFrame,
    organism_col: str,
    antibiotic_cols: list,
) -> pd.DataFrame:
    """Classifies each isolate as Susceptible / Non-MDR / MDR / XDR / PDR.

    Uses simplified WHO definitions:
    - Susceptible: resistant to 0 antibiotic classes
    - Non-MDR: resistant to 1-2 antibiotic classes
    - MDR: resistant to >=3 antibiotic classes
    - XDR: resistant to all but <=2 antibiotic classes
    - PDR: resistant to all tested antibiotic classes

    Returns df with added 'mdr_class' column.
    """
    if df_micro.empty:
        return df_micro.copy()

    # Define antibiotic class groupings
    abx_classes = {
        "Penicillins": ["AMX"],
        "BL-BLI": ["AMC"],
        "Fluoroquinolones": ["CIP"],
        "3GC": ["CTX", "CAZ"],
        "Carbapenems": ["MEM"],
        "Aminoglycosides": ["GEN"],
        "Folate pathway": ["SXT"],
        "Anti-staph penicillins": ["OXA"],
        "Glycopeptides": ["VAN"],
        "Oxazolidinones": ["LZD"],
        "Polymyxins": ["COL"],
    }

    df = df_micro.copy()
    available_cols = [c for c in antibiotic_cols if c in df.columns]

    def classify_row(row):
        n_classes_tested = 0
        n_classes_resistant = 0
        for cls_name, cls_abx in abx_classes.items():
            cls_cols = [c for c in cls_abx if c in available_cols]
            if not cls_cols:
                continue
            tested = [c for c in cls_cols if row.get(c) in ["R", "I", "S"]]
            if not tested:
                continue
            n_classes_tested += 1
            if any(row.get(c) == "R" for c in tested):
                n_classes_resistant += 1

        if n_classes_tested == 0:
            return "Unknown"
        if n_classes_resistant == 0:
            return "Susceptible"
        if n_classes_resistant >= n_classes_tested and n_classes_tested > 0:
            return "PDR"
        if n_classes_resistant >= n_classes_tested - 2 and n_classes_tested >= 5:
            return "XDR"
        if n_classes_resistant >= 3:
            return "MDR"
        return "Non-MDR"

    df["mdr_class"] = df.apply(classify_row, axis=1)
    return df


def prescribing_ddd(
    df_prescribing: pd.DataFrame,
    antibiotic_col: str,
    ddd_col: str,
    admissions_col: str,
) -> pd.DataFrame:
    """Calculates DDD/100 bed-days and AWaRe category distribution.

    Returns summary DataFrame with antibiotic, total_ddd, ddd_per_100_admissions,
    aware_category.
    """
    if df_prescribing.empty:
        return pd.DataFrame(
            columns=["antibiotic", "total_ddd", "ddd_per_100_admissions", "aware_category"]
        )

    n_admissions = df_prescribing[admissions_col].nunique() if admissions_col in df_prescribing.columns else len(df_prescribing)
    if n_admissions == 0:
        n_admissions = 1

    grouped = df_prescribing.groupby(antibiotic_col).agg(
        total_ddd=(ddd_col, "sum"),
        count=("subjid", "count") if "subjid" in df_prescribing.columns else (ddd_col, "count"),
    ).reset_index()

    grouped.rename(columns={antibiotic_col: "antibiotic"}, inplace=True)
    grouped["ddd_per_100_admissions"] = round(grouped["total_ddd"] / n_admissions * 100, 1)

    # Add AWaRe category if available
    if "presc_aware_category" in df_prescribing.columns:
        aware_map = (
            df_prescribing.groupby(antibiotic_col)["presc_aware_category"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown")
            .reset_index()
        )
        aware_map.rename(
            columns={antibiotic_col: "antibiotic", "presc_aware_category": "aware_category"},
            inplace=True,
        )
        grouped = grouped.merge(aware_map, on="antibiotic", how="left")
    else:
        grouped["aware_category"] = "Unknown"

    return grouped[["antibiotic", "total_ddd", "ddd_per_100_admissions", "aware_category"]]


def temporal_trend(
    df_micro: pd.DataFrame,
    date_col: str,
    organism_col: str,
    antibiotic_col: str,
    resistance_col: str,
    freq: str = "Q",
) -> pd.DataFrame:
    """Calculates resistance rate over time at specified frequency.

    Returns DataFrame: period, organism, antibiotic, pct_resistant, n_tested.
    """
    if df_micro.empty:
        return pd.DataFrame(
            columns=["period", "organism", "antibiotic", "pct_resistant", "n_tested"]
        )

    df = df_micro.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.loc[df[resistance_col].isin(["R", "I", "S"])]

    if df.empty:
        return pd.DataFrame(
            columns=["period", "organism", "antibiotic", "pct_resistant", "n_tested"]
        )

    df["period"] = df[date_col].dt.to_period(freq)

    grouped = df.groupby(["period", organism_col]).agg(
        n_tested=(resistance_col, "count"),
        n_resistant=(resistance_col, lambda x: (x == "R").sum()),
    ).reset_index()

    grouped["pct_resistant"] = round(grouped["n_resistant"] / grouped["n_tested"] * 100, 1)
    grouped["antibiotic"] = antibiotic_col
    grouped.rename(columns={organism_col: "organism"}, inplace=True)
    grouped["period"] = grouped["period"].astype(str)

    return grouped[["period", "organism", "antibiotic", "pct_resistant", "n_tested"]]


def outbreak_cluster_detection(
    df_micro: pd.DataFrame,
    organism_col: str,
    date_col: str,
    ward_col: str,
    window_days: int = 14,
    threshold: int = 3,
) -> pd.DataFrame:
    """Detects potential outbreak clusters using rolling window counts.

    Returns alert DataFrame: organism, ward, start_date, end_date, count,
    alert_level.
    """
    if df_micro.empty:
        return pd.DataFrame(
            columns=["organism", "ward", "start_date", "end_date", "count", "alert_level"]
        )

    df = df_micro.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    alerts = []
    for (organism, ward), grp in df.groupby([organism_col, ward_col]):
        grp = grp.sort_values(date_col)
        dates = grp[date_col].values

        for i in range(len(dates)):
            window_end = dates[i]
            window_start = window_end - np.timedelta64(window_days, "D")
            in_window = (dates >= window_start) & (dates <= window_end)
            count = in_window.sum()

            if count >= threshold:
                alert_level = "Warning" if count < threshold * 2 else "Critical"
                alerts.append(
                    {
                        "organism": organism,
                        "ward": ward,
                        "start_date": pd.Timestamp(window_start),
                        "end_date": pd.Timestamp(window_end),
                        "count": int(count),
                        "alert_level": alert_level,
                    }
                )

    if not alerts:
        return pd.DataFrame(
            columns=["organism", "ward", "start_date", "end_date", "count", "alert_level"]
        )

    result = pd.DataFrame(alerts)
    # Deduplicate overlapping windows — keep the highest count per organism/ward
    result = (
        result.sort_values("count", ascending=False)
        .drop_duplicates(subset=["organism", "ward", "start_date"], keep="first")
        .sort_values(["organism", "ward", "start_date"])
        .reset_index(drop=True)
    )
    return result


def glass_summary(
    df_micro: pd.DataFrame,
    organism_col: str,
    specimen_col: str,
    antibiotic_col: str,
    result_col: str,
    age_col: str,
    sex_col: str,
    origin_col: str,
) -> pd.DataFrame:
    """Generates GLASS-aligned summary: pathogen x specimen x antibiotic
    resistance rates stratified by age group, sex, and infection origin
    (community/hospital).

    Returns tidy summary table.
    """
    if df_micro.empty:
        return pd.DataFrame(
            columns=[
                "organism",
                "specimen",
                "antibiotic",
                "origin",
                "n_tested",
                "n_resistant",
                "pct_resistant",
            ]
        )

    records = []
    group_cols = [organism_col, specimen_col, origin_col]
    df_valid = df_micro.loc[df_micro[result_col].isin(["R", "I", "S"])].copy()

    if df_valid.empty:
        return pd.DataFrame(
            columns=[
                "organism",
                "specimen",
                "antibiotic",
                "origin",
                "n_tested",
                "n_resistant",
                "pct_resistant",
            ]
        )

    for keys, grp in df_valid.groupby(group_cols):
        organism, specimen, origin = keys
        n_tested = len(grp)
        n_resistant = (grp[result_col] == "R").sum()
        pct_resistant = round(n_resistant / n_tested * 100, 1) if n_tested > 0 else 0.0

        records.append(
            {
                "organism": organism,
                "specimen": specimen,
                "antibiotic": antibiotic_col,
                "origin": origin,
                "n_tested": n_tested,
                "n_resistant": int(n_resistant),
                "pct_resistant": pct_resistant,
            }
        )

    return pd.DataFrame(records)


def glass_summary_multi_abx(
    df_micro: pd.DataFrame,
    organism_col: str,
    specimen_col: str,
    antibiotic_cols: list,
    age_col: str = None,
    sex_col: str = None,
    origin_col: str = None,
) -> pd.DataFrame:
    """Generates GLASS-aligned summary across multiple antibiotics.

    Returns tidy summary table with columns: organism, specimen, antibiotic,
    origin, n_tested, n_resistant, pct_resistant.
    """
    if df_micro.empty:
        return pd.DataFrame(
            columns=[
                "organism",
                "specimen",
                "antibiotic",
                "origin",
                "n_tested",
                "n_resistant",
                "pct_resistant",
            ]
        )

    frames = []
    for abx in antibiotic_cols:
        if abx not in df_micro.columns:
            continue

        group_cols = [organism_col, specimen_col]
        if origin_col and origin_col in df_micro.columns:
            group_cols.append(origin_col)

        df_valid = df_micro.loc[df_micro[abx].isin(["R", "I", "S"])].copy()
        if df_valid.empty:
            continue

        for keys, grp in df_valid.groupby(group_cols):
            if origin_col and origin_col in df_micro.columns:
                organism, specimen, origin = keys
            else:
                organism, specimen = keys if isinstance(keys, tuple) else (keys, "Unknown")
                origin = "All"

            n_tested = len(grp)
            n_resistant = (grp[abx] == "R").sum()
            pct_resistant = round(n_resistant / n_tested * 100, 1) if n_tested > 0 else 0.0

            frames.append(
                {
                    "organism": organism,
                    "specimen": specimen,
                    "antibiotic": abx,
                    "origin": origin,
                    "n_tested": n_tested,
                    "n_resistant": int(n_resistant),
                    "pct_resistant": pct_resistant,
                }
            )

    if not frames:
        return pd.DataFrame(
            columns=[
                "organism",
                "specimen",
                "antibiotic",
                "origin",
                "n_tested",
                "n_resistant",
                "pct_resistant",
            ]
        )

    return pd.DataFrame(frames).reset_index(drop=True)
