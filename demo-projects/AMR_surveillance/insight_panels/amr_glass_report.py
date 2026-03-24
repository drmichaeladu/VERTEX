import os

import numpy as np
import pandas as pd

import vertex.IsaricAMR as iamr
import vertex.IsaricDraw as idw


def define_button():
    return {"item": "AMR Surveillance", "label": "GLASS Summary"}


def create_visuals(
    df_map, df_forms_dict, dictionary, quality_report, filepath, suffix, save_inputs
):
    # Load microbiology data
    df_micro = df_forms_dict.get("microbiology", pd.DataFrame())
    if df_micro.empty:
        csv_path = os.path.join(filepath, "analysis_data", "microbiology.csv")
        if os.path.exists(csv_path):
            df_micro = pd.read_csv(csv_path)

    antibiotic_cols = [
        "AMX", "AMC", "CIP", "CTX", "CAZ", "MEM",
        "GEN", "SXT", "OXA", "VAN", "LZD", "COL",
    ]
    available_abx = [c for c in antibiotic_cols if c in df_micro.columns]

    # GLASS priority pathogens for blood cultures
    glass_pathogens = [
        "E. coli",
        "K. pneumoniae",
        "S. aureus",
        "A. baumannii",
        "S. pneumoniae",
        "P. aeruginosa",
    ]
    # Key GLASS antibiotics per organism
    glass_key_abx = {
        "E. coli": ["CTX", "MEM", "CIP"],
        "K. pneumoniae": ["CTX", "MEM", "CIP"],
        "S. aureus": ["OXA", "VAN"],
        "A. baumannii": ["MEM", "CIP"],
        "S. pneumoniae": ["AMX", "SXT"],
        "P. aeruginosa": ["MEM", "CAZ"],
    }

    # --- Figure 1: Descriptive table of GLASS indicators by origin ---
    if df_micro.empty or not available_abx:
        fig1 = idw.fig_placeholder(
            None,
            title="No GLASS data available",
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_label="GLASS Indicators",
            graph_about="No data available for GLASS summary.",
        )
    else:
        glass_data = iamr.glass_summary_multi_abx(
            df_micro,
            organism_col="micro_organism",
            specimen_col="micro_specimen_type",
            antibiotic_cols=available_abx,
            origin_col="micro_origin",
        )
        if glass_data.empty:
            fig1 = idw.fig_placeholder(
                None,
                title="No GLASS data available",
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_label="GLASS Indicators",
                graph_about="No data available for GLASS summary.",
            )
        else:
            # Summarise by organism, antibiotic, and origin
            glass_summary_table = (
                glass_data.groupby(["organism", "antibiotic", "origin"])
                .agg(
                    n_tested=("n_tested", "sum"),
                    n_resistant=("n_resistant", "sum"),
                )
                .reset_index()
            )
            glass_summary_table["pct_resistant"] = round(
                glass_summary_table["n_resistant"]
                / glass_summary_table["n_tested"]
                * 100,
                1,
            )
            # Pivot so origins become columns
            glass_pivot = glass_summary_table.pivot_table(
                index=["organism", "antibiotic"],
                columns="origin",
                values=["n_tested", "pct_resistant"],
                aggfunc="first",
            )
            # Flatten column names
            glass_pivot.columns = [
                f"{stat} ({origin})"
                for stat, origin in glass_pivot.columns
            ]
            glass_pivot = glass_pivot.reset_index()
            glass_pivot.columns = [
                c.replace("n_tested", "N Tested").replace(
                    "pct_resistant", "% Resistant"
                )
                for c in glass_pivot.columns
            ]
            glass_pivot.rename(
                columns={"organism": "Organism", "antibiotic": "Antibiotic"},
                inplace=True,
            )

            fig1 = idw.fig_table(
                glass_pivot,
                table_key="GLASS Key Indicators by Infection Origin (Community vs Hospital)",
                height=600,
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_id="fig_table_glass",
                graph_label="GLASS Summary Table*",
                graph_about="GLASS-aligned summary table showing resistance rates for key pathogen-antibiotic combinations, stratified by community vs hospital origin.",
            )

    # --- Figure 2: Blood culture resistance rates (GLASS priority pathogens) ---
    if df_micro.empty or not available_abx:
        fig2 = idw.fig_placeholder(
            None,
            title="No blood culture data available",
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_label="Blood Culture Resistance",
            graph_about="No blood culture data available.",
        )
    else:
        df_blood = df_micro[
            (df_micro["micro_specimen_type"] == "Blood")
            & (df_micro["micro_organism"].isin(glass_pathogens))
        ].copy()
        if df_blood.empty:
            fig2 = idw.fig_placeholder(
                None,
                title="No blood culture data available",
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_label="Blood Culture Resistance",
                graph_about="No blood culture data available.",
            )
        else:
            # Calculate resistance for key GLASS antibiotic per organism
            freq_records = []
            for organism in glass_pathogens:
                org_data = df_blood[df_blood["micro_organism"] == organism]
                if org_data.empty:
                    continue
                key_abxs = glass_key_abx.get(organism, [])
                for abx in key_abxs:
                    if abx not in org_data.columns:
                        continue
                    valid = org_data[abx].isin(["R", "I", "S"])
                    n_tested = valid.sum()
                    if n_tested == 0:
                        continue
                    n_resistant = (org_data.loc[valid, abx] == "R").sum()
                    pct = n_resistant / n_tested
                    freq_records.append(
                        {
                            "label": f"{organism} - {abx} (n={n_tested})",
                            "proportion": pct,
                            "short_label": f"{organism} - {abx}",
                        }
                    )

            if freq_records:
                df_freq = pd.DataFrame(freq_records)
                df_freq = df_freq.sort_values("proportion", ascending=True)
                fig2 = idw.fig_frequency_chart(
                    df_freq,
                    title="Blood Culture Resistance: GLASS Priority Pathogens*",
                    xlabel="Proportion Resistant",
                    ylabel="Organism - Antibiotic",
                    height=max(350, len(df_freq) * 35 + 100),
                    suffix=suffix,
                    filepath=filepath,
                    save_inputs=save_inputs,
                    graph_id="fig_freq_blood_glass",
                    graph_label="Blood Culture Resistance (GLASS)*",
                    graph_about="Horizontal bar chart showing resistance rates for GLASS priority pathogen-antibiotic combinations from blood cultures.",
                )
            else:
                fig2 = idw.fig_placeholder(
                    None,
                    title="No blood culture resistance data available",
                    suffix=suffix,
                    filepath=filepath,
                    save_inputs=save_inputs,
                    graph_label="Blood Culture Resistance",
                    graph_about="No blood culture resistance data available.",
                )

    # --- Figure 3: Population pyramid of BSI patients by age/sex ---
    if df_micro.empty:
        fig3 = idw.fig_placeholder(
            None,
            title="No BSI data available",
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_label="BSI Age-Sex Pyramid",
            graph_about="No bloodstream infection data available.",
        )
    else:
        df_bsi = df_micro[df_micro["micro_specimen_type"] == "Blood"].copy()
        if df_bsi.empty or "micro_age_group" not in df_bsi.columns:
            fig3 = idw.fig_placeholder(
                None,
                title="No BSI data available",
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_label="BSI Age-Sex Pyramid",
                graph_about="No bloodstream infection data available.",
            )
        else:
            sex_map = {1: "Male", "1": "Male", 2: "Female", "2": "Female"}
            df_bsi["sex_label"] = df_bsi["micro_sex"].map(sex_map)
            df_bsi = df_bsi.dropna(subset=["sex_label"])

            age_order = ["0-17", "18-49", "50-64", "65+"]
            # Count by age group, sex, and organism
            pyramid_data = (
                df_bsi.groupby(
                    ["micro_age_group", "sex_label", "micro_organism"]
                )
                .size()
                .reset_index(name="value")
            )
            pyramid_data.rename(
                columns={
                    "micro_age_group": "y_axis",
                    "sex_label": "side",
                    "micro_organism": "stack_group",
                },
                inplace=True,
            )
            # Assign left_side: Female=left (1), Male=right (0)
            pyramid_data["left_side"] = (
                pyramid_data["side"] == "Female"
            ).astype(int)
            # Order age groups
            pyramid_data["y_axis"] = pd.Categorical(
                pyramid_data["y_axis"], categories=age_order, ordered=True
            )
            pyramid_data = pyramid_data.sort_values("y_axis").reset_index(
                drop=True
            )

            # Color map for organisms
            organism_colors = {
                "E. coli": "#4A90D9",
                "K. pneumoniae": "#00C26F",
                "S. aureus": "#FF8C00",
                "A. baumannii": "#DF0069",
                "S. pneumoniae": "#9B59B6",
                "P. aeruginosa": "#1ABC9C",
            }
            # Filter to only organisms present
            color_map = {
                k: v
                for k, v in organism_colors.items()
                if k in pyramid_data["stack_group"].unique()
            }

            fig3 = idw.fig_dual_stack_pyramid(
                pyramid_data,
                title="Bloodstream Infections by Age Group and Sex*",
                xlabel="Number of Isolates",
                ylabel="Age Group",
                base_color_map=color_map,
                height=430,
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_id="fig_pyramid_bsi",
                graph_label="BSI Age-Sex Pyramid*",
                graph_about="Population pyramid showing bloodstream infection isolates by age group and sex, with organisms as stacked groups.",
            )

    # Disclaimer
    disclaimer_text = (
        "Disclaimer: the underlying data for these figures is synthetic data. "
        "Results may not be clinically relevant or accurate."
    )
    disclaimer_df = pd.DataFrame(
        disclaimer_text, columns=["paragraphs"], index=range(1)
    )
    disclaimer = idw.fig_text(
        disclaimer_df,
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label="*DISCLAIMER: SYNTHETIC DATA*",
        graph_about=disclaimer_text,
    )

    return (fig1, fig2, fig3, disclaimer)
