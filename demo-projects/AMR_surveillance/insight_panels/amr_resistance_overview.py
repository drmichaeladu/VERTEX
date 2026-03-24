import os

import pandas as pd

import vertex.IsaricAMR as iamr
import vertex.IsaricDraw as idw


def define_button():
    return {"item": "AMR Surveillance", "label": "Resistance Overview"}


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

    # --- Figure 1: Horizontal bar chart of overall resistance rates ---
    if df_micro.empty or not available_abx:
        fig1 = idw.fig_placeholder(
            None,
            title="No microbiology data available",
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_label="Resistance Rates",
            graph_about="No microbiology data available.",
        )
    else:
        rates = iamr.resistance_rates(df_micro, "micro_organism", available_abx)
        if rates.empty:
            fig1 = idw.fig_placeholder(
                None,
                title="No resistance data available",
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_label="Resistance Rates",
                graph_about="No resistance data available.",
            )
        else:
            # Overall resistance rate per organism (across all antibiotics)
            overall = (
                rates.groupby("organism")
                .agg(
                    n_tested=("n_tested", "sum"),
                    n_resistant=("n_resistant", "sum"),
                )
                .reset_index()
            )
            overall["pct_resistant"] = round(
                overall["n_resistant"] / overall["n_tested"] * 100, 1
            )
            overall = overall.sort_values("pct_resistant", ascending=True).tail(6)
            freq_data = pd.DataFrame(
                {
                    "label": overall["organism"]
                    + " (n="
                    + overall["n_tested"].astype(str)
                    + ")",
                    "proportion": overall["pct_resistant"] / 100,
                    "short_label": overall["organism"],
                }
            )
            fig1 = idw.fig_frequency_chart(
                freq_data,
                title="Overall Resistance Rate by Organism*",
                xlabel="Proportion Resistant",
                ylabel="Organism",
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_id="fig_freq_resistance_overview",
                graph_label="Resistance Rates by Organism*",
                graph_about="Horizontal bar chart showing overall resistance rates for top organisms across all tested antibiotics.",
            )

    # --- Figure 2: Heatmap antibiogram ---
    if df_micro.empty or not available_abx:
        fig2 = idw.fig_placeholder(
            None,
            title="No antibiogram data available",
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_label="Antibiogram",
            graph_about="No antibiogram data available.",
        )
    else:
        matrix = iamr.antibiogram_matrix(df_micro, "micro_organism", available_abx)
        if matrix.empty:
            fig2 = idw.fig_placeholder(
                None,
                title="No antibiogram data available",
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_label="Antibiogram",
                graph_about="No antibiogram data available.",
            )
        else:
            fig2 = idw.fig_heatmaps(
                matrix,
                title="Antibiogram: % Susceptibility*",
                ylabel="Organism",
                xlabel="Antibiotic",
                colorbar_label="% Susceptible",
                index_column="index",
                zmin=0,
                zmax=100,
                include_annotations=True,
                base_color_map="RdYlGn",
                height=450,
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_id="fig_heatmap_antibiogram",
                graph_label="Antibiogram Heatmap*",
                graph_about="Heatmap showing % susceptibility for each organism-antibiotic combination. Green = high susceptibility, Red = low susceptibility.",
            )

    # --- Figure 3: Pie chart of MDR classification ---
    if df_micro.empty or not available_abx:
        fig3 = idw.fig_placeholder(
            None,
            title="No MDR classification data available",
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_label="MDR Classification",
            graph_about="No MDR classification data available.",
        )
    else:
        df_mdr = iamr.mdr_classification(df_micro, "micro_organism", available_abx)
        mdr_counts = df_mdr["mdr_class"].value_counts().reset_index()
        mdr_counts.columns = ["names", "values"]
        # Filter out 'Unknown'
        mdr_counts = mdr_counts[mdr_counts["names"] != "Unknown"]
        if mdr_counts.empty:
            fig3 = idw.fig_placeholder(
                None,
                title="No MDR classification data available",
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_label="MDR Classification",
                graph_about="No MDR classification data available.",
            )
        else:
            color_map = {
                "Susceptible": "#00C26F",
                "Non-MDR": "#FFF500",
                "MDR": "#FF8C00",
                "XDR": "#DF0069",
                "PDR": "#8B0000",
            }
            fig3 = idw.fig_pie(
                mdr_counts,
                title="MDR Classification Distribution*",
                names="names",
                values="values",
                base_color_map=color_map,
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_id="fig_pie_mdr",
                graph_label="MDR Classification*",
                graph_about="Distribution of isolates by multidrug resistance classification (Susceptible, Non-MDR, MDR, XDR, PDR).",
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
