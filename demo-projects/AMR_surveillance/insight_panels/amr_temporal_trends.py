import os

import pandas as pd

import vertex.IsaricAMR as iamr
import vertex.IsaricDraw as idw


def define_button():
    return {"item": "AMR Surveillance", "label": "Temporal Trends"}


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
        "AMC", "AMP", "CAZ", "CIP", "CRO", "CTX",
        "GEN", "AMK", "LNZ", "MEM", "OXA", "SXT", "TZP", "VAN", "CLI", "ERY", "TET",
    ]
    available_abx = [c for c in antibiotic_cols if c in df_micro.columns]

    # --- Figure 1: Quarterly resistance trends (bar chart by quarter) ---
    # Use bar chart since fig_line_chart only supports a single line
    if df_micro.empty or not available_abx:
        fig1 = idw.fig_placeholder(
            None,
            title="No temporal trend data available",
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_label="Temporal Trends",
            graph_about="No temporal trend data available.",
        )
    else:
        # Select top 3 organism-antibiotic pairs by resistance rate
        rates = iamr.resistance_rates(df_micro, "micro_organism", available_abx)
        if rates.empty:
            fig1 = idw.fig_placeholder(
                None,
                title="No temporal trend data available",
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_label="Temporal Trends",
                graph_about="No temporal trend data available.",
            )
        else:
            top_pairs = (
                rates.sort_values("n_tested", ascending=False)
                .drop_duplicates(subset=["organism", "antibiotic"])
                .head(3)
            )

            # Build quarterly trends for each top pair
            all_trends = []
            for _, row in top_pairs.iterrows():
                org = row["organism"]
                abx = row["antibiotic"]
                df_subset = df_micro[df_micro["micro_organism"] == org].copy()
                trend = iamr.temporal_trend(
                    df_subset,
                    date_col="micro_sample_date",
                    organism_col="micro_organism",
                    antibiotic_col=abx,
                    resistance_col=abx,
                    freq="Q",
                )
                if not trend.empty:
                    trend["series"] = f"{org} - {abx}"
                    all_trends.append(trend)

            if all_trends:
                df_trends = pd.concat(all_trends, ignore_index=True)
                # Pivot for bar chart: index=period, columns=series, values=pct_resistant
                df_pivot = df_trends.pivot_table(
                    index="period",
                    columns="series",
                    values="pct_resistant",
                    aggfunc="first",
                ).fillna(0)
                df_pivot = df_pivot.reset_index().rename(columns={"period": "index"})
                df_pivot = df_pivot.sort_values("index").reset_index(drop=True)

                fig1 = idw.fig_bar_chart(
                    df_pivot,
                    title="Quarterly Resistance Trends (Top 3 Organism-Antibiotic Pairs)*",
                    xlabel="Quarter",
                    ylabel="% Resistant",
                    index_column="index",
                    barmode="group",
                    xaxis_tickformat="%Y-Q",
                    height=400,
                    suffix=suffix,
                    filepath=filepath,
                    save_inputs=save_inputs,
                    graph_id="fig_bar_temporal_trends",
                    graph_label="Quarterly Resistance Trends*",
                    graph_about="Grouped bar chart showing quarterly resistance trends for the top 3 organism-antibiotic combinations by volume.",
                )
            else:
                fig1 = idw.fig_placeholder(
                    None,
                    title="No temporal trend data available",
                    suffix=suffix,
                    filepath=filepath,
                    save_inputs=save_inputs,
                    graph_label="Temporal Trends",
                    graph_about="No temporal trend data available.",
                )

    # --- Figure 2: Bar chart of isolate counts by specimen type ---
    if df_micro.empty:
        fig2 = idw.fig_placeholder(
            None,
            title="No specimen data available",
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_label="Specimen Counts",
            graph_about="No specimen data available.",
        )
    else:
        specimen_counts = (
            df_micro.groupby(["micro_specimen_type", "micro_organism"])
            .size()
            .reset_index(name="count")
        )
        df_spec_pivot = specimen_counts.pivot_table(
            index="micro_specimen_type",
            columns="micro_organism",
            values="count",
            aggfunc="sum",
        ).fillna(0)
        df_spec_pivot = df_spec_pivot.reset_index().rename(
            columns={"micro_specimen_type": "index"}
        )

        fig2 = idw.fig_bar_chart(
            df_spec_pivot,
            title="Isolate Counts by Specimen Type*",
            xlabel="Specimen Type",
            ylabel="Number of Isolates",
            index_column="index",
            barmode="stack",
            height=380,
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_id="fig_bar_specimen_counts",
            graph_label="Isolates by Specimen Type*",
            graph_about="Stacked bar chart showing the number of isolates by specimen type, coloured by organism.",
        )

    # --- Figure 3: Stacked bar chart of MDR class by ward ---
    if df_micro.empty or not available_abx:
        fig3 = idw.fig_placeholder(
            None,
            title="No MDR data available",
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_label="MDR by Ward",
            graph_about="No MDR data available.",
        )
    else:
        df_mdr = iamr.mdr_classification(df_micro, "micro_organism", available_abx)
        df_mdr_filtered = df_mdr[df_mdr["mdr_class"] != "Unknown"]
        if df_mdr_filtered.empty:
            fig3 = idw.fig_placeholder(
                None,
                title="No MDR data available",
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_label="MDR by Ward",
                graph_about="No MDR data available.",
            )
        else:
            mdr_ward = (
                df_mdr_filtered.groupby(["micro_ward", "mdr_class"])
                .size()
                .reset_index(name="count")
            )
            df_mdr_pivot = mdr_ward.pivot_table(
                index="micro_ward",
                columns="mdr_class",
                values="count",
                aggfunc="sum",
            ).fillna(0)
            # Reorder columns
            col_order = ["Susceptible", "Non-MDR", "MDR", "XDR", "PDR"]
            col_order = [c for c in col_order if c in df_mdr_pivot.columns]
            df_mdr_pivot = df_mdr_pivot[col_order]
            df_mdr_pivot = df_mdr_pivot.reset_index().rename(
                columns={"micro_ward": "index"}
            )

            color_map = {
                "Susceptible": "#00C26F",
                "Non-MDR": "#FFF500",
                "MDR": "#FF8C00",
                "XDR": "#DF0069",
                "PDR": "#8B0000",
            }
            fig3 = idw.fig_bar_chart(
                df_mdr_pivot,
                title="MDR Classification by Ward*",
                xlabel="Ward",
                ylabel="Number of Isolates",
                index_column="index",
                barmode="stack",
                base_color_map=color_map,
                height=400,
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_id="fig_bar_mdr_ward",
                graph_label="MDR Classification by Ward*",
                graph_about="Stacked bar chart showing multidrug resistance classification distribution across hospital wards.",
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
