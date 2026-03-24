import os

import pandas as pd

import vertex.IsaricAMR as iamr
import vertex.IsaricDraw as idw


def define_button():
    return {"item": "AMR Surveillance", "label": "Prescribing & Stewardship"}


def create_visuals(
    df_map, df_forms_dict, dictionary, quality_report, filepath, suffix, save_inputs
):
    # Load prescribing data
    df_presc = df_forms_dict.get("antibiotic_prescribing", pd.DataFrame())
    if df_presc.empty:
        csv_path = os.path.join(
            filepath, "analysis_data", "antibiotic_prescribing.csv"
        )
        if os.path.exists(csv_path):
            df_presc = pd.read_csv(csv_path)

    # --- Figure 1: Pie chart of AWaRe category distribution ---
    if df_presc.empty or "presc_aware_category" not in df_presc.columns:
        fig1 = idw.fig_placeholder(
            None,
            title="No prescribing data available",
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_label="AWaRe Distribution",
            graph_about="No prescribing data available.",
        )
    else:
        aware_counts = (
            df_presc["presc_aware_category"].value_counts().reset_index()
        )
        aware_counts.columns = ["names", "values"]
        color_map = {
            "Access": "#00C26F",
            "Watch": "#FFF500",
            "Reserve": "#DF0069",
        }
        fig1 = idw.fig_pie(
            aware_counts,
            title="WHO AWaRe Category Distribution*",
            names="names",
            values="values",
            base_color_map=color_map,
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_id="fig_pie_aware",
            graph_label="AWaRe Distribution*",
            graph_about="Distribution of antibiotic prescriptions by WHO AWaRe (Access, Watch, Reserve) classification.",
        )

    # --- Figure 2: Stacked bar chart by indication ---
    if df_presc.empty or "presc_indication" not in df_presc.columns:
        fig2 = idw.fig_placeholder(
            None,
            title="No prescribing data available",
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_label="Prescribing by Indication",
            graph_about="No prescribing data available.",
        )
    else:
        # Group by antibiotic and indication
        abx_indication = (
            df_presc.groupby(["presc_antibiotic", "presc_indication"])
            .size()
            .reset_index(name="count")
        )
        # Get top 10 antibiotics by total prescriptions
        top_abx = (
            df_presc["presc_antibiotic"]
            .value_counts()
            .head(10)
            .index.tolist()
        )
        abx_indication = abx_indication[
            abx_indication["presc_antibiotic"].isin(top_abx)
        ]
        df_ind_pivot = abx_indication.pivot_table(
            index="presc_antibiotic",
            columns="presc_indication",
            values="count",
            aggfunc="sum",
        ).fillna(0)
        df_ind_pivot = df_ind_pivot.reset_index().rename(
            columns={"presc_antibiotic": "index"}
        )

        color_map = {
            "Empiric": "#4A90D9",
            "Targeted": "#00C26F",
            "Prophylaxis": "#FFF500",
        }
        fig2 = idw.fig_bar_chart(
            df_ind_pivot,
            title="Antibiotic Prescribing by Indication (Top 10)*",
            xlabel="Antibiotic",
            ylabel="Number of Prescriptions",
            index_column="index",
            barmode="stack",
            base_color_map=color_map,
            height=420,
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_id="fig_bar_presc_indication",
            graph_label="Prescribing by Indication*",
            graph_about="Stacked bar chart showing antibiotic prescriptions by indication (Empiric, Targeted, Prophylaxis) for the top 10 most prescribed antibiotics.",
        )

    # --- Figure 3: Table of DDD/100 admissions ---
    if df_presc.empty:
        fig3 = idw.fig_placeholder(
            None,
            title="No prescribing data available",
            suffix=suffix,
            filepath=filepath,
            save_inputs=save_inputs,
            graph_label="DDD Summary",
            graph_about="No prescribing data available.",
        )
    else:
        ddd_summary = iamr.prescribing_ddd(
            df_presc,
            antibiotic_col="presc_antibiotic",
            ddd_col="presc_ddd",
            admissions_col="subjid",
        )
        if ddd_summary.empty:
            fig3 = idw.fig_placeholder(
                None,
                title="No DDD data available",
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_label="DDD Summary",
                graph_about="No DDD data available.",
            )
        else:
            # Format for display
            ddd_display = ddd_summary.copy()
            ddd_display.columns = [
                "Antibiotic",
                "Total DDD",
                "DDD/100 Admissions",
                "AWaRe Category",
            ]
            ddd_display["Total DDD"] = ddd_display["Total DDD"].round(1)
            ddd_display = ddd_display.sort_values(
                "DDD/100 Admissions", ascending=False
            ).reset_index(drop=True)

            fig3 = idw.fig_table(
                ddd_display,
                table_key="Defined Daily Doses (DDD) per 100 Admissions by Antibiotic",
                height=500,
                suffix=suffix,
                filepath=filepath,
                save_inputs=save_inputs,
                graph_id="fig_table_ddd",
                graph_label="DDD/100 Admissions Table*",
                graph_about="Table showing Defined Daily Doses (DDD) per 100 admissions for each antibiotic, with WHO AWaRe classification.",
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
