import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests

# ---------------------------------------------------------------------------
# Ghana administrative regions – all 16 post-2019 regions
# ISO codes match geoBoundaries GHA-ADM1 shapeISO property
# ---------------------------------------------------------------------------
GHANA_REGIONS = {
    "Greater Accra Region":  "GH-AA",
    "Ashanti Region":        "GH-AH",
    "Eastern Region":        "GH-EP",
    "Central Region":        "GH-CP",
    "Western Region":        "GH-WP",
    "Northern Region":       "GH-NP",
    "Volta Region":          "GH-TV",
    "Bono Region":           "GH-BO",
    "Upper East Region":     "GH-UE",
    "Upper West Region":     "GH-UW",
    "Western North Region":  "GH-WN",
    "Bono East Region":      "GH-BE",
    "Ahafo Region":          "GH-AF",
    "Oti Region":            "GH-OT",
    "North East Region":     "GH-NE",
    "Savannah Region":       "GH-SV",
}

# AMR antibiotic metadata – WHONET codes used as column names
AMR_ANTIBIOTICS = {
    "AMX": {"name": "Amoxicillin",        "class": "Penicillins"},
    "AMC": {"name": "Amox-Clavulanate",   "class": "Penicillins"},
    "CIP": {"name": "Ciprofloxacin",      "class": "Fluoroquinolones"},
    "CTX": {"name": "Cefotaxime",         "class": "Cephalosporins"},
    "CAZ": {"name": "Ceftazidime",        "class": "Cephalosporins"},
    "MEM": {"name": "Meropenem",          "class": "Carbapenems"},
    "GEN": {"name": "Gentamicin",         "class": "Aminoglycosides"},
    "SXT": {"name": "Trimethoprim-Sulfa", "class": "Sulfonamides"},
    "OXA": {"name": "Oxacillin",          "class": "Penicillins"},
    "VAN": {"name": "Vancomycin",         "class": "Glycopeptides"},
    "LZD": {"name": "Linezolid",          "class": "Oxazolidinones"},
    "COL": {"name": "Colistin",           "class": "Polymyxins"},
}


# ---------------------------------------------------------------------------
# Country-level helpers (unchanged – used by all non-AMR projects)
# ---------------------------------------------------------------------------

def merge_data_with_countries(df_map, add_capital_location=False):
    """Add country metadata to df_map via assets/countries.csv + GeoJSON capitals."""
    contries_path = "assets/countries.csv"
    countries = pd.read_csv(contries_path, encoding="latin-1")

    geojson_url = os.path.join(
        "https://raw.githubusercontent.com/",
        "martynafford/natural-earth-geojson/master/",
        "50m/cultural/ne_50m_populated_places_simple.json",
    )
    capitals = json.loads(requests.get(geojson_url).text)
    features = ["adm0_a3", "latitude", "longitude", "featurecla"]
    capitals = [{k: x["properties"][k] for k in features} for x in capitals["features"]]
    capitals = pd.DataFrame.from_dict(capitals)
    capitals = capitals.sort_values(by=["adm0_a3", "featurecla"])
    capitals = capitals.drop_duplicates(["adm0_a3"]).reset_index(drop=True)
    capitals.drop(columns=["featurecla"], inplace=True)
    capitals.rename(columns={"adm0_a3": "Code"}, inplace=True)

    countries = pd.merge(countries, capitals, how="left", on="Code")
    countries.rename(
        columns={
            "Code": "country_iso",
            "Country": "country_name",
            "Region": "country_region",
            "Income group": "country_income",
            "latitude": "country_capital_lat",
            "longitude": "country_capital_lon",
        },
        inplace=True,
    )
    df_map = pd.merge(df_map, countries, on="country_iso", how="left")
    return df_map


def get_countries(df_map):
    df_countries = df_map[["country_iso", "country_name", "subjid"]]
    df_countries = df_countries.groupby(["country_iso", "country_name"]).count().reset_index()
    df_countries.rename(columns={"subjid": "country_count"}, inplace=True)
    return df_countries


def get_public_countries(path):
    data_file = os.path.join(path, "dashboard_data.csv")
    df_countries = pd.read_csv(data_file)
    return df_countries


# ---------------------------------------------------------------------------
# Ghana regional map helpers
# ---------------------------------------------------------------------------

def get_ghana_region_data(df_map, df_micro, map_mode="volume", specimen_type="All", antibiotic="CIP"):
    """
    Aggregate data by Ghana administrative region for the sub-national map.

    Parameters
    ----------
    df_map        : patient-level DataFrame – must have 'ghana_region' and 'ghana_region_iso'
    df_micro      : microbiology DataFrame – must have 'ghana_region', 'ghana_region_iso',
                    'micro_specimen_type', and antibiotic R/I/S columns
    map_mode      : "volume"     → colour regions by isolate count
                    "resistance" → colour regions by % resistant for chosen antibiotic
    specimen_type : "All" or a specific value (Blood, Urine, Sputum, Wound)
    antibiotic    : WHONET code (e.g. "CIP", "MEM")

    Returns
    -------
    DataFrame: region_iso, region_name, value, hover_text
    """
    empty = pd.DataFrame(columns=["region_iso", "region_name", "value", "hover_text"])

    if df_micro is None or df_micro.empty:
        return empty
    if "ghana_region_iso" not in df_micro.columns:
        return empty

    df = df_micro.copy()

    # Filter by specimen type
    if specimen_type != "All" and "micro_specimen_type" in df.columns:
        df = df[df["micro_specimen_type"] == specimen_type]

    if df.empty:
        return empty

    spec_label = specimen_type if specimen_type != "All" else "All specimens"
    grp = df.groupby(["ghana_region_iso", "ghana_region"])

    if map_mode == "volume":
        agg = grp.size().reset_index(name="value")
        agg["hover_text"] = (
            "<b>" + agg["ghana_region"] + "</b>"
            + "<br>Specimen: " + spec_label
            + "<br>Isolates: <b>" + agg["value"].astype(str) + "</b>"
        )
    else:
        # Resistance rate mode
        if antibiotic not in df.columns:
            return empty

        def _rrate(sub):
            mask = sub[antibiotic].isin(["R", "I", "S"])
            n_tested = int(mask.sum())
            if n_tested == 0:
                return pd.Series({"value": np.nan, "n_tested": 0, "n_resistant": 0})
            n_resistant = int((sub.loc[mask, antibiotic] == "R").sum())
            pct = round(100.0 * n_resistant / n_tested, 1)
            return pd.Series({"value": pct, "n_tested": n_tested, "n_resistant": n_resistant})

        agg = grp.apply(_rrate).reset_index()
        abx_meta = AMR_ANTIBIOTICS.get(antibiotic, {})
        abx_name  = abx_meta.get("name", antibiotic)
        abx_class = abx_meta.get("class", "")

        agg["hover_text"] = (
            "<b>" + agg["ghana_region"] + "</b>"
            + "<br>Specimen: " + spec_label
            + "<br>" + abx_name + " (" + abx_class + ")"
            + "<br>Resistance: <b>" + agg["value"].astype(str) + "%</b>"
            + "<br>Tested: " + agg["n_tested"].astype(str)
            + "  |  Resistant: " + agg["n_resistant"].astype(str)
        )

    agg.rename(columns={"ghana_region_iso": "region_iso", "ghana_region": "region_name"}, inplace=True)
    return agg[["region_iso", "region_name", "value", "hover_text"]]


def create_ghana_region_map(df_regions, map_layout_dict=None, map_mode="volume", antibiotic="CIP"):
    """
    Choropleth map over Ghana's 16 administrative regions using local GeoJSON.

    Parameters
    ----------
    df_regions      : output of get_ghana_region_data()
    map_layout_dict : Plotly map layout dict (centre / zoom from project config)
    map_mode        : "volume" or "resistance"
    antibiotic      : WHONET code – controls colorbar title in resistance mode
    """
    geojson_path = "assets/ghana_regions.geojson"

    # Load local GeoJSON (saved during project setup)
    with open(geojson_path) as f:
        geojson = json.load(f)

    df_plot = df_regions.dropna(subset=["value"]) if not df_regions.empty else df_regions

    if df_plot.empty:
        fig = go.Figure()
        if map_layout_dict:
            fig.update_layout(map_layout_dict)
        return fig

    if map_mode == "resistance":
        colorscale = _resistance_colorscale()
        zmin, zmax = 0, 100
        abx_name = AMR_ANTIBIOTICS.get(antibiotic, {}).get("name", antibiotic)
        colorbar_title = f"% Resistant<br>({abx_name})"
    else:
        colorscale = _volume_colorscale(df_plot["value"])
        zmin = max(1, int(df_plot["value"].min()))
        zmax = int(df_plot["value"].max())
        colorbar_title = "Isolate<br>Count"

    fig = go.Figure(
        go.Choroplethmap(
            geojson=geojson,
            featureidkey="properties.shapeISO",
            locations=df_plot["region_iso"],
            z=df_plot["value"],
            text=df_plot["hover_text"],
            hovertemplate="%{text}<extra></extra>",
            colorscale=colorscale,
            showscale=True,
            zmin=zmin,
            zmax=zmax,
            marker_line_color="white",
            marker_opacity=0.75,
            marker_line_width=1.2,
            colorbar={
                "bgcolor": "rgba(255,255,255,0.95)",
                "thickness": 18,
                "ticklen": 4,
                "x": 1,
                "xref": "paper",
                "xanchor": "right",
                "xpad": 5,
                "title": {"text": colorbar_title, "side": "right", "font": {"size": 11}},
            },
        )
    )

    # Default to Ghana-centred view if no layout provided
    default_layout = {
        "map": {
            "style": "carto-positron",
            "center": {"lat": 7.9465, "lon": -1.0232},
            "zoom": 5.8,
        },
        "margin": {"r": 0, "t": 0, "l": 0, "b": 0},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
    }
    if map_layout_dict:
        fig.update_layout(map_layout_dict)
    else:
        fig.update_layout(default_layout)

    return fig


# ---------------------------------------------------------------------------
# Colour scales
# ---------------------------------------------------------------------------

def _resistance_colorscale():
    """Green → yellow → red for 0–100% resistance rates."""
    return [
        [0.00, "rgb(0,104,55)"],
        [0.25, "rgb(102,189,99)"],
        [0.50, "rgb(255,255,191)"],
        [0.75, "rgb(252,141,89)"],
        [1.00, "rgb(215,48,39)"],
    ]


def _volume_colorscale(series):
    """Blue → green → orange → red for isolate volumes."""
    cutoffs = np.percentile(series.dropna(), [10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100])
    mx = series.max()
    if mx == 0:
        return [[0, "rgb(0,0,255)"], [1, "rgb(255,53,0)"]]
    cutoffs = cutoffs / mx
    cutoffs = np.insert(np.repeat(cutoffs, 2)[:-1], 0, 0)
    n = len(cutoffs) // 2
    colors = _interpolate_colors(["0000FF", "00EA66", "A7FA00", "FFBE00", "FF7400", "FF3500"], n)
    colors = np.repeat(colors, 2)
    return [[float(x), y] for x, y in zip(cutoffs, colors)]


def _interpolate_colors(colors, n):
    rgbs = [tuple(int(c.lstrip("#")[i: i + 2], 16) for i in (0, 2, 4)) for c in colors]
    result = []
    transitions = len(colors) - 1
    steps = n // transitions
    steps_list = [steps + 1] * (n % transitions) + [steps] * (transitions - (n % transitions))
    for i in range(transitions):
        for step in range(steps_list[i]):
            rgb = [int(rgbs[i][j] + (step / steps_list[i]) * (rgbs[i + 1][j] - rgbs[i][j])) for j in range(3)]
            result.append(f"rgb({rgb[0]},{rgb[1]},{rgb[2]})")
    if len(result) < n:
        result.append(f"rgb({rgbs[-1][0]},{rgbs[-1][1]},{rgbs[-1][2]})")
    return result[:n]


# ---------------------------------------------------------------------------
# Legacy helpers kept for non-AMR projects
# ---------------------------------------------------------------------------

def interpolate_colors(colors, n):
    return _interpolate_colors(colors, n)


def get_map_colorscale(df_countries, map_percentile_cutoffs=None):
    if map_percentile_cutoffs is None:
        map_percentile_cutoffs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100]
    cutoffs = np.percentile(df_countries["country_count"], map_percentile_cutoffs)
    if df_countries["country_count"].count() < len(map_percentile_cutoffs):
        cutoffs = df_countries["country_count"].sort_values()
    cutoffs = cutoffs / df_countries["country_count"].max()
    num_colors = len(cutoffs)
    cutoffs = np.insert(np.repeat(cutoffs, 2)[:-1], 0, 0)
    colors = _interpolate_colors(["0000FF", "00EA66", "A7FA00", "FFBE00", "FF7400", "FF3500"], num_colors)
    colors = np.repeat(colors, 2)
    return [[x, y] for x, y in zip(cutoffs, colors)]


def create_map(df_countries, map_layout_dict=None):
    geojson = os.path.join(
        "https://raw.githubusercontent.com/",
        "martynafford/natural-earth-geojson/master/",
        "50m/cultural/ne_50m_admin_0_countries.json",
    )
    map_colorscale = get_map_colorscale(df_countries)
    fig = go.Figure(
        go.Choroplethmap(
            geojson=geojson,
            featureidkey="properties.ADM0_A3",
            locations=df_countries["country_iso"],
            z=df_countries["country_count"],
            text=df_countries["country_name"],
            colorscale=map_colorscale,
            showscale=True,
            zmin=1,
            zmax=df_countries["country_count"].max(),
            marker_line_color="black",
            marker_opacity=0.5,
            marker_line_width=0.3,
            colorbar={
                "bgcolor": "rgba(255,255,255,1)",
                "thickness": 20,
                "ticklen": 5,
                "x": 1,
                "xref": "paper",
                "xanchor": "right",
                "xpad": 5,
            },
        )
    )
    fig.update_layout(map_layout_dict)
    return fig


def filter_df_map(df_map, sex_value, age_value, country_value, admdate_value, admdate_marks, outcome_value):
    df_map["filters_age"] = df_map["filters_age"].astype(float)
    admdate_min = pd.to_datetime(admdate_marks[str(admdate_value[0])]["label"])
    admdate_max = pd.to_datetime(admdate_marks[str(admdate_value[1])]["label"]) + pd.DateOffset(months=1)
    df_map_filtered = df_map[
        (df_map["filters_sex"].isin(sex_value))
        & ((df_map["filters_age"] >= age_value[0]) | df_map["filters_age"].isna())
        & ((df_map["filters_age"] <= age_value[1]) | df_map["filters_age"].isna())
        & ((df_map["filters_admdate"] >= admdate_min) | df_map["filters_admdate"].isna())
        & ((df_map["filters_admdate"] <= admdate_max) | df_map["filters_admdate"].isna())
        & (df_map["filters_outcome"].isin(outcome_value))
        & (df_map["filters_country"].isin(country_value) | df_map["filters_country"].isna())
    ]
    return df_map_filtered.reset_index(drop=True)
