import dash_bootstrap_components as dbc
from dash import dcc, html

from vertex.layout.footer import footer
from vertex.layout.menu import define_menu
from vertex.layout.modals import login_modal, register_modal
from vertex.map import AMR_ANTIBIOTICS


def define_shell_layout(init_project_path, initial_body=None):
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="selected-project-path", data=init_project_path),
            dcc.Store(id="login-state", storage_type="session", data=False),
            # Header
            html.Div(
                [
                    html.H1("VERTEX - Visual Evidence & Research Tool for EXploration", id="title"),
                    html.P("Visual Evidence, Vital Answers"),
                    dbc.Button("Login", id="open-login", color="primary", size="sm", style={"display": "none"}),
                    dbc.Button("Logout", id="logout-button", style={"display": "none"}),
                    html.Div(id="auth-button-container"),
                ],
                style={"position": "absolute", "top": 0, "left": 10, "zIndex": 1000},
            ),
            # Main content area
            html.Div(id="project-body", children=initial_body),
            footer,
            login_modal,
            register_modal,
            dbc.Modal(id="modal", children=[dbc.ModalBody("")], is_open=False, size="xl"),
        ]
    )


def _amr_map_controls(specimen_options=None, organism_options=None):
    """
    Floating control panel overlaid on the map for AMR projects.
    Rows: map-level toggle | map-mode toggle | organism | specimen | antibiotic.
    Always rendered; visibility is controlled by the callback.
    """
    specimen_opts = specimen_options or [{"label": "All", "value": "All"}]
    organism_opts = organism_options or [{"label": "All organisms", "value": "All"}]
    abx_opts = [{"label": f"{k} – {v['name']}", "value": k} for k, v in AMR_ANTIBIOTICS.items()]

    _label_style = {"color": "#444", "fontWeight": "600", "fontSize": "0.78rem",
                    "marginRight": "6px", "whiteSpace": "nowrap"}
    _row_style   = {"display": "flex", "alignItems": "center", "marginBottom": "7px"}

    return html.Div(
        id="amr-map-controls",
        children=[
            # ── Row 0: title / badge ──────────────────────────────────
            html.Div(
                [
                    html.Span("AMR Map Controls",
                              style={"fontWeight": "700", "fontSize": "0.82rem",
                                     "color": "#1a1a2e", "letterSpacing": "0.03em"}),
                    html.Span("BETA",
                              style={"fontSize": "0.65rem", "background": "#e74c3c",
                                     "color": "white", "borderRadius": "3px",
                                     "padding": "1px 5px", "marginLeft": "8px",
                                     "verticalAlign": "middle"}),
                ],
                style={"marginBottom": "9px", "borderBottom": "1px solid #eee", "paddingBottom": "6px"},
            ),

            # ── Row 1: geographic level ───────────────────────────────
            html.Div(
                [
                    html.Small("Level:", style=_label_style),
                    dbc.RadioItems(
                        id="amr-map-level",
                        options=[
                            {"label": "Country", "value": "country"},
                            {"label": "Ghana Regions", "value": "region"},
                        ],
                        value="region",
                        inline=True,
                        inputStyle={"marginRight": "3px", "marginLeft": "8px"},
                        labelStyle={"fontSize": "0.79rem", "color": "#222"},
                    ),
                ],
                style=_row_style,
            ),

            # ── Row 2: metric mode ────────────────────────────────────
            html.Div(
                [
                    html.Small("Show:", style=_label_style),
                    dbc.RadioItems(
                        id="amr-map-mode",
                        options=[
                            {"label": "Isolate Volume", "value": "volume"},
                            {"label": "Resistance Rate", "value": "resistance"},
                        ],
                        value="volume",
                        inline=True,
                        inputStyle={"marginRight": "3px", "marginLeft": "8px"},
                        labelStyle={"fontSize": "0.79rem", "color": "#222"},
                    ),
                ],
                style=_row_style,
            ),

            # ── Row 3: organism filter ────────────────────────────────
            html.Div(
                [
                    html.Small("Organism:", style=_label_style),
                    dcc.Dropdown(
                        id="amr-organism-filter",
                        options=organism_opts,
                        value="All",
                        clearable=False,
                        style={"minWidth": "185px", "fontSize": "0.79rem"},
                    ),
                ],
                style=_row_style,
            ),

            # ── Row 4: specimen type ──────────────────────────────────
            html.Div(
                [
                    html.Small("Specimen:", style=_label_style),
                    dcc.Dropdown(
                        id="amr-specimen-filter",
                        options=specimen_opts,
                        value="All",
                        clearable=False,
                        style={"minWidth": "140px", "fontSize": "0.79rem"},
                    ),
                ],
                style=_row_style,
            ),

            # ── Row 5: antibiotic (only active when mode = resistance) ─
            html.Div(
                id="amr-antibiotic-row",
                children=[
                    html.Small("Antibiotic:", style=_label_style),
                    dcc.Dropdown(
                        id="amr-antibiotic-filter",
                        options=abx_opts,
                        value="CIP",
                        clearable=False,
                        style={"minWidth": "185px", "fontSize": "0.79rem"},
                    ),
                ],
                style={**_row_style, "marginBottom": "0"},
            ),
        ],
        style={
            "position": "absolute",
            "top": "70px",
            "right": "60px",
            "zIndex": 1100,
            "backgroundColor": "rgba(255,255,255,0.95)",
            "border": "1px solid #dce1e7",
            "borderRadius": "9px",
            "padding": "11px 15px",
            "boxShadow": "0 3px 12px rgba(0,0,0,0.13)",
            "minWidth": "270px",
            "display": "none",   # shown by callback when AMR project is active
        },
    )


def define_inner_layout(
    fig, buttons, map_layout_dict,
    filter_options=None, project_name=None,
    project_options=None, selected_project_value=None,
    has_amr=False, specimen_options=None, organism_options=None,
):
    """
    Inner layout: full-height map + AMR controls overlay + side menu.

    Parameters
    ----------
    has_amr           : True when the active project has microbiology data
    specimen_options  : [{"label": ..., "value": ...}] for specimen dropdown
    organism_options  : [{"label": ..., "value": ...}] for organism dropdown
    """
    amr_controls = _amr_map_controls(
        specimen_options=specimen_options,
        organism_options=organism_options,
    )

    if has_amr:
        amr_controls.style["display"] = "block"

    return html.Div(
        [
            dcc.Store(id="button", data={"item": "", "label": "", "suffix": ""}),
            dcc.Store(id="map-layout", data=map_layout_dict),
            dcc.Store(id="amr-project-active", data=has_amr),
            # Full-height map
            dcc.Graph(id="world-map", figure=fig, style={"height": "92vh", "margin": "0px"}),
            # AMR floating controls
            amr_controls,
            # Side menu
            define_menu(
                buttons,
                filter_options=filter_options,
                project_name=project_name,
                project_options=project_options,
                selected_project_value=selected_project_value,
            ),
            html.Div(id="trigger-on-load", style={"display": "none"}),
        ],
        style={"position": "relative"},
    )
