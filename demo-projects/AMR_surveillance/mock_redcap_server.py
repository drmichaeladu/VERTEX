"""Mock REDCap API server for VERTEX AMR Surveillance demo.

Serves a REDCap-compatible POST /api/ endpoint on port 5001, backed by
the CSV files in analysis_data/.  The VERTEX pipeline
(vertex/getREDCapData.py) can consume this without modification.
"""

import io
import os

import numpy as np
import pandas as pd
from flask import Flask, Response, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VALID_TOKEN = "AMR_MOCK_TOKEN_2024_GHANA_ABCDEF12"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_THIS_DIR, "analysis_data")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _csv_response(text, status=200):
    return Response(text, status=status, mimetype="text/csv")


def _read_csv(filename):
    return pd.read_csv(os.path.join(_DATA_DIR, filename), keep_default_na=False)


# ---------------------------------------------------------------------------
# Metadata (data dictionary) — built once, cached
# ---------------------------------------------------------------------------

_METADATA_CSV = None


def _build_metadata():
    """Return the full REDCap-style data dictionary as a CSV string."""
    global _METADATA_CSV
    if _METADATA_CSV is not None:
        return _METADATA_CSV

    cols = [
        "field_name", "form_name", "section_header", "field_type",
        "field_label", "select_choices_or_calculations", "field_note",
        "text_validation_type_or_show_slider_number",
        "text_validation_min", "text_validation_max", "identifier",
        "branching_logic", "required_field", "custom_alignment",
        "question_number", "matrix_group_name", "matrix_ranking",
        "field_annotation",
    ]

    rows = []

    def _add(field_name, form_name, field_type, field_label,
             choices="", validation="", **kw):
        row = {c: "" for c in cols}
        row["field_name"] = field_name
        row["form_name"] = form_name
        row["field_type"] = field_type
        row["field_label"] = field_label
        row["select_choices_or_calculations"] = choices
        row["text_validation_type_or_show_slider_number"] = validation
        row.update(kw)
        rows.append(row)

    # --- presentation ---
    _add("subjid", "presentation", "text", "Subject ID")
    _add("demog_age", "presentation", "text", "Age", validation="number")
    _add("demog_sex", "presentation", "radio", "Sex",
         choices="1, Male | 2, Female | 3, Other / Unknown")
    _add("country_iso", "presentation", "text", "Country ISO Code")
    _add("pres_date", "presentation", "text", "Presentation Date",
         validation="date_dmy")
    _add("amr_ward", "presentation", "radio", "Ward",
         choices="1, In-patient | 2, Out-patient")
    _add("ghana_region", "presentation", "radio", "Ghana Region",
         choices="1, Greater Accra | 2, Ashanti | 3, Northern | 4, Volta | 5, Eastern | 6, Western | 7, Central | 8, Upper East | 9, Upper West | 10, Bono East | 11, Ahafo | 12, Western North | 13, Oti | 14, Bono | 15, North East | 16, Savannah")
    _add("ghana_region_iso", "presentation", "text", "Ghana Region ISO Code")
    _add("facility_id", "presentation", "text", "Facility ID")
    _add("facility_type", "presentation", "radio", "Facility Type",
         choices="1, Teaching Hospital | 2, Regional Hospital | 3, District Hospital | 4, Specialist Hospital | 5, Research Centre")
    _add("facility_lat", "presentation", "text", "Facility Latitude",
         validation="number")
    _add("facility_lon", "presentation", "text", "Facility Longitude",
         validation="number")
    _add("demog_country", "presentation", "radio", "Country",
         choices="1, Ghana")

    # --- outcome ---
    _add("outco_outcome", "outcome", "radio", "Outcome",
         choices="1, Discharged alive | 2, Death | 3, Censored")

    # --- microbiology (repeating) ---
    micro_sir = "1, S | 2, I | 3, R"
    _add("micro_sample_date", "microbiology", "text", "Sample Date",
         validation="date_dmy")
    _add("micro_specimen_type", "microbiology", "radio", "Specimen Type",
         choices="1, Blood | 2, Urine | 3, Sputum | 4, Wound | 5, CSF | 6, Other")
    _add("micro_organism", "microbiology", "radio", "Organism",
         choices="1, Escherichia coli | 2, Klebsiella pneumoniae | 3, Staphylococcus aureus | 4, Staphylococcus, coagulase negative | 5, Streptococcus pneumoniae | 6, Pseudomonas aeruginosa | 7, Acinetobacter baumannii | 8, Enterococcus species | 9, Other")
    _add("micro_ward", "microbiology", "radio", "Ward at Collection",
         choices="1, In-patient | 2, Out-patient")
    _add("micro_origin", "microbiology", "radio", "Infection Origin",
         choices="1, Community | 2, Hospital")
    _add("micro_age_group", "microbiology", "radio", "Age Group",
         choices="1, 0-17 | 2, 18-64 | 3, 65+")
    _add("micro_sex", "microbiology", "radio", "Sex",
         choices="1, Male | 2, Female | 3, Other / Unknown")
    _add("ghana_region", "microbiology", "text", "Ghana Region")
    _add("ghana_region_iso", "microbiology", "text", "Ghana Region ISO Code")
    _add("facility_id", "microbiology", "text", "Facility ID")
    _add("facility_type", "microbiology", "text", "Facility Type")
    _add("prior_antibiotic_exposure", "microbiology", "yesno",
         "Prior Antibiotic Exposure")
    _add("community_acquired", "microbiology", "yesno",
         "Community Acquired")
    for ab in ["AMC", "AMK", "AMP", "CAZ", "CIP", "CRO", "CTX", "GEN",
               "LNZ", "MEM", "OXA", "SXT", "TZP", "VAN", "CLI", "ERY", "TET"]:
        _add(ab, "microbiology", "radio", f"{ab} Susceptibility",
             choices=micro_sir)

    # --- antibiotic_prescribing (repeating) ---
    _add("presc_date", "antibiotic_prescribing", "text", "Prescription Date",
         validation="date_dmy")
    _add("presc_antibiotic", "antibiotic_prescribing", "radio", "Antibiotic",
         choices="1, Amoxicillin | 2, Amoxicillin-Clavulanate | 3, Ceftriaxone | 4, Cefotaxime | 5, Ciprofloxacin | 6, Gentamicin | 7, Meropenem | 8, Metronidazole | 9, Vancomycin | 10, Colistin | 11, Other")
    _add("presc_route", "antibiotic_prescribing", "radio", "Route",
         choices="1, IV | 2, PO | 3, IM | 4, Other")
    _add("presc_duration_days", "antibiotic_prescribing", "text",
         "Duration (days)", validation="integer")
    _add("presc_ddd", "antibiotic_prescribing", "text",
         "Defined Daily Dose", validation="number")
    _add("presc_indication", "antibiotic_prescribing", "radio", "Indication",
         choices="1, Empiric | 2, Directed | 3, Prophylaxis | 4, Other")
    _add("presc_aware_category", "antibiotic_prescribing", "radio",
         "AWaRe Category", choices="1, Access | 2, Watch | 3, Reserve")
    _add("facility_id", "antibiotic_prescribing", "text", "Facility ID")
    _add("ghana_region", "antibiotic_prescribing", "text", "Ghana Region")
    _add("ghana_region_iso", "antibiotic_prescribing", "text",
         "Ghana Region ISO Code")

    df = pd.DataFrame(rows, columns=cols)
    _METADATA_CSV = df.to_csv(index=False)
    return _METADATA_CSV


# ---------------------------------------------------------------------------
# Record export — the critical endpoint
# ---------------------------------------------------------------------------

_RECORDS_CSV = None


def _build_records():
    """Build a flat CSV export combining patient, micro and prescribing rows."""
    global _RECORDS_CSV
    if _RECORDS_CSV is not None:
        return _RECORDS_CSV

    # Collect all field names from the metadata
    meta_df = pd.read_csv(io.StringIO(_build_metadata()))
    all_fields = meta_df["field_name"].unique().tolist()

    system_cols = [
        "subjid", "redcap_event_name", "redcap_repeat_instrument",
        "redcap_repeat_instance", "redcap_data_access_group",
    ]
    all_columns = system_cols + [f for f in all_fields if f not in system_cols]

    # --- Patient rows (from df_map.csv) ---
    df_map = _read_csv("df_map.csv")
    patient = pd.DataFrame("", index=range(len(df_map)), columns=all_columns)
    patient["subjid"] = df_map["subjid"].values
    patient["redcap_event_name"] = "Enrollment"
    patient["redcap_repeat_instrument"] = ""
    patient["redcap_repeat_instance"] = ""
    patient["redcap_data_access_group"] = "ghana"

    # Map df_map columns into the patient rows
    outcome_map = {
        "Survived": "Discharged alive",
        "Died": "Death",
        "Unknown": "Censored",
    }
    for col in df_map.columns:
        if col == "subjid":
            continue
        if col == "outco_binary_outcome":
            # Export as outco_outcome (pipeline creates outco_binary_outcome)
            patient["outco_outcome"] = df_map[col].map(
                lambda x: outcome_map.get(x, "Censored"))
            continue
        if col in patient.columns:
            patient[col] = df_map[col].values

    # demog_country is not in df_map.csv but needed for country_iso derivation
    patient["demog_country"] = "Ghana"

    # --- Microbiology rows ---
    df_micro = _read_csv("microbiology.csv")
    micro = pd.DataFrame("", index=range(len(df_micro)), columns=all_columns)
    micro["subjid"] = df_micro["subjid"].values
    micro["redcap_event_name"] = ""
    micro["redcap_repeat_instrument"] = "Microbiology"
    micro["redcap_data_access_group"] = "ghana"

    # Per-patient sequential instance numbers
    micro["redcap_repeat_instance"] = (
        df_micro.groupby("subjid").cumcount() + 1).values

    # Map yes/no fields
    yn_map = {1: "Yes", 0: "No", "1": "Yes", "0": "No"}
    for col in df_micro.columns:
        if col == "subjid":
            continue
        vals = df_micro[col].values
        if col in ("prior_antibiotic_exposure", "community_acquired"):
            vals = pd.Series(vals).map(yn_map).fillna("").values
        if col in micro.columns:
            micro[col] = vals

    # --- Prescribing rows ---
    df_presc = _read_csv("antibiotic_prescribing.csv")
    presc = pd.DataFrame("", index=range(len(df_presc)), columns=all_columns)
    presc["subjid"] = df_presc["subjid"].values
    presc["redcap_event_name"] = ""
    presc["redcap_repeat_instrument"] = "Antibiotic Prescribing"
    presc["redcap_data_access_group"] = "ghana"

    presc["redcap_repeat_instance"] = (
        df_presc.groupby("subjid").cumcount() + 1).values

    for col in df_presc.columns:
        if col == "subjid":
            continue
        if col in presc.columns:
            presc[col] = df_presc[col].values

    # Combine all row types
    combined = pd.concat([patient, micro, presc], ignore_index=True)
    _RECORDS_CSV = combined.to_csv(index=False)
    return _RECORDS_CSV


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.route("/api/", methods=["POST"])
def api():
    token = request.form.get("token", "")
    if token != VALID_TOKEN:
        return Response("Invalid token", status=403)

    content = request.form.get("content", "")

    if content == "dag":
        # User not assigned to DAG → 200
        return Response("", status=200)

    if content == "project":
        csv_text = (
            "project_id,project_title,missing_data_codes\n"
            "75738,Ghana AMR Surveillance,\"NI, Not indicated\"\n"
        )
        return _csv_response(csv_text)

    if content == "instrument":
        csv_text = (
            "instrument_name,instrument_label\n"
            "presentation,Presentation\n"
            "outcome,Outcome\n"
            "microbiology,Microbiology\n"
            "antibiotic_prescribing,Antibiotic Prescribing\n"
        )
        return _csv_response(csv_text)

    if content == "event":
        csv_text = (
            "event_name,arm_num,unique_event_name,custom_event_label,event_id\n"
            "Enrollment,1,enrollment_arm_1,,1\n"
        )
        return _csv_response(csv_text)

    if content == "formEventMapping":
        csv_text = (
            "arm_num,unique_event_name,form\n"
            "1,enrollment_arm_1,presentation\n"
            "1,enrollment_arm_1,outcome\n"
        )
        return _csv_response(csv_text)

    if content == "metadata":
        return _csv_response(_build_metadata())

    if content == "record":
        return _csv_response(_build_records())

    return Response(f"Unknown content type: {content}", status=400)


# ---------------------------------------------------------------------------
# Thread-based start (for embedding in the VERTEX dashboard process)
# ---------------------------------------------------------------------------

def start_mock_server_thread():
    """Start mock REDCap server in a background daemon thread."""
    import threading
    t = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0", port=5001, debug=False, use_reloader=False),
        daemon=True,
    )
    t.start()


if __name__ == "__main__":
    print("[mock_redcap_server] Starting on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)
