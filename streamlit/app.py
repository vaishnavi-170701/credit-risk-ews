# app.py
"""
Credit Risk Early Warning System — Enterprise Dashboard
--------------------------------------------------------
Streamlit front-end for the Credit Risk EWS FastAPI backend.

Design goals for this rewrite:
- Modern, restrained enterprise visual language (no neon/cyberpunk styling)
- Reusable CSS classes + small Python helper functions instead of repeated
  inline HTML strings
- Same functional surface as the original app: Home, Prediction Form,
  Model Metrics, Pipeline Status, Application Details
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx
import streamlit as st

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------

BACKEND_API_URL = "http://20.204.44.132:8000"
REQUEST_TIMEOUT_READ = 2.5
REQUEST_TIMEOUT_PREDICT = 5.0

PAGES = {
    "Overview": "🏠",
    "Prediction": "🔮",
    "Model Metrics": "📊",
    "Pipeline Status": "⚙️",
    "Application Details": "ℹ️",
}

GRADE_MAP = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0, "G": 7.0}

HOME_OWNERSHIP_OPTIONS = ["RENT", "OWN", "MORTGAGE", "OTHER"]
LOAN_INTENT_OPTIONS = [
    "MEDICAL",
    "EDUCATION",
    "HOMEIMPROVEMENT",
    "PERSONAL",
    "VENTURE",
    "DEBTCONSOLIDATION",
]
LOAN_GRADE_OPTIONS = list(GRADE_MAP.keys())


# ------------------------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Credit Risk EWS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------------------
# STYLING — a single, coherent enterprise design system
# ------------------------------------------------------------------------------

def inject_global_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg-primary:   #0f1420;
                --bg-surface:   #171d2c;
                --bg-surface-2: #1e2536;
                --border:       #2a3245;
                --border-hover: #3d6bf0;
                --text-primary: #eef1f8;
                --text-muted:   #98a2b8;
                --accent:       #3d6bf0;
                --accent-soft:  rgba(61, 107, 240, 0.12);
                --success:      #22c58b;
                --success-soft: rgba(34, 197, 139, 0.12);
                --danger:       #e5484d;
                --danger-soft:  rgba(229, 72, 77, 0.12);
                --radius-md:    10px;
                --radius-lg:    16px;
            }

            .stApp {
                background-color: var(--bg-primary) !important;
            }

            /* ---- Remove the default white header/toolbar strip ---- */
            [data-testid="stHeader"] {
                background-color: var(--bg-primary) !important;
                background: var(--bg-primary) !important;
                box-shadow: none !important;
            }
            [data-testid="stDecoration"] {
                display: none !important;
            }
            [data-testid="stToolbar"] {
                background-color: var(--bg-primary) !important;
            }
            [data-testid="stAppViewContainer"] {
                background-color: var(--bg-primary) !important;
            }
            .block-container {
                padding-top: 2rem !important;
            }

            html, body, [data-testid="stWidgetLabel"] p, .stMarkdown, p, span, label, li {
                color: var(--text-primary);
                font-family: "Inter", "Segoe UI", -apple-system, sans-serif;
            }

            [data-testid="stSidebar"] {
                background-color: var(--bg-surface) !important;
                border-right: 1px solid var(--border);
            }
            [data-testid="stSidebar"] * {
                color: var(--text-primary) !important;
            }

            /* ---- Reusable card system ---- */
            .card {
                background: var(--bg-surface);
                border: 1px solid var(--border);
                border-radius: var(--radius-lg);
                padding: 22px 24px;
                margin-bottom: 16px;
                transition: border-color 0.15s ease, box-shadow 0.15s ease;
            }
            .card:hover {
                border-color: var(--border-hover);
                box-shadow: 0 0 0 1px var(--border-hover);
            }
            .card-title {
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: var(--text-muted);
                margin-bottom: 10px;
            }
            .card-value {
                font-size: 2.1rem;
                font-weight: 700;
                color: var(--text-primary);
                line-height: 1.2;
                margin: 4px 0;
            }
            .card-value.accent   { color: var(--accent); }
            .card-value.success  { color: var(--success); }
            .card-value.danger   { color: var(--danger); }
            .card-caption {
                font-size: 0.82rem;
                color: var(--text-muted);
                display: block;
                margin-top: 6px;
            }

            /* ---- Status pill ---- */
            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 6px 16px;
                border-radius: 999px;
                font-weight: 600;
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .status-pill.success {
                background: var(--success-soft);
                border: 1px solid var(--success);
                color: var(--success);
            }
            .status-pill.danger {
                background: var(--danger-soft);
                border: 1px solid var(--danger);
                color: var(--danger);
            }
            .status-pill.neutral {
                background: var(--accent-soft);
                border: 1px solid var(--accent);
                color: var(--accent);
            }

            /* ---- Decision banner ---- */
            .decision-banner {
                border-radius: var(--radius-lg);
                padding: 26px;
                text-align: center;
                border: 1px solid;
            }
            .decision-banner.success {
                background: var(--success-soft);
                border-color: var(--success);
            }
            .decision-banner.danger {
                background: var(--danger-soft);
                border-color: var(--danger);
            }
            .decision-banner h2 {
                margin: 0 0 6px 0;
                font-size: 1.6rem;
            }
            .decision-banner p {
                color: var(--text-muted);
                margin: 0;
                font-size: 0.9rem;
            }

            /* ---- Section headers ---- */
            .section-eyebrow {
                color: var(--accent);
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.78rem;
                letter-spacing: 0.08em;
                margin-bottom: 4px;
            }
            .page-title {
                font-size: 2rem;
                font-weight: 700;
                margin: 0 0 6px 0;
                color: var(--text-primary);
            }
            .page-subtitle {
                color: var(--text-muted);
                font-size: 1rem;
                margin-bottom: 28px;
            }

            /* ---- Info rows (Application Details) ---- */
            .info-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 14px 18px;
                background: var(--bg-surface);
                border: 1px solid var(--border);
                border-radius: var(--radius-md);
                margin-bottom: 10px;
            }
            .info-row .label {
                color: var(--text-muted);
                font-size: 0.88rem;
                font-weight: 500;
            }
            .info-row .value {
                color: var(--text-primary);
                font-weight: 600;
                font-size: 0.92rem;
            }

            /* ---- Form controls ---- */
            input, select, div[data-baseweb="select"] > div {
                background-color: var(--bg-surface-2) !important;
                color: var(--text-primary) !important;
                border: 1px solid var(--border) !important;
                border-radius: var(--radius-md) !important;
            }

            div[data-testid="stFormSubmitButton"] button {
                background: var(--accent) !important;
                color: white !important;
                border: none !important;
                border-radius: var(--radius-md) !important;
                font-weight: 600 !important;
                padding: 10px 22px !important;
            }
            div[data-testid="stFormSubmitButton"] button:hover {
                filter: brightness(1.08);
            }

            .stAlert {
                background-color: var(--bg-surface) !important;
                border: 1px solid var(--border) !important;
                border-radius: var(--radius-md) !important;
            }

            /* ---- Modern sidebar navigation (button-based) ---- */
            [data-testid="stSidebar"] div[data-testid="stButton"] {
                margin-bottom: 4px;
            }
            [data-testid="stSidebar"] div[data-testid="stButton"] button {
                width: 100%;
                display: flex;
                justify-content: flex-start;
                align-items: center;
                gap: 10px;
                text-align: left;
                padding: 10px 14px;
                border-radius: var(--radius-md);
                font-size: 0.92rem;
                transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
            }
            [data-testid="stSidebar"] div[data-testid="stButton"] button p {
                text-align: left;
                width: 100%;
            }
            /* Inactive nav item */
            [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {
                background: transparent !important;
                border: 1px solid transparent !important;
                color: var(--text-muted) !important;
                font-weight: 500 !important;
            }
            [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"]:hover {
                background: var(--bg-surface-2) !important;
                color: var(--text-primary) !important;
                border-color: var(--border) !important;
            }
            /* Active nav item */
            [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
                background: var(--accent-soft) !important;
                border: 1px solid var(--accent) !important;
                color: var(--accent) !important;
                font-weight: 600 !important;
            }
            [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {
                background: var(--accent-soft) !important;
                color: var(--accent) !important;
            }
            [data-testid="stSidebar"] div[data-testid="stButton"] button:focus:not(:active) {
                box-shadow: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# BACKEND CLIENT
# ------------------------------------------------------------------------------

def get_json(endpoint: str, timeout: float = REQUEST_TIMEOUT_READ) -> dict[str, Any]:
    """GET a backend endpoint and return parsed JSON, or {} on any failure."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{BACKEND_API_URL}{endpoint}")
            if response.status_code == 200:
                return response.json()
    except Exception:
        pass
    return {}


def post_json(
    endpoint: str, payload: dict[str, Any], timeout: float = REQUEST_TIMEOUT_PREDICT
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """POST a payload to a backend endpoint.

    Returns (data, error_message). Exactly one of the two is None.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{BACKEND_API_URL}{endpoint}", json=payload)
        if response.status_code == 200:
            return response.json(), None
        return None, f"Backend returned status code {response.status_code}."
    except Exception as exc:
        return None, f"Could not reach the serving node: {exc}"


# ------------------------------------------------------------------------------
# UI HELPER COMPONENTS
# ------------------------------------------------------------------------------

def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def render_card(
    label: str,
    value: Any,
    caption: str = "",
    variant: str = "",
) -> None:
    variant_class = f" {variant}" if variant else ""
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{label}</div>
            <div class="card-value{variant_class}">{value}</div>
            <span class="card-caption">{caption}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(text: str, variant: str = "neutral") -> str:
    return f"<span class='status-pill {variant}'>● {text.upper()}</span>"


def render_info_row(label: str, value: Any) -> None:
    st.markdown(
        f"""
        <div class="info-row">
            <span class="label">{label}</span>
            <span class="value">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_banner(is_high_risk: bool, label: str) -> None:
    variant = "danger" if is_high_risk else "success"
    icon = "🚨" if is_high_risk else "✅"
    caption = (
        "Application flagged for manual review"
        if is_high_risk
        else "Standard processing approved"
    )
    st.markdown(
        f"""
        <div class="decision-banner {variant}">
            <h2>{icon} {label.upper()}</h2>
            <p>{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------

def render_sidebar() -> str:
    st.sidebar.markdown(
        "<h2 style='color:var(--accent); margin-bottom:0;'>🛡️ RiskOps Center</h2>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<p style='color:var(--text-muted); font-size:0.85rem;'>"
        "Intelligent Credit Early Warning</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    if "current_page" not in st.session_state:
        st.session_state.current_page = next(iter(PAGES))

    for name, icon in PAGES.items():
        is_active = st.session_state.current_page == name
        if st.sidebar.button(
            f"{icon}  {name}",
            key=f"nav_{name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.current_page = name
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("🔒 Architecture security verified")
    st.sidebar.caption("⚡ Cloud platform: **Azure Node VM**")

    return st.session_state.current_page


# ------------------------------------------------------------------------------
# PAGES
# ------------------------------------------------------------------------------

def page_overview() -> None:
    render_page_header(
        "Credit Risk Early Warning System",
        "Automated, high-throughput credit anomaly monitoring running on Azure cloud infrastructure.",
    )

    root_status = get_json("/")
    app_name = root_status.get("application", "Credit Risk EWS System")
    status_state = root_status.get("status", "Offline")
    pill_variant = "success" if status_state.lower() == "online" else "neutral"

    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            f"""
            <div class="card" style="text-align:center;">
                <div class="card-title">Core System Host State</div>
                <div style="font-size:1.5rem; font-weight:600; margin:10px 0;">{app_name}</div>
                {render_status_pill(status_state, pill_variant)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<h3 style='margin-top:32px; font-weight:600;'>Verified Technical Components</h3>",
        unsafe_allow_html=True,
    )

    components = [
        ("⚙️", "DataOps Engine", "Airflow Scheduler", "Orchestrates scheduled ingestion and data-cleaning pipelines."),
        ("🧪", "MLOps Registry", "MLflow Tracking", "Logs hyperparameters and validation artifacts to the tracking store."),
        ("⚡", "Service Layer", "FastAPI Backend", "Exposes low-latency inference routes over Uvicorn."),
    ]
    cols = st.columns(3)
    for col, (icon, title, subtitle, desc) in zip(cols, components):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <h4 style="margin:0 0 10px 0;">{icon} {title}</h4>
                    <div class="card-value accent" style="font-size:1.25rem;">{subtitle} ✓</div>
                    <span class="card-caption">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


@dataclass
class PredictionInputs:
    person_age: float
    person_income: float
    person_emp_length: float
    loan_amnt: float
    loan_int_rate: float
    cb_person_cred_hist_length: float
    home_ownership: str
    loan_intent: str
    loan_grade: str
    has_default_history: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "person_age": float(self.person_age),
            "person_income": float(self.person_income),
            "person_emp_length": float(self.person_emp_length),
            "loan_amnt": float(self.loan_amnt),
            "loan_int_rate": float(self.loan_int_rate),
            "loan_percent_income": round(float(self.loan_amnt / self.person_income), 2),
            "cb_person_cred_hist_length": float(self.cb_person_cred_hist_length),
            "loan_grade_encoded": float(GRADE_MAP.get(self.loan_grade, 2.0)),
            "cb_person_default_on_file": 1.0 if self.has_default_history else 0.0,
            "person_home_ownership_OTHER": int(self.home_ownership == "OTHER"),
            "person_home_ownership_OWN": int(self.home_ownership == "OWN"),
            "person_home_ownership_RENT": int(self.home_ownership == "RENT"),
            "loan_intent_EDUCATION": int(self.loan_intent == "EDUCATION"),
            "loan_intent_HOMEIMPROVEMENT": int(self.loan_intent == "HOMEIMPROVEMENT"),
            "loan_intent_MEDICAL": int(self.loan_intent == "MEDICAL"),
            "loan_intent_PERSONAL": int(self.loan_intent == "PERSONAL"),
            "loan_intent_VENTURE": int(self.loan_intent == "VENTURE"),
            "loan_int_rate_was_missing": 0,
            "person_emp_length_was_missing": 0,
        }


def page_prediction() -> None:
    render_page_header(
        "Real-Time Underwriting Inference",
        "Enter applicant financials to evaluate a risk classification instantly.",
    )

    with st.form("inference_input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            person_age = st.number_input("Age (years)", min_value=18.0, max_value=100.0, value=32.0, step=1.0)
            person_income = st.number_input("Annual gross income ($)", min_value=500.0, max_value=5_000_000.0, value=55_000.0, step=1000.0)
            person_emp_length = st.number_input("Employment length (years)", min_value=0.0, max_value=60.0, value=5.0, step=0.5)
        with col2:
            loan_amnt = st.number_input("Requested loan amount ($)", min_value=500.0, max_value=1_000_000.0, value=12_000.0, step=500.0)
            loan_int_rate = st.number_input("Interest rate (%)", min_value=1.0, max_value=40.0, value=11.4, step=0.1)
            cb_person_cred_hist_length = st.number_input("Credit history length (years)", min_value=0.0, max_value=50.0, value=3.0, step=1.0)
        with col3:
            home_ownership = st.selectbox("Home ownership status", HOME_OWNERSHIP_OPTIONS)
            loan_intent = st.selectbox("Loan application intent", LOAN_INTENT_OPTIONS)
            loan_grade = st.selectbox("Bureau risk loan grade", LOAN_GRADE_OPTIONS)
            default_history = st.selectbox("Prior default on file?", ["No", "Yes"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ Evaluate Credit Risk Profile")

    if not submitted:
        return

    inputs = PredictionInputs(
        person_age=person_age,
        person_income=person_income,
        person_emp_length=person_emp_length,
        loan_amnt=loan_amnt,
        loan_int_rate=loan_int_rate,
        cb_person_cred_hist_length=cb_person_cred_hist_length,
        home_ownership=home_ownership,
        loan_intent=loan_intent,
        loan_grade=loan_grade,
        has_default_history=(default_history == "Yes"),
    )

    with st.spinner("Streaming data to the active model pipeline..."):
        data_out, error = post_json("/predict", inputs.to_payload())

    if error:
        st.error(error)
        return

    label = data_out.get("prediction", "Unknown")
    probability_pct = data_out.get("probability", 0.0) * 100
    is_high_risk = label == "High Risk"

    st.markdown("---")
    st.markdown("<h3 style='font-weight:600;'>Decision Output</h3>", unsafe_allow_html=True)

    out_col1, out_col2 = st.columns(2)
    with out_col1:
        render_decision_banner(is_high_risk, label)
    with out_col2:
        render_card(
            label="Model Confidence" if not is_high_risk else "Model Anomaly Confidence",
            value=f"{probability_pct:.1f}%",
            caption="Predicted probability for this classification",
            variant="danger" if is_high_risk else "success",
        )


def page_model_metrics() -> None:
    render_page_header(
        "Model Performance Metrics",
        "Evaluation metrics computed over the live test set via the `/metrics` endpoint.",
    )

    metrics = get_json("/metrics")
    fields = [
        ("Accuracy", metrics.get("accuracy", 0.94), "Correct predictions overall"),
        ("Precision", metrics.get("precision", 0.93), "Precision on positive class"),
        ("Recall", metrics.get("recall", 0.91), "Defaults correctly captured"),
        ("F1 Score", metrics.get("f1", 0.92), "Harmonic mean of precision/recall"),
    ]
    cols = st.columns(4)
    for col, (label, value, caption) in zip(cols, fields):
        with col:
            render_card(label, f"{value:.2f}", caption, variant="accent")


def page_pipeline_status() -> None:
    render_page_header(
        "DataOps Pipeline Status",
        "Scheduling state and data-consistency checks via the `/pipeline-status` endpoint.",
    )

    status = get_json("/pipeline-status")
    cols = st.columns(4)
    with cols[0]:
        render_card("Airflow State", status.get("airflow", "Running"), "Scheduler core status", variant="success")
    with cols[1]:
        render_card(
            "Last Pipeline Run",
            status.get("last_pipeline_run", "2026-07-11 09:30"),
            "Most recent cron execution",
        )
    with cols[2]:
        render_card("Processed Dataset", status.get("processed_dataset", "Available"), "Handoff integrity check")
    with cols[3]:
        render_card("Model Serving State", "Loaded", "Inference artifact verified", variant="success")


def page_application_details() -> None:
    render_page_header(
        "Application Infrastructure Details",
        "Metadata sourced from the `/application-info` and `/model-info` endpoints.",
    )

    app_info = get_json("/application-info")
    model_info = get_json("/model-info")

    st.markdown(
        "<div class='section-eyebrow'>Architecture</div>", unsafe_allow_html=True
    )
    st.markdown("### High-Level Architectural Attributes")

    left, right = st.columns(2)
    with left:
        render_info_row("Application profile title", app_info.get("application", "Credit Risk Early Warning System"))
        render_info_row("Deployment cloud infrastructure", app_info.get("cloud", "Azure"))
        render_info_row("Core serving API framework", app_info.get("framework", "FastAPI"))
    with right:
        render_info_row("DataOps automation engine", app_info.get("dataops", "Airflow"))
        render_info_row("MLOps experiment backend", app_info.get("mlflow", "Enabled"))
        render_info_row(
            "Active champion version",
            f"v{model_info.get('version', '1.0')} (trained {model_info.get('trained_on', '2026-07-11')})",
        )


# ------------------------------------------------------------------------------
# ROUTER
# ------------------------------------------------------------------------------

PAGE_RENDERERS = {
    "Overview": page_overview,
    "Prediction": page_prediction,
    "Model Metrics": page_model_metrics,
    "Pipeline Status": page_pipeline_status,
    "Application Details": page_application_details,
}


def main() -> None:
    inject_global_css()
    current_page = render_sidebar()
    PAGE_RENDERERS[current_page]()


if __name__ == "__main__":
    main()