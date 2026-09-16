from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "final_model"
    / "XGBoost.pkl"
)
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "hotel_booking_2017_cleaned.csv"
)

BEST_THRESHOLD = 0.7425999999999402

RISK_BINS = [-0.001, 0.3, BEST_THRESHOLD, 1.001]
RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]

ACTION_MAP = {
    "Low Risk": "Routine monitoring without intervention",
    "Medium Risk": "Send an automated reminder",
    "High Risk": "Reconfirmation to guest (still come/change the date/cancel)",
}

RISK_PILL_CLASS = {
    "Low Risk": "low",
    "Medium Risk": "medium",
    "High Risk": "high",
}

RISK_ROW_COLOR = {
    "Low Risk": "background-color: #f0fdf4",
    "Medium Risk": "background-color: #fffbeb",
    "High Risk": "background-color: #fef2f2",
}

RISK_EMOJI_LABEL = {
    "Low Risk": "🟢 Low Risk",
    "Medium Risk": "🟡 Medium Risk",
    "High Risk": "🔴 High Risk",
}

STYLER_CELL_LIMIT = 262_144

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
data = load_data()


num_cols = [
    "lead_time",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

discrete_num_cols = [column for column in num_cols if column != "adr"]
continuous_num_cols = ["adr"]

cat_cols = [
    "hotel",
    "arrival_date_year",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "agent",
    "company",
]

RAW_INPUT_COLS = num_cols + cat_cols

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["has_children"] = (df["children"] > 0).astype(int)
    df["has_babies"] = (df["babies"] > 0).astype(int)
    df["is_family"] = ((df["has_children"] == 1) | (df["has_babies"] == 1)).astype(int)
    df["has_agent"] = (df["agent"] != "no agent").astype(int)
    df["has_company"] = (df["company"] != "no company").astype(int)
    df["room_type_changed"] = (
        df["assigned_room_type"] != df["reserved_room_type"]
    ).astype(int)
    return df


def classify_risk(probabilities):
    return pd.cut(probabilities, bins=RISK_BINS, labels=RISK_LABELS, right=False)


def get_recommended_action(risk_label):
    return ACTION_MAP.get(risk_label, "Unknown risk level")


def compute_predictions(engineered_df: pd.DataFrame):
    probabilities = model.predict_proba(engineered_df)[:, 1]
    predictions = (probabilities >= BEST_THRESHOLD).astype(int)
    risk_labels = classify_risk(probabilities)
    actions = [get_recommended_action(label) for label in risk_labels]
    return probabilities, predictions, risk_labels, actions


def fill_missing_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort fill for missing values in an uploaded CSV, using medians
    / modes from the reference dataset. This does NOT replace a real
    imputation strategy from your pipeline - it just keeps obviously
    incomplete rows from crashing the whole batch."""
    filled = df.copy()
    for column in num_cols:
        if column in filled.columns and filled[column].isna().any():
            filled[column] = filled[column].fillna(data[column].median())
    for column in cat_cols:
        if column in filled.columns and filled[column].isna().any():
            filled[column] = filled[column].fillna(data[column].mode().iloc[0])
    return filled

st.set_page_config(page_title="Hotel Booking Prediction")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .block-container {
        max-width: 1080px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.01em;
        border-bottom: 2px solid #35507A;
        padding-bottom: 0.6rem;
        margin-bottom: 0 !important;
    }

    /* description / brief panel */
    .brief {
        border-left: 3px solid #35507A;
        background: #FBFAF7;
        padding: 0.95rem 1.3rem;
        margin: 1.1rem 0 1.6rem 0;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #4B5567;
    }
    .brief b { color: #1B2431; font-weight: 600; }
    .brief code {
        background: #F0EEE7;
        color: #1B2431;
        padding: 0.05rem 0.35rem;
        border-radius: 3px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85em;
    }
    .brief ol, .brief ul { margin: 0.4rem 0 0.7rem 1.15rem; padding: 0; }
    .brief li { margin-bottom: 0.25rem; }
    .brief .brief-title {
        font-weight: 600;
        color: #1B2431;
        font-size: 0.8rem;
        letter-spacing: 0.02em;
        margin-bottom: 0.35rem;
    }

    /* section labels replacing default subheaders */
    .section-label {
        font-weight: 600;
        font-size: 0.98rem;
        color: #1B2431;
        border-bottom: 1px solid #D9D4C7;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }

    div.stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.02rem;
        font-weight: 600;
        background: #1B2431;
        color: #FBFAF7;
        border: none;
        border-radius: 4px;
    }
    div.stButton > button:hover {
        background: #35507A;
        color: #fff;
    }

    /* result / risk-assessment panel */
    .assessment {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 1.6rem;
        border: 1px solid #D9D4C7;
        border-left: 5px solid #35507A;
        border-radius: 6px;
        margin-top: 1.7rem;
        padding: 1.3rem 1.6rem;
        background: #ffffff;
    }
    .assessment.low { border-left-color: #1F7A4D; }
    .assessment.medium { border-left-color: #B45309; }
    .assessment.high { border-left-color: #B3231C; }

    .assessment-figure {
        flex: 0 0 150px;
    }
    .assessment-figure .figure {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.3rem;
        font-weight: 600;
        color: #1B2431;
        line-height: 1.1;
    }
    .assessment-figure .figure-label {
        font-size: 0.72rem;
        color: #4B5567;
        margin-top: 0.4rem;
    }
    .assessment-details {
        flex: 1 1 320px;
        border-left: 1px solid #F0EEE7;
        padding-left: 1.6rem;
    }
    .assessment-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #F0EEE7;
    }
    .assessment-row:last-child { border-bottom: none; }
    .assessment-row .label {
        font-size: 0.82rem;
        color: #4B5567;
        white-space: nowrap;
    }
    .assessment-row .value {
        font-weight: 600;
        font-size: 0.95rem;
        text-align: right;
        color: #1B2431;
    }
    .risk-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid;
    }
    .risk-pill.low { color: #1F7A4D; background: #EAF6EE; border-color: #BFE3CC; }
    .risk-pill.medium { color: #B45309; background: #FFFBEB; border-color: #FDE68A; }
    .risk-pill.high { color: #B3231C; background: #FCEBEA; border-color: #F2C4C0; }

    /* batch summary stats (replaces st.metric so risk tiers can carry their own color) */
    .stat-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 0.6rem 0 1.4rem 0;
    }
    .stat {
        flex: 1 1 140px;
        border: 1px solid #D9D4C7;
        border-top: 3px solid #35507A;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        background: #ffffff;
    }
    .stat.high { border-top-color: #B3231C; }
    .stat.medium { border-top-color: #B45309; }
    .stat.low { border-top-color: #1F7A4D; }
    .stat .stat-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #1B2431;
        line-height: 1.1;
    }
    .stat .stat-label {
        font-size: 0.78rem;
        color: #4B5567;
        margin-top: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Hotel Booking Cancellation Prediction")

st.markdown(
    """
    <div class="brief">
        <div class="brief-title">Tentang</div>
        Aplikasi ini memperkirakan risiko pembatalan booking hotel dengan model <b>XGBoost</b>
        yang dilatih pada data booking 2015&ndash;2016 dan diuji pada data 2017.
        Setiap booking mendapat probabilitas pembatalan, tier risiko, dan rekomendasi tindakan
        operasional.
        <div class="brief-title">Cara Kerja</div>
        <ol>
            <li>Isi detail booking, atau unggah CSV berisi banyak booking.</li>
            <li>Data diproses lewat feature engineering yang sama seperti saat training.</li>
            <li>Model menghitung probabilitas pembatalan, lalu mengelompokkannya ke 3 tier risiko (Low, Medium, High).</li>
            <li>Setiap tier risiko dipetakan ke rekomendasi tindakan: monitoring rutin,
                reminder otomatis, atau konfirmasi ulang ke tamu.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

single_tab, batch_tab = st.tabs(["Single Booking", "Batch (CSV Upload)"])

with single_tab:
    with st.form("prediction_form"):
        values = {}

        st.markdown('<div class="section-label">Booking Details</div>', unsafe_allow_html=True)

        for start in range(0, len(num_cols), 2):
            input_columns = st.columns(2)
            for container, column in zip(input_columns, num_cols[start:start + 2]):
                with container:
                    if column in discrete_num_cols:
                        values[column] = st.number_input(
                            column,
                            min_value=0,
                            value=int(data[column].median()),
                            step=1,
                        )
                    else:
                        values[column] = st.number_input(
                            column,
                            value=float(data[column].median()),
                            step=0.01,
                            format="%.2f",
                        )

        st.markdown('<div class="section-label">Booking Categories</div>', unsafe_allow_html=True)

        for start in range(0, len(cat_cols), 2):
            input_columns = st.columns(2)
            for container, column in zip(input_columns, cat_cols[start:start + 2]):
                with container:
                    options = data[column].dropna().unique().tolist()
                    values[column] = st.selectbox(column, options)

        submitted = st.form_submit_button("Predict")

    if submitted:
        raw_input = pd.DataFrame([values])
        input_data = engineer_features(raw_input)

        probabilities, predictions, risk_labels, actions = compute_predictions(input_data)
        prediction_probability = probabilities[0]
        prediction = predictions[0]
        risk_label = risk_labels[0]
        recommended_action = actions[0]

        assert (risk_label == "High Risk") == (prediction == 1), (
            "RISK_BINS is not in sync with BEST_THRESHOLD - check the configuration above."
        )

        result_text = (
            "Booking will be canceled"
            if prediction == 1
            else "Booking will not be canceled"
        )
        risk_pill_class = RISK_PILL_CLASS[risk_label]

        st.markdown(
            f"""
            <div class="assessment {risk_pill_class}">
                <div class="assessment-figure">
                    <div class="figure">{prediction_probability:.1%}</div>
                    <div class="figure-label">cancellation probability</div>
                </div>
                <div class="assessment-details">
                    <div class="assessment-row">
                        <span class="label">Outcome</span>
                        <span class="value">{result_text}</span>
                    </div>
                    <div class="assessment-row">
                        <span class="label">Risk tier</span>
                        <span class="value">
                            <span class="risk-pill {risk_pill_class}">{risk_label}</span>
                        </span>
                    </div>
                    <div class="assessment-row">
                        <span class="label">Recommended action</span>
                        <span class="value">{recommended_action}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


with batch_tab:
    st.markdown('<div class="section-label">Upload a CSV of bookings</div>', unsafe_allow_html=True)
    st.caption(
        "The file must contain all columns listed below. Missing values are "
        "filled with the median (numeric) or mode (categorical) from the "
        "reference dataset as a fallback."
    )

    template_df = pd.DataFrame(columns=RAW_INPUT_COLS)
    st.download_button(
        "Download CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="booking_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Booking CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read this file as CSV. Details: {exc}")
            uploaded_df = None

        if uploaded_df is not None:
            missing_columns = set(RAW_INPUT_COLS) - set(uploaded_df.columns)
            if missing_columns:
                st.error(
                    "The uploaded CSV is missing required columns: "
                    + ", ".join(sorted(missing_columns))
                )
            else:
                working_df = fill_missing_batch(uploaded_df[RAW_INPUT_COLS])
                engineered_batch = engineer_features(working_df)

                try:
                    probabilities, predictions, risk_labels, actions = compute_predictions(
                        engineered_batch
                    )
                except Exception as exc:
                    st.error(
                        "Prediction failed on this batch, most likely due to a "
                        f"category value the model hasn't seen before. Details: {exc}"
                    )
                else:
                    results_df = engineered_batch.copy()
                    results_df.insert(0, "recommended_action", actions)
                    results_df.insert(0, "risk_tier", risk_labels.astype(str))
                    results_df.insert(
                        0,
                        "prediction",
                        [
                            "Will be canceled" if p == 1 else "Will not be canceled"
                            for p in predictions
                        ],
                    )
                    results_df.insert(0, "cancellation_probability", probabilities)

                    total = len(results_df)
                    high_risk = int((results_df["risk_tier"] == "High Risk").sum())
                    medium_risk = int((results_df["risk_tier"] == "Medium Risk").sum())
                    low_risk = int((results_df["risk_tier"] == "Low Risk").sum())

                    st.markdown(
                        f"""
                        <div class="stat-row">
                            <div class="stat">
                                <div class="stat-value">{total}</div>
                                <div class="stat-label">Total bookings</div>
                            </div>
                            <div class="stat high">
                                <div class="stat-value">{high_risk}</div>
                                <div class="stat-label">High risk</div>
                            </div>
                            <div class="stat medium">
                                <div class="stat-value">{medium_risk}</div>
                                <div class="stat-label">Medium risk</div>
                            </div>
                            <div class="stat low">
                                <div class="stat-value">{low_risk}</div>
                                <div class="stat-label">Low risk</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    risk_filter = st.multiselect(
                        "Filter by risk tier",
                        options=RISK_LABELS,
                        default=RISK_LABELS,
                    )
                    filtered_df = results_df[results_df["risk_tier"].isin(risk_filter)]

                    display_df = filtered_df.copy()
                    display_df["risk_tier"] = display_df["risk_tier"].map(
                        RISK_EMOJI_LABEL
                    ).fillna(display_df["risk_tier"])

                    num_cells = display_df.shape[0] * display_df.shape[1]
                    if num_cells <= STYLER_CELL_LIMIT:
                        styled_df = display_df.style.format(
                            {"cancellation_probability": "{:.2%}"}
                        )
                        st.dataframe(styled_df, use_container_width=True)
                    else:
                        display_df = display_df.copy()
                        display_df["cancellation_probability"] = display_df[
                            "cancellation_probability"
                        ].map("{:.2%}".format)
                        st.caption(
                            f"Showing {len(display_df):,} rows without extra row "
                            "styling - the batch is too large for full cell "
                            "formatting. Risk tier is still color-coded via emoji."
                        )
                        st.dataframe(display_df, use_container_width=True)

                    st.download_button(
                        "Download results as CSV",
                        data=filtered_df.to_csv(index=False).encode("utf-8"),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                    )