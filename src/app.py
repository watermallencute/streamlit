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

RISK_STYLE_CLASS = {
    "Low Risk": "risk-low",
    "Medium Risk": "",
    "High Risk": "risk-high",
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
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    div.stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.05rem;
        font-weight: 600;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 14px;
        min-height: 135px;
        border: 1px solid #bfdbfe;
        background: #eff6ff !important;
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-title {
        color: #1e3a8a !important;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    .result-value {
        color: #172554 !important;
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.35;
    }

    .probability-value {
        font-size: 2.4rem;
    }

    .outcome-card {
        border-color: #bbf7d0;
        background: #f0fdf4 !important;
    }

    .outcome-card .result-title {
        color: #166534 !important;
    }

    .outcome-card .result-value {
        color: #14532d !important;
    }

    .risk-card {
        border-color: #fde68a;
        background: #fffbeb !important;
    }

    .risk-card .result-title {
        color: #92400e !important;
    }

    .risk-card .result-value {
        color: #78350f !important;
    }

    .risk-card.risk-low {
        border-color: #bbf7d0;
        background: #f0fdf4 !important;
    }

    .risk-card.risk-low .result-title,
    .risk-card.risk-low .result-value {
        color: #166534 !important;
    }

    .risk-card.risk-high {
        border-color: #fecaca;
        background: #fef2f2 !important;
    }

    .risk-card.risk-high .result-title,
    .risk-card.risk-high .result-value {
        color: #991b1b !important;
    }

    .action-card {
        border-color: #e5e7eb;
        background: #f9fafb !important;
        text-align: left;
    }

    .action-card .result-title {
        color: #374151 !important;
        text-align: center;
    }

    .action-card .result-value {
        color: #111827 !important;
        font-size: 1.1rem;
        font-weight: 600;
        line-height: 1.5;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Hotel Booking Cancellation Prediction")
st.caption("Enter the booking details to generate a cancellation prediction.")

single_tab, batch_tab = st.tabs(["Single Booking", "Batch (CSV Upload)"])

with single_tab:
    with st.form("prediction_form"):
        values = {}

        st.subheader("Booking Details")

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

        st.subheader("Booking Categories")

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
        risk_style_class = RISK_STYLE_CLASS[risk_label]

        result_columns = st.columns(2)
        with result_columns[0]:
            st.markdown(
                f"""
                <div class="result-card probability-card">
                    <div class="result-title">Cancellation Probability</div>
                    <div class="result-value probability-value">
                        {prediction_probability:.2%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with result_columns[1]:
            st.markdown(
                f"""
                <div class="result-card outcome-card">
                    <div class="result-title">Prediction Result</div>
                    <div class="result-value">{result_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        detail_columns = st.columns(2)
        with detail_columns[0]:
            st.markdown(
                f"""
                <div class="result-card risk-card {risk_style_class}">
                    <div class="result-title">Risk Tier</div>
                    <div class="result-value">{risk_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with detail_columns[1]:
            st.markdown(
                f"""
                <div class="result-card action-card">
                    <div class="result-title">Recommended Action</div>
                    <div class="result-value">{recommended_action}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


with batch_tab:
    st.subheader("Upload a CSV of bookings")
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

                    metric_columns = st.columns(4)
                    metric_columns[0].metric("Total bookings", total)
                    metric_columns[1].metric("High risk", high_risk)
                    metric_columns[2].metric("Medium risk", medium_risk)
                    metric_columns[3].metric("Low risk", low_risk)

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