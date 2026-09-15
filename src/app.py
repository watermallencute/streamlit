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

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)
BEST_THRESHOLD = 0.7425999999999402

RISK_BINS = [-0.001, 0.3, BEST_THRESHOLD, 1.001]
RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]

# REFUND_ACTION_MAP = {
#     "Refundable": "Full refund, resell the room",
#     "No Deposit": "No refund applies, resell the room",
#     "Non Refund": "No need to refund, resell the room"
# }


ACTION_MAP = {
    "Low Risk": "Routine monitoring without intervention",
    "Medium Risk": "Send an automated reminder",
    "High Risk": "Reconfirmation to guest (still come/change the date/cancel)"
}

def get_recommended_action(risk_label):
    return ACTION_MAP.get(risk_label, "Unknown risk level")

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

discrete_num_cols = [
    column for column in num_cols
    if column != "adr"
]

continuous_num_cols = [
    "adr",
]

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

    values["has_children"] = int(values["children"] > 0)
    values["has_babies"] = int(values["babies"] > 0)
    values["is_family"] = int(
        values["has_children"] == 1 or values["has_babies"] == 1
    )
    values["has_agent"] = int(values["agent"] != "no agent")
    values["has_company"] = int(values["company"] != "no company")
    values["room_type_changed"] = int(
        values["assigned_room_type"] != values["reserved_room_type"]
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([values])

    prediction_probability = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_probability >= BEST_THRESHOLD)

    risk_label = pd.cut(
        [prediction_probability],
        bins=RISK_BINS,
        labels=RISK_LABELS,
        right=False,
    )[0]

    assert (risk_label == "High Risk") == (prediction == 1), (
        "RISK_BINS is not in sync with BEST_THRESHOLD - check the configuration above."
    )

    recommended_action = get_recommended_action(risk_label)

    result = (
        "Booking will be canceled"
        if prediction == 1
        else "Booking will not be canceled"
    )

    risk_style_class = {
        "Low Risk": "risk-low",
        "Medium Risk": "",
        "High Risk": "risk-high",
    }[risk_label]

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
                <div class="result-value">{result}</div>
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