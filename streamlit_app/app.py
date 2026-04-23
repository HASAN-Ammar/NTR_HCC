# ============================================================
# Streamlit app: HCC Recurrence and NTR Risk (RF)
# Stage1_Preop: P(Recurrence)
# Stage2_Preop: P(NTR | Recurrence)
# Overall: P(NTR) = P(Recurrence) * P(NTR | Recurrence)
# ============================================================

import io
import json
import os
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "shiny_app", "models")
STAGE1_MODEL_PATH = os.path.join(MODELS_DIR, "stage1_rf.pkl")
STAGE2_MODEL_PATH = os.path.join(MODELS_DIR, "stage2_rf.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.json")

# ----------------------------
# Helpers
# ----------------------------

def normalize_col(c: str) -> str:
    c = "".join(ch for ch in unicodedata.normalize("NFKD", str(c)) if not unicodedata.combining(ch))
    c = c.strip()
    c = re.sub(r"[^\w]+", "_", c)
    c = re.sub(r"_+", "_", c)
    return c.strip("_")


def load_features() -> list[str]:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["preop_features"]


def compute_outcome_label(df: pd.DataFrame) -> pd.Series:
    rec = df.get("Recurrence_target", np.nan)
    ntr = df.get("NTR_target", np.nan)
    labels = np.select(
        [
            rec == 0,
            ntr == 0,
            ntr == 1,
        ],
        [
            "No recurrence",
            "In Milan",
            "Out Milan",
        ],
        default="Unknown",
    )
    return pd.Series(labels, index=df.index)


def build_predictions(df: pd.DataFrame, stage1_model, stage2_model, features: list[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required features: {missing}")

    X = df[features]
    p_recur = stage1_model.predict_proba(X)[:, 1]
    p_ntr_given = stage2_model.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "P_Recurrence": p_recur,
        "P_NTR_given_Recurrence": p_ntr_given,
        "P_NTR": p_recur * p_ntr_given,
    })
    return out


def validate_uploaded_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed_categories = {
        "Gender": {"M", "F"},
        "ALBI_grade": {"Grade 1", "Grade 2", "Grade 3"},
        "BCLC_before_intervention": {"0", "A", "B", "C"},
    }
    binary_cols = {
        "Alcohol",
        "HCV",
        "HBV",
        "NASH",
        "Hemochromatosis",
        "Cirrhosis",
    }
    numeric_cols = {
        "Age_at_intervention",
        "Largest_nodule_diameter",
        "Number_of_tumors_on_the_specimen",
        "Preop_AFP",
    }

    errors = []

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.upper()
        df.loc[df["Gender"].isin(["MALE", "MASCULIN", "MAN"]), "Gender"] = "M"
        df.loc[df["Gender"].isin(["FEMALE", "FEMME", "WOMAN"]), "Gender"] = "F"

    if "BCLC_before_intervention" in df.columns:
        df["BCLC_before_intervention"] = df["BCLC_before_intervention"].astype(str).str.strip().str.upper()

    if "ALBI_grade" in df.columns:
        df["ALBI_grade"] = df["ALBI_grade"].astype(str).str.strip()
        df.loc[df["ALBI_grade"].isin(["1", "GRADE 1", "G1"]), "ALBI_grade"] = "Grade 1"
        df.loc[df["ALBI_grade"].isin(["2", "GRADE 2", "G2"]), "ALBI_grade"] = "Grade 2"
        df.loc[df["ALBI_grade"].isin(["3", "GRADE 3", "G3"]), "ALBI_grade"] = "Grade 3"

    for col, allowed in allowed_categories.items():
        if col in df.columns:
            mask = ~df[col].isna() & ~df[col].astype(str).isin(allowed)
            if mask.any():
                bad_vals = sorted(df.loc[mask, col].astype(str).unique().tolist())
                errors.append({"column": col, "invalid_values": ", ".join(bad_vals)})

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df.loc[df[col].isin(["yes", "y", "true", "t", "1"]), col] = "1"
            df.loc[df[col].isin(["no", "n", "false", "f", "0"]), col] = "0"
            mask = ~df[col].isna() & ~df[col].isin([0, 1, "0", "1"])
            if mask.any():
                bad_vals = sorted(df.loc[mask, col].astype(str).unique().tolist())
                errors.append({"column": col, "invalid_values": ", ".join(bad_vals)})
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in numeric_cols:
        if col in df.columns:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if (coerced.isna() & ~df[col].isna()).any():
                errors.append({"column": col, "invalid_values": "Non-numeric values"})
            df[col] = coerced

    return df, pd.DataFrame(errors)


def coerce_input_df(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = {
        "Age_at_intervention",
        "Largest_nodule_diameter",
        "Number_of_tumors_on_the_specimen",
        "Preop_AFP",
    }
    binary_cols = {
        "Alcohol",
        "HCV",
        "HBV",
        "NASH",
        "Hemochromatosis",
        "Cirrhosis",
    }
    categorical_cols = {
        "Gender",
        "ALBI_grade",
        "BCLC_before_intervention",
    }
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def recommendation_text(p_stage1: float, p_stage2: float, t1: float, t2: float) -> str:
    if p_stage1 >= t1 and p_stage2 >= t2:
        return "High Stage 1 risk + high Stage 2 risk -> consider early transplant evaluation"
    if p_stage1 < t1:
        return "Low Stage 1 risk -> resection appropriate"
    if p_stage1 >= t1 and p_stage2 < t2:
        return "High Stage 1 but low Stage 2 -> resection with standard surveillance"
    return "Insufficient data to generate recommendation"

# ----------------------------
# Load models
# ----------------------------
if not os.path.exists(STAGE1_MODEL_PATH) or not os.path.exists(STAGE2_MODEL_PATH):
    st.error("Missing model files. Run shiny_app/train_models.py to generate models.")
    st.stop()

stage1_model = joblib.load(STAGE1_MODEL_PATH)
stage2_model = joblib.load(STAGE2_MODEL_PATH)
preop_features = load_features()

# ----------------------------
# UI
# ----------------------------

st.title("Hepatocellular Carcinoma (HCC): Risk of Recurrence and Non-Transplantable Recurrence")
st.caption("Two-stage prediction model (Random Forest)")

st.markdown(
    """
    <style>
    /* Custom styles can go here, but background and image removed */
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Clinical use case")
st.sidebar.markdown(
    """
**How to interpret outputs**

- High Stage 1 risk + high Stage 2 risk -> consider early transplant evaluation
- Low Stage 1 risk -> resection appropriate
- High Stage 1 but low Stage 2 -> resection with standard surveillance
"""
)

st.sidebar.header("Risk thresholds")
threshold_stage1 = st.sidebar.slider("Stage 1 risk threshold", 0.0, 1.0, 0.50, 0.01)
threshold_stage2 = st.sidebar.slider("Stage 2 risk threshold", 0.0, 1.0, 0.50, 0.01)
show_full_labels = st.sidebar.checkbox("Show full quadrant labels", value=False)

st.header("Upload data")
file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

st.divider()

st.header("Manual entry (one patient)")
binary_options = [0, 1]

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["M", "F"], key="gender_manual")
with col2:
    age_at_intervention = st.number_input("Age at intervention", min_value=1, value=1, step=1, format="%d", key="age_manual")
with col3:
    alcohol = st.selectbox("Alcohol (0/1)", binary_options, key="alcohol_manual")

col4, col5, col6, col7 = st.columns(4)
with col4:
    vhc = st.selectbox("VHC (0/1)", binary_options, key="vhc_manual")
with col5:
    vhb = st.selectbox("VHB (0/1)", binary_options, key="vhb_manual")
with col6:
    nash = st.selectbox("NASH (0/1)", binary_options, key="nash_manual")
with col7:
    hemochromatosis = st.selectbox("Hemochromatosis (0/1)", binary_options, key="hemo_manual")

col8, col9, col10 = st.columns(3)
with col8:
    albi_grade = st.selectbox("ALBI grade", ["Grade 1", "Grade 2", "Grade 3"], key="albi_manual")
with col9:
    largest_nodule_diameter = st.number_input("Size of the largest nodule (cm)", min_value=1, value=1, step=1, format="%d", key="nodule_manual")
with col10:
    number_of_tumors = st.number_input("Number of tumors", min_value=0, value=0, step=1, format="%d", key="tumors_manual")

col11, col12, col13 = st.columns(3)
with col11:
    bclc_before_intervention = st.selectbox("BCLC before intervention", ["0", "A", "B", "C"], key="bclc_manual")
with col12:
    preop_afp = st.number_input("Preop AFP", min_value=0, value=0, step=1, format="%d", key="afp_manual")
with col13:
    cirrhosis = st.selectbox("Cirrhosis (0/1)", binary_options, key="cirrhosis_manual")

manual = {
    "Gender": gender,
    "Age_at_intervention": age_at_intervention,
    "Alcohol": alcohol,
    "HCV": vhc,
    "HBV": vhb,
    "NASH": nash,
    "Hemochromatosis": hemochromatosis,
    "ALBI_grade": albi_grade,
    "Largest_nodule_diameter": largest_nodule_diameter,
    "Number_of_tumors_on_the_specimen": number_of_tumors,
    "BCLC_before_intervention": bclc_before_intervention,
    "Preop_AFP": preop_afp,
    "Cirrhosis": cirrhosis,
}

run_manual = st.button("Run manual prediction")

st.divider()

st.header("Predictions")

pred_df = None

if file is not None:
    name = file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    df.columns = [normalize_col(c) for c in df.columns]

    df, validation_errors = validate_uploaded_df(df)
    if not validation_errors.empty:
        st.error("Uploaded file has invalid values. Please fix and re-upload.")
        st.dataframe(validation_errors)
        df = None
    
    if df is not None:
        df = coerce_input_df(df)
        preds = build_predictions(df, stage1_model, stage2_model, preop_features)
        pred_df = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        if "Recurrence_target" in pred_df.columns or "NTR_target" in pred_df.columns:
            pred_df["Outcome_label"] = compute_outcome_label(pred_df)

if run_manual:
    df_manual = pd.DataFrame([manual])
    df_manual = coerce_input_df(df_manual)
    preds = build_predictions(df_manual, stage1_model, stage2_model, preop_features)
    pred_df = pd.concat([df_manual, preds], axis=1)
    pred_df["Recommendation"] = [
        recommendation_text(
            float(pred_df.loc[0, "P_Recurrence"]),
            float(pred_df.loc[0, "P_NTR_given_Recurrence"]),
            float(threshold_stage1),
            float(threshold_stage2),
        )
    ]


if pred_df is None:
    st.info("Upload a file or run manual prediction.")
else:
    st.dataframe(pred_df)

    if "P_Recurrence" in pred_df.columns and "P_NTR_given_Recurrence" in pred_df.columns:
        st.subheader("Clinical recommendation")
        p1 = float(pred_df.iloc[0]["P_Recurrence"])
        p2 = float(pred_df.iloc[0]["P_NTR_given_Recurrence"])
        st.success(recommendation_text(p1, p2, float(threshold_stage1), float(threshold_stage2)))

    st.subheader("Download results")
    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        if "Recommendation" not in pred_df.columns and "P_Recurrence" in pred_df.columns:
            pred_df = pred_df.copy()
            pred_df["Recommendation"] = [
                recommendation_text(float(a), float(b), float(threshold_stage1), float(threshold_stage2))
                for a, b in zip(pred_df["P_Recurrence"], pred_df["P_NTR_given_Recurrence"])
            ]
        pred_df.to_excel(writer, index=False)
    buff.seek(0)
    st.download_button(
        label="Download Excel",
        data=buff,
        file_name="predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Quadrant plot")
    plot_df = pred_df.copy()
    plot_df["Recommendation"] = [
        recommendation_text(float(a), float(b), float(threshold_stage1), float(threshold_stage2))
        for a, b in zip(plot_df["P_Recurrence"], plot_df["P_NTR_given_Recurrence"])
    ]
    fig = px.scatter(
        plot_df,
        x="P_Recurrence",
        y="P_NTR_given_Recurrence",
        hover_data={
            "P_Recurrence": ":.3f",
            "P_NTR_given_Recurrence": ":.3f",
            "Recommendation": True,
        },
        title="Clinical decision quadrants",
        range_x=[0, 1],
        range_y=[0, 1],
        opacity=0.9,
        color_discrete_sequence=["#0074D9"],
        height=600,
    )
    # Add quadrant lines (bolder and solid)
    fig.add_shape(
        type="line",
        x0=float(threshold_stage1),
        x1=float(threshold_stage1),
        y0=0,
        y1=1,
        line=dict(color="#FF4136", dash="solid", width=3),
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=float(threshold_stage2),
        y1=float(threshold_stage2),
        line=dict(color="#2ECC40", dash="solid", width=3),
    )
    if show_full_labels:
        label_low = "Low Stage 1 risk -> resection appropriate"
        label_high_low = "High Stage 1 but low Stage 2 -> resection with standard surveillance"
        label_high_high = "High Stage 1 + high Stage 2 -> consider early transplant evaluation"
    else:
        label_low = "Low Stage 1 risk"
        label_high_low = "High Stage 1, low Stage 2"
        label_high_high = "High Stage 1 + High Stage 2"

    # Improved quadrant annotations with background and larger font
    fig.add_annotation(
        x=0.15,
        y=0.85,
        xref="x",
        yref="y",
        text=label_low,
        showarrow=False,
        font=dict(size=16, color="#222"),
        align="center",
        bgcolor="#E3F6FC",
        bordercolor="#0074D9",
        borderpad=4,
        opacity=0.95,
    )
    fig.add_annotation(
        x=0.85,
        y=0.15,
        xref="x",
        yref="y",
        text=label_high_low,
        showarrow=False,
        font=dict(size=16, color="#222"),
        align="center",
        bgcolor="#F6FCE3",
        bordercolor="#2ECC40",
        borderpad=4,
        opacity=0.95,
    )
    fig.add_annotation(
        x=0.85,
        y=0.85,
        xref="x",
        yref="y",
        text=label_high_high,
        showarrow=False,
        font=dict(size=16, color="#222"),
        align="center",
        bgcolor="#FCE3E3",
        bordercolor="#FF4136",
        borderpad=4,
        opacity=0.95,
    )
    fig.update_traces(marker=dict(color="#0074D9", size=18, line=dict(color="#fff", width=2)))
    fig.update_layout(
        xaxis_title="P(Recurrence)",
        yaxis_title="P(NTR | Recurrence)",
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="#f9f9f9",
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Arial", size=14),
    )
    st.plotly_chart(fig, use_container_width=True)
