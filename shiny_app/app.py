from __future__ import annotations

import io
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

# ----------------------------
# CONFIG
# ----------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
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


# ----------------------------
# Load models
# ----------------------------
if not os.path.exists(STAGE1_MODEL_PATH) or not os.path.exists(STAGE2_MODEL_PATH):
    raise FileNotFoundError(
        "Missing model files. Run shiny_app/train_models.py to generate models."
    )

stage1_model = joblib.load(STAGE1_MODEL_PATH)
stage2_model = joblib.load(STAGE2_MODEL_PATH)
preop_features = load_features()

# ----------------------------
# UI
# ----------------------------
app_ui = ui.page_fluid(
    ui.h2("HCC Recurrence and NTR Risk (RF)") ,
    ui.p("Stage1: P(Recurrence). Stage2: P(NTR | Recurrence). Overall: P(NTR)=product."),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.h4("Upload data"),
            ui.input_file("data_file", "Upload CSV/XLSX", accept=[".csv", ".xlsx", ".xls"]),
            ui.hr(),
            ui.h4("Manual entry (one patient)"),
            ui.input_text("Gender", "Gender", value=""),
            ui.input_numeric("Age_at_intervention", "Age at intervention", value=0),
            ui.input_numeric("Alcohol", "Alcohol (0/1)", value=0),
            ui.input_numeric("HCV", "HCV (0/1)", value=0),
            ui.input_numeric("HBV", "HBV (0/1)", value=0),
            ui.input_numeric("NASH", "NASH (0/1)", value=0),
            ui.input_numeric("Hemochromatosis", "Hemochromatosis (0/1)", value=0),
            ui.input_text("ALBI_grade", "ALBI grade", value=""),
            ui.input_numeric("Largest_nodule_diameter", "Largest nodule diameter", value=0),
            ui.input_numeric("Number_of_tumors_on_the_specimen", "Number of tumors", value=0),
            ui.input_text("BCLC_before_intervention", "BCLC before intervention", value=""),
            ui.input_numeric("Preop_AFP", "Preop AFP", value=0),
            ui.input_numeric("Cirrhosis", "Cirrhosis (0/1)", value=0),
            ui.input_action_button("run_manual", "Run manual prediction"),
        ),
        ui.panel_main(
            ui.h4("Predictions"),
            ui.output_table("pred_table"),
            ui.hr(),
            ui.download_button("download_results", "Download results (XLSX)"),
            ui.hr(),
            ui.h4("P(NTR) distribution"),
            ui.output_plot("p_ntr_hist"),
        ),
    ),
)

# ----------------------------
# Server
# ----------------------------

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.calc
    def uploaded_df() -> pd.DataFrame | None:
        fileinfo = input.data_file()
        if not fileinfo:
            return None
        path = fileinfo[0]["datapath"]
        name = fileinfo[0]["name"].lower()
        if name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        df.columns = [normalize_col(c) for c in df.columns]
        return df

    @reactive.calc
    def upload_predictions() -> pd.DataFrame | None:
        df = uploaded_df()
        if df is None:
            return None
        preds = build_predictions(df, stage1_model, stage2_model, preop_features)
        out = df.copy()
        out = pd.concat([out.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        if "Recurrence_target" in out.columns or "NTR_target" in out.columns:
            out["Outcome_label"] = compute_outcome_label(out)
        return out

    @reactive.calc
    def manual_predictions() -> pd.DataFrame | None:
        if input.run_manual() == 0:
            return None
        row = {
            "Gender": input.Gender(),
            "Age_at_intervention": input.Age_at_intervention(),
            "Alcohol": input.Alcohol(),
            "HCV": input.HCV(),
            "HBV": input.HBV(),
            "NASH": input.NASH(),
            "Hemochromatosis": input.Hemochromatosis(),
            "ALBI_grade": input.ALBI_grade(),
            "Largest_nodule_diameter": input.Largest_nodule_diameter(),
            "Number_of_tumors_on_the_specimen": input.Number_of_tumors_on_the_specimen(),
            "BCLC_before_intervention": input.BCLC_before_intervention(),
            "Preop_AFP": input.Preop_AFP(),
            "Cirrhosis": input.Cirrhosis(),
        }
        df = pd.DataFrame([row])
        preds = build_predictions(df, stage1_model, stage2_model, preop_features)
        return pd.concat([df, preds], axis=1)

    @render.table
    def pred_table():
        if upload_predictions() is not None:
            return upload_predictions().head(200)
        if manual_predictions() is not None:
            return manual_predictions()
        return pd.DataFrame({"Info": ["Upload a file or run manual prediction."]})

    @render.plot
    def p_ntr_hist():
        df = upload_predictions()
        if df is None or "P_NTR" not in df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Upload data to see distribution", ha="center", va="center")
            ax.axis("off")
            return fig
        fig, ax = plt.subplots()
        ax.hist(df["P_NTR"].values, bins=20, color="#2b6cb0", edgecolor="white")
        ax.set_xlabel("P(NTR)")
        ax.set_ylabel("Count")
        ax.set_title("P(NTR) distribution")
        return fig

    @render.download(filename="predictions.xlsx")
    def download_results():
        df = upload_predictions()
        if df is None:
            return None
        buff = io.BytesIO()
        with pd.ExcelWriter(buff, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        buff.seek(0)
        return buff


app = App(app_ui, server)
