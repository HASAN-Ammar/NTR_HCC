# ============================================================
# Train RF models for Shiny app deployment
# Stage1_Preop: P(Recurrence)
# Stage2_Preop: P(NTR | Recurrence)
# ============================================================

import json
import os
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

# ----------------------------
# CONFIG (EDIT THESE)
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "HCC_dataset_export.xlsx"))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "models"))

TRAIN_ON_ALL_CENTERS = True
EXTERNAL_CENTER = "Montpellier"  # used only if TRAIN_ON_ALL_CENTERS is False

CV_MODE = "stratified"  # "group_by_center" or "stratified"
N_SPLITS = 10
RANDOM_STATE = 0
TUNE_HYPERPARAMS = True
N_JOBS = -1

# ----------------------------
# Helpers
# ----------------------------

def normalize_col(c: str) -> str:
    c = "".join(ch for ch in unicodedata.normalize("NFKD", str(c)) if not unicodedata.combining(ch))
    c = c.strip()
    c = re.sub(r"[^\w]+", "_", c)
    c = re.sub(r"_+", "_", c)
    return c.strip("_")


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def get_cv_splitter(X, y, groups):
    if CV_MODE == "group_by_center":
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            raise ValueError("Need at least 2 centers in TRAIN for GroupKFold.")
        n_splits = min(n_groups, 3)
        return GroupKFold(n_splits=n_splits)
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def roc_auc_proba_scorer(estimator, X, y):
    proba = estimator.predict_proba(X)
    scores = proba[:, 1] if proba.ndim > 1 else proba
    return roc_auc_score(y, scores)

# ----------------------------
# Load data
# ----------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

file_ext = os.path.splitext(DATA_PATH)[1].lower()
if file_ext in [".xlsx", ".xls"]:
    data = pd.read_excel(DATA_PATH)
else:
    try:
        data = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8")
    except (UnicodeDecodeError, pd.errors.ParserError):
        data = pd.read_csv(
            DATA_PATH,
            sep=";",
            encoding="latin1",
            engine="python",
            on_bad_lines="skip",
        )

data.columns = [normalize_col(c) for c in data.columns]

if "Center" not in data.columns:
    raise KeyError("Missing required column: Center")

# ----------------------------
# Targets
# ----------------------------
if "mode_de_recidive" in data.columns:
    s_rec = data["mode_de_recidive"].astype(str).str.strip().str.lower()
    data["Recurrence_target"] = np.select(
        [
            s_rec.str.contains("no recurrence"),
            s_rec.str.contains("milan"),
        ],
        [
            0,
            1,
        ],
        default=np.nan,
    )
elif "recurrence" in data.columns:
    data["Recurrence_target"] = data["recurrence"].astype(int)
else:
    raise KeyError("Missing recurrence target column: expected 'mode_de_recidive' or 'recurrence'.")

if "mode_de_recidive" not in data.columns:
    raise KeyError("Missing 'mode_de_recidive' column needed for NTR target.")

s = data["mode_de_recidive"].astype(str).str.strip().str.lower()
# Accept variants like "out milan", "out of milan", "milan out", "in milan"
is_out = s.str.contains("milan") & s.str.contains("out")
is_in = s.str.contains("milan") & s.str.contains("in")
data["NTR_target"] = np.where(is_out, 1, np.where(is_in, 0, np.nan))

data.loc[data["Recurrence_target"] == 0, "NTR_target"] = np.nan

# ----------------------------
# Feature sets (preop only)
# ----------------------------
preop_features = [
    "Gender", "Age_at_intervention", "Alcohol", "HCV", "HBV",
    "NASH", "Hemochromatosis", "ALBI_grade",
    "Largest_nodule_diameter", "Number_of_tumors_on_the_specimen",
    "BCLC_before_intervention", "Preop_AFP", "Cirrhosis",
]

missing_feats = [c for c in preop_features if c not in data.columns]
if missing_feats:
    raise KeyError(f"Missing preop features: {missing_feats}")

# ----------------------------
# Models and tuning
# ----------------------------
RF = RandomForestClassifier(
    n_estimators=600,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    max_features="sqrt",
)

param_grid_rf = {
    "model__n_estimators": [300, 600],
    "model__max_depth": [None, 6, 12],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt", "log2"],
}

# ----------------------------
# Stage1 (Recurrence)
# ----------------------------
stage1_df = data.dropna(subset=["Recurrence_target"]).copy()
if not TRAIN_ON_ALL_CENTERS:
    stage1_df = stage1_df[stage1_df["Center"] != EXTERNAL_CENTER].copy()

X_train1 = stage1_df[preop_features]
Y_train1 = stage1_df["Recurrence_target"].astype(int).values
G_train1 = stage1_df["Center"].values

pipe1 = Pipeline([("prep", make_preprocessor(X_train1)), ("model", RF)])
if TUNE_HYPERPARAMS and np.unique(Y_train1).size > 1:
    cv1 = get_cv_splitter(X_train1, Y_train1, G_train1)
    search1 = GridSearchCV(
        pipe1,
        param_grid_rf,
        scoring=roc_auc_proba_scorer,
        cv=cv1,
        n_jobs=N_JOBS,
    )
    if CV_MODE == "group_by_center":
        search1.fit(X_train1, Y_train1, groups=G_train1)
    else:
        search1.fit(X_train1, Y_train1)
    final1 = search1.best_estimator_
else:
    final1 = pipe1.fit(X_train1, Y_train1)

# ----------------------------
# Stage2 (NTR | Recurrence)
# ----------------------------
stage2_df = data[data["Recurrence_target"] == 1].dropna(subset=["NTR_target"]).copy()
if not TRAIN_ON_ALL_CENTERS:
    stage2_df = stage2_df[stage2_df["Center"] != EXTERNAL_CENTER].copy()

X_train2 = stage2_df[preop_features]
Y_train2 = stage2_df["NTR_target"].astype(int).values
G_train2 = stage2_df["Center"].values

pipe2 = Pipeline([("prep", make_preprocessor(X_train2)), ("model", RF)])
if TUNE_HYPERPARAMS and np.unique(Y_train2).size > 1:
    cv2 = get_cv_splitter(X_train2, Y_train2, G_train2)
    search2 = GridSearchCV(
        pipe2,
        param_grid_rf,
        scoring=roc_auc_proba_scorer,
        cv=cv2,
        n_jobs=N_JOBS,
    )
    if CV_MODE == "group_by_center":
        search2.fit(X_train2, Y_train2, groups=G_train2)
    else:
        search2.fit(X_train2, Y_train2)
    final2 = search2.best_estimator_
else:
    final2 = pipe2.fit(X_train2, Y_train2)

# ----------------------------
# Save artifacts
# ----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

joblib.dump(final1, os.path.join(OUTPUT_DIR, "stage1_rf.pkl"))
joblib.dump(final2, os.path.join(OUTPUT_DIR, "stage2_rf.pkl"))

with open(os.path.join(OUTPUT_DIR, "features.json"), "w", encoding="utf-8") as f:
    json.dump({"preop_features": preop_features}, f, indent=2)

print("\n✅ Saved models to:")
print(" -", os.path.join(OUTPUT_DIR, "stage1_rf.pkl"))
print(" -", os.path.join(OUTPUT_DIR, "stage2_rf.pkl"))
print(" -", os.path.join(OUTPUT_DIR, "features.json"))
