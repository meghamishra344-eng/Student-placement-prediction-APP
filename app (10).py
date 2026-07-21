"""
Student Placement Prediction System — v2
=========================================
Upgrades over v1:
  • Honest evaluation: the model is scored on a held-out 20% test set
    (accuracy, ROC-AUC, precision, recall) before being refit on all data.
  • Adjustable decision threshold in the sidebar (default 0.50, which
    maximised hold-out accuracy; raise it to flag more students for help).
  • Action plan: the top improvable weaknesses are turned into concrete
    "what-if" simulations — e.g. "raise Coding Score to 75 → probability
    goes from 48% to 61%".
  • Batch scoring: upload a CSV of many students, get a ranked risk list
    and a downloadable results file.
  • Model performance tab with confusion matrix and feature importances.
  • A custom "readiness band" visual and a cohesive visual theme.

The model still trains directly from the CSV at startup (cached), so there
is no pickle file and no Python-version mismatch.
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from matplotlib.patches import Rectangle
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATA_FILE = "Student_Placement_Dataset_10000_v2.csv"
DEFAULT_THRESHOLD = 0.50  # maximised accuracy on the hold-out set

SKILL_COLS = ["Aptitude_Score", "Coding_Score", "Communication_Score",
              "Technical_Score", "Mock_Interview_Score", "Resume_Score"]

RAW_NUM = ["10th_Percentage", "12th_Percentage", "Graduation_Percentage",
           "CGPA", "Backlogs", "Attendance", "Internship_Months",
           "Projects", "Certifications"] + SKILL_COLS
RAW_CAT = ["Internship"]
RAW_INPUTS = RAW_NUM + RAW_CAT  # what a user / uploaded CSV must provide

ENGINEERED = ["Academic_Avg", "Skill_Avg", "Total_Experience", "CGPA_x_Skill"]
FINAL_COLS = RAW_NUM + ENGINEERED + RAW_CAT

# Visual theme -----------------------------------------------------------------
INK = "#1C2B3A"       # deep slate — text & axes
PAPER = "#FAF9F6"     # warm off-white background
GREEN = "#1F7A5C"     # "ready / placed"
AMBER = "#D9A441"     # "borderline"
CORAL = "#C94F3D"     # "at risk"
MUTED = "#6B7A8C"

st.set_page_config(page_title="Megha Mishra's Placement Predictor",
                   page_icon="🎓", layout="wide")

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&display=swap');

h1, h2, h3 {{
    font-family: 'Fraunces', Georgia, serif !important;
    color: {INK} !important;
    letter-spacing: 0.2px;
}}
/* Force the light look for every visitor, even if their browser is in
   dark mode (otherwise Streamlit's dark theme paints white text on our
   light background and labels become invisible). */
:root {{ color-scheme: light; }}
.stApp {{ background-color: {PAPER}; }}
section[data-testid="stSidebar"] {{ background-color: #EFEDE6; }}
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"],
[data-testid="stSliderThumbValue"],
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"],
.stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] p {{
    color: {INK} !important;
}}
/* Inputs: keep white fields with dark text */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {{
    background-color: #ffffff !important; color: {INK} !important;
}}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    background-color: #ffffff !important; color: {INK} !important;
}}
[data-testid="stFileUploaderDropzone"] {{
    background-color: #ffffff !important; color: {INK} !important;
}}
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {{ color: {MUTED} !important; }}
/* Buttons keep light text on the green primary color */
.stButton button p, [data-testid="stFormSubmitButton"] button p,
[data-testid="stDownloadButton"] button p {{ color: #ffffff !important; }}

/* ---------- visual polish ---------- */
/* Hero */
.hero-eyebrow {{
    font-size: 0.72rem; letter-spacing: 2.5px; text-transform: uppercase;
    color: {MUTED}; margin-bottom: 2px; font-weight: 600;
}}
.hero-title {{
    font-family: 'Fraunces', Georgia, serif; font-weight: 650;
    font-size: 2.9rem; color: {INK}; line-height: 1.1; margin: 0;
}}
.hero-tag {{ color: {MUTED}; font-size: 1.02rem; margin-top: 8px; }}
.hero-rule {{
    height: 4px; border: none; border-radius: 2px; margin: 18px 0 6px 0;
    background: linear-gradient(90deg, {CORAL} 0%, {CORAL} 33%,
                {AMBER} 33%, {AMBER} 55%, {GREEN} 55%, {GREEN} 100%);
}}

/* Pill tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px; border-bottom: none; padding: 4px 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: #ffffff; border: 1px solid #E3E0D8; border-radius: 999px;
    padding: 6px 18px; transition: all .15s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{ border-color: {GREEN}; }}
.stTabs [aria-selected="true"] {{
    background: {INK} !important; border-color: {INK} !important;
}}
.stTabs [aria-selected="true"] p {{ color: {PAPER} !important; }}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* Card look for forms and expanders */
[data-testid="stForm"] {{
    background: #ffffff; border: 1px solid #E9E6DE; border-radius: 16px;
    padding: 1.6rem 1.8rem; box-shadow: 0 2px 14px rgba(28,43,58,0.06);
}}

/* Metric cards */
[data-testid="stMetric"] {{
    background: #ffffff; border: 1px solid #E9E6DE;
    border-left: 4px solid {GREEN}; border-radius: 12px;
    padding: 12px 16px; box-shadow: 0 1px 8px rgba(28,43,58,0.05);
}}
[data-testid="stMetricValue"] {{
    font-family: 'Fraunces', Georgia, serif; font-weight: 650;
}}

/* Buttons — several selectors so this works across Streamlit versions */
.stButton button, .stFormSubmitButton button, .stDownloadButton button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button,
button[kind="primaryFormSubmit"], button[kind="secondaryFormSubmit"],
[data-testid="stBaseButton-primaryFormSubmit"],
[data-testid="stBaseButton-secondaryFormSubmit"] {{
    background-color: {GREEN} !important;
    color: #ffffff !important;
    border-radius: 10px; border: none; font-weight: 600;
    box-shadow: 0 2px 8px rgba(31,122,92,0.25);
    transition: transform .1s ease, box-shadow .1s ease;
}}
.stButton button *, .stFormSubmitButton button *, .stDownloadButton button *,
[data-testid="stFormSubmitButton"] button *,
[data-testid="stDownloadButton"] button *,
button[kind="primaryFormSubmit"] *, button[kind="secondaryFormSubmit"] * {{
    color: #ffffff !important;
}}
.stButton button:hover, .stFormSubmitButton button:hover,
.stDownloadButton button:hover,
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {{
    background-color: #17654C !important;
    transform: translateY(-1px); box-shadow: 0 4px 12px rgba(31,122,92,0.35);
}}

/* Alert boxes: rounder, softer */
[data-testid="stAlert"] {{ border-radius: 12px; }}

/* Dataframe container */
[data-testid="stDataFrame"] {{
    border: 1px solid #E9E6DE; border-radius: 12px; overflow: hidden;
}}

/* Sidebar headers a touch smaller */
section[data-testid="stSidebar"] h2 {{ font-size: 1.25rem; }}
.readiness-wrap {{ margin: 0.4rem 0 1.2rem 0; }}
.readiness-band {{
    position: relative; height: 18px; border-radius: 9px; overflow: hidden;
    display: flex; box-shadow: inset 0 1px 2px rgba(0,0,0,.15);
}}
.readiness-band .seg-risk {{ background:{CORAL}; }}
.readiness-band .seg-mid  {{ background:{AMBER}; }}
.readiness-band .seg-ok   {{ background:{GREEN}; }}
.readiness-marker {{
    position: absolute; top: -7px; width: 4px; height: 32px;
    background: {INK}; border-radius: 2px; box-shadow: 0 0 0 2px {PAPER};
}}
.readiness-labels {{
    display:flex; justify-content:space-between;
    font-size: 0.78rem; color:{MUTED}; margin-top: 10px;
}}
</style>
"""


# ----------------------------------------------------------------------------
# Feature engineering & model
# ----------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Academic_Avg"] = df[["10th_Percentage", "12th_Percentage",
                             "Graduation_Percentage"]].mean(axis=1)
    df["Skill_Avg"] = df[SKILL_COLS].mean(axis=1)
    df["Total_Experience"] = (df["Internship_Months"] + df["Projects"]
                              + df["Certifications"])
    df["CGPA_x_Skill"] = df["CGPA"] * df["Skill_Avg"]
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]),
         make_column_selector(dtype_include=np.number)),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]),
         make_column_selector(dtype_exclude=np.number)),
    ])
    model = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1)
    return Pipeline([("prep", preprocessor), ("model", model)])


@st.cache_resource(show_spinner=False)
def train_model():
    """Evaluate on a hold-out split for honest metrics, then refit on all data."""
    df = pd.read_csv(DATA_FILE).drop(
        columns=["Student_ID", "Company_Type", "Placement_Package_LPA"])
    df["Placement_Status"] = (df["Placement_Status"] == "Placed").astype(int)
    df = add_engineered_features(df)

    X, y = df[FINAL_COLS], df["Placement_Status"]

    # --- honest evaluation on unseen data --------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    eval_pipe = build_pipeline()
    eval_pipe.fit(X_tr, y_tr)
    proba_te = eval_pipe.predict_proba(X_te)[:, 1]
    pred_te = (proba_te >= DEFAULT_THRESHOLD).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_te, pred_te),
        "roc_auc": roc_auc_score(y_te, proba_te),
        "precision": precision_score(y_te, pred_te),
        "recall": recall_score(y_te, pred_te),
        "f1": f1_score(y_te, pred_te),
        "confusion": confusion_matrix(y_te, pred_te),
        "n_test": len(y_te),
    }

    # --- final model uses every row --------------------------------------
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    # aggregate feature importances back to raw feature names
    prep = pipeline.named_steps["prep"]
    names = [n.split("__")[-1] for n in prep.get_feature_names_out()]
    raw_names = [n.rsplit("_Yes", 1)[0].rsplit("_No", 1)[0]
                 if n.startswith("Internship_") else n for n in names]
    imp = (pd.Series(pipeline.named_steps["model"].feature_importances_,
                     index=raw_names).groupby(level=0).sum()
           .sort_values(ascending=False))
    metrics["importances"] = imp
    return pipeline, metrics


def predict_proba(pipeline: Pipeline, raw_df: pd.DataFrame) -> np.ndarray:
    """raw_df has RAW_INPUTS columns; engineering happens here."""
    d = add_engineered_features(raw_df)[FINAL_COLS]
    return pipeline.predict_proba(d)[:, 1]


def shap_contributions(pipeline: Pipeline, raw_row: pd.DataFrame) -> pd.Series:
    """SHAP values for one student, mapped back to raw feature names."""
    d = add_engineered_features(raw_row)[FINAL_COLS]
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    X_t = prep.transform(d)
    X_t = X_t.toarray() if hasattr(X_t, "toarray") else np.asarray(X_t)

    sv = shap.TreeExplainer(model).shap_values(X_t)
    arr = np.array(sv)
    if isinstance(sv, list):
        contrib = sv[1][0]
    elif arr.ndim == 3:
        contrib = arr[0, :, 1]
    else:
        contrib = arr[0]

    names = [n.split("__")[-1] for n in prep.get_feature_names_out()]
    s = pd.Series(contrib, index=names)
    # merge the two one-hot Internship columns into a single entry
    internship = s[[n for n in s.index if n.startswith("Internship_")]].sum()
    s = s[[n for n in s.index if not n.startswith("Internship_")]]
    s["Internship"] = internship
    return s.sort_values(key=np.abs, ascending=False)


# ----------------------------------------------------------------------------
# Action plan: realistic "what-if" improvements per feature
# ----------------------------------------------------------------------------
def _improve(row: dict, feature: str):
    """Return (new_row, human_description) for one realistic improvement,
    or None if the feature is already maxed out / not improvable."""
    r = dict(row)
    if feature in SKILL_COLS:
        if r[feature] >= 95:
            return None
        target = min(100, r[feature] + 10)
        r[feature] = target
        label = feature.replace("_", " ")
        return r, f"Raise {label} from {row[feature]} to {target} (practice tests, mock sessions)"
    if feature == "Attendance":
        if r["Attendance"] >= 95:
            return None
        target = min(100, r["Attendance"] + 10)
        r["Attendance"] = target
        return r, f"Lift attendance from {row['Attendance']}% to {target}%"
    if feature == "CGPA":
        if r["CGPA"] >= 9.5:
            return None
        target = round(min(10.0, r["CGPA"] + 0.5), 1)
        r["CGPA"] = target
        return r, f"Improve CGPA from {row['CGPA']} to {target} in coming semesters"
    if feature == "Backlogs":
        if r["Backlogs"] == 0:
            return None
        r["Backlogs"] = 0
        return r, f"Clear all {row['Backlogs']} backlog(s)"
    if feature == "Projects":
        r["Projects"] = row["Projects"] + 2
        return r, f"Build 2 more projects ({row['Projects']} → {r['Projects']})"
    if feature == "Certifications":
        r["Certifications"] = row["Certifications"] + 2
        return r, f"Complete 2 relevant certifications ({row['Certifications']} → {r['Certifications']})"
    if feature in ("Internship", "Internship_Months"):
        if row["Internship"] == "Yes" and row["Internship_Months"] >= 6:
            return None
        r["Internship"] = "Yes"
        r["Internship_Months"] = max(3, row["Internship_Months"] + 3)
        return r, f"Take an internship ({row['Internship_Months']} → {r['Internship_Months']} months)"
    return None  # 10th/12th/Graduation percentages are in the past


# maps engineered / derived SHAP names back to something improvable
DERIVED_TO_RAW = {
    "Skill_Avg": SKILL_COLS,
    "CGPA_x_Skill": ["CGPA"] + SKILL_COLS,
    "Total_Experience": ["Projects", "Certifications", "Internship_Months"],
    "Academic_Avg": ["CGPA"],
}


def action_plan(pipeline, row: dict, contrib: pd.Series, base_proba: float,
                max_items: int = 3):
    """Turn the strongest negative SHAP drivers into what-if simulations."""
    # Expand derived features into their improvable raw components,
    # keeping the order of how much each hurts the prediction.
    candidates, seen = [], set()
    for feat, val in contrib.items():
        if val >= 0:
            continue
        raw_feats = DERIVED_TO_RAW.get(feat, [feat])
        for rf in raw_feats:
            if rf not in seen:
                seen.add(rf)
                candidates.append(rf)

    plans = []
    for feat in candidates:
        step = _improve(row, feat)
        if step is None:
            continue
        new_row, desc = step
        new_p = float(predict_proba(pipeline, pd.DataFrame([new_row]))[0])
        gain = new_p - base_proba
        if gain > 0.005:
            plans.append((desc, new_p, gain))
        if len(plans) >= max_items:
            break
    plans.sort(key=lambda t: -t[2])
    return plans


# ----------------------------------------------------------------------------
# Visual helpers
# ----------------------------------------------------------------------------
def readiness_band(proba: float, threshold: float):
    """Custom three-segment readiness band with a marker at the probability."""
    mid_lo = max(0.0, threshold - 0.15)
    pos = proba * 100
    html = f"""
    <div class="readiness-wrap">
      <div class="readiness-band">
        <div class="seg-risk" style="width:{mid_lo*100:.1f}%"></div>
        <div class="seg-mid"  style="width:{(threshold-mid_lo)*100:.1f}%"></div>
        <div class="seg-ok"   style="width:{(1-threshold)*100:.1f}%"></div>
        <div class="readiness-marker" style="left:calc({pos:.1f}% - 2px)"></div>
      </div>
      <div class="readiness-labels">
        <span>At risk</span><span>Borderline</span><span>Placement-ready</span>
      </div>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


def styled_ax(ax):
    ax.set_facecolor(PAPER)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK)
    return ax


def shap_chart(contrib: pd.Series, top: int = 8):
    s = contrib.head(top).iloc[::-1]
    colors = [GREEN if v > 0 else CORAL for v in s.values]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_facecolor(PAPER)
    styled_ax(ax)
    ax.barh([n.replace("_", " ") for n in s.index], s.values, color=colors,
            height=0.62)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_xlabel("Push toward Not Placed  ←   →  Push toward Placed",
                  color=MUTED, fontsize=9)
    fig.tight_layout()
    return fig


def band_of(p: float, threshold: float) -> str:
    if p >= threshold:
        return "Placement-ready"
    if p >= max(0.0, threshold - 0.15):
        return "Borderline"
    return "At risk"


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-eyebrow">Placement cell · decision support</div>
        <h1 class="hero-title">🎓 Placement Readiness</h1>
        <p class="hero-tag">Predict a student's placement chances, understand
        exactly why, and get a concrete plan to improve them.</p>
        <p style="margin-top:10px; font-size:0.92rem; color:{muted};">
            An end-to-end machine learning project — designed, built &amp;
            deployed by
            <a href="https://www.linkedin.com/in/megha-mishra-9014b336a" target="_blank"
               style="color:{ink}; font-weight:700; text-decoration:none;
                      border-bottom: 2px solid {green};">Megha Mishra</a>
        </p>
        <hr class="hero-rule">
        """.format(muted=MUTED, ink=INK, green=GREEN), unsafe_allow_html=True)

    with st.spinner("Training and validating the model (first load only)…"):
        pipeline, metrics = train_model()

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.header("Decision threshold")
        threshold = st.slider(
            "Flag as 'Placement-ready' when probability ≥", 0.30, 0.80,
            DEFAULT_THRESHOLD, 0.05)
        st.caption("0.50 gave the best accuracy on unseen test data. "
                   "Raise it to be stricter — more students get flagged "
                   "for early help.")
        st.divider()
        st.header("Model card")
        st.caption(f"Random Forest · 300 trees · trained on 10,000 students. "
                   f"Hold-out accuracy **{metrics['accuracy']*100:.1f}%**, "
                   f"ROC-AUC **{metrics['roc_auc']:.3f}**. "
                   "Uses academics, skills and experience only — no gender or "
                   "college demographics.")
        st.divider()
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #E3E0D8;
                        border-radius:12px; padding:14px 16px;">
              <div style="font-size:0.7rem; letter-spacing:2px;
                          text-transform:uppercase; color:{MUTED};
                          font-weight:600; margin-bottom:6px;">Developer</div>
              <div style="font-family:'Fraunces', Georgia, serif;
                          font-size:1.15rem; font-weight:650; color:{INK};">
                  Megha Mishra</div>
              <div style="font-size:0.82rem; color:{MUTED}; margin:2px 0 10px 0;">
                  Machine Learning · Data Science</div>
              <a href="https://www.linkedin.com/in/megha-mishra-9014b336a" target="_blank"
                 style="font-size:0.85rem; color:{GREEN}; font-weight:600;
                        text-decoration:none;">↗ LinkedIn</a>
              &nbsp;&nbsp;
              <a href="https://github.com/meghamishra344-eng" target="_blank"
                 style="font-size:0.85rem; color:{GREEN}; font-weight:600;
                        text-decoration:none;">↗ GitHub Portfolio</a>
            </div>
            """, unsafe_allow_html=True)

    tab_predict, tab_batch, tab_model = st.tabs(
        ["🎯 Single student", "📂 Batch scoring", "📊 Model performance"])

    # ================= TAB 1: single prediction =================
    with tab_predict:
        with st.form("student_form"):
            st.subheader("Academic record")
            a1, a2, a3 = st.columns(3)
            p10 = a1.slider("10th Percentage", 40, 100, 75)
            p12 = a2.slider("12th Percentage", 40, 100, 75)
            grad = a3.slider("Graduation Percentage", 40, 100, 70)
            a4, a5, a6 = st.columns(3)
            cgpa = a4.slider("CGPA", 5.0, 10.0, 7.5, 0.1)
            backlogs = a5.number_input("Backlogs", 0, 15, 0)
            attendance = a6.slider("Attendance %", 50, 100, 80)

            st.subheader("Skill assessments")
            s1, s2, s3 = st.columns(3)
            aptitude = s1.slider("Aptitude Score", 30, 100, 70)
            coding = s2.slider("Coding Score", 30, 100, 65)
            communication = s3.slider("Communication Score", 30, 100, 67)
            s4, s5, s6 = st.columns(3)
            technical = s4.slider("Technical Score", 30, 100, 65)
            mock = s5.slider("Mock Interview Score", 30, 100, 70)
            resume = s6.slider("Resume Score", 30, 100, 70)

            st.subheader("Experience")
            e1, e2, e3, e4 = st.columns(4)
            internship = e1.selectbox("Internship", ["Yes", "No"])
            internship_months = e2.number_input("Internship Months", 0, 24, 3)
            projects = e3.number_input("Projects", 0, 20, 4)
            certifications = e4.number_input("Certifications", 0, 20, 5)

            submitted = st.form_submit_button("Predict placement",
                                              type="primary",
                                              use_container_width=True)

        if submitted:
            row = {
                "10th_Percentage": p10, "12th_Percentage": p12,
                "Graduation_Percentage": grad, "CGPA": cgpa,
                "Backlogs": backlogs, "Attendance": attendance,
                "Internship_Months": internship_months, "Projects": projects,
                "Certifications": certifications, "Aptitude_Score": aptitude,
                "Coding_Score": coding, "Communication_Score": communication,
                "Technical_Score": technical, "Mock_Interview_Score": mock,
                "Resume_Score": resume, "Internship": internship,
            }
            proba = float(predict_proba(pipeline, pd.DataFrame([row]))[0])
            band = band_of(proba, threshold)

            st.divider()
            r1, r2 = st.columns([2.6, 1])
            with r1:
                if band == "Placement-ready":
                    st.balloons()
                    st.success("### 🎉 Congratulations — likely to be PLACED!")
                    st.caption("Keep this momentum going into the interviews. 🚀")
                elif band == "Borderline":
                    st.warning("### 🟡 Borderline — targeted practice will tip the balance")
                else:
                    st.error("### ⚠️ At risk — recommend early intervention")
                readiness_band(proba, threshold)
            with r2:
                st.metric("Placement probability", f"{proba*100:.1f}%")
                st.caption(f"Decision threshold: {threshold:.2f}")

            contrib = shap_contributions(pipeline, pd.DataFrame([row]))

            c1, c2 = st.columns([1.15, 1])
            with c1:
                st.subheader("Why this prediction?")
                st.pyplot(shap_chart(contrib))
            with c2:
                st.subheader("Action plan")
                plans = action_plan(pipeline, row, contrib, proba)
                if not plans:
                    st.info("This profile is already strong across the board — "
                            "no single change moves the needle much. Keep "
                            "scores consistent and prepare for interviews.")
                else:
                    for i, (desc, new_p, gain) in enumerate(plans, 1):
                        st.markdown(
                            f"**{i}. {desc}**  \n"
                            f"→ probability rises to **{new_p*100:.1f}%** "
                            f"(+{gain*100:.1f} pts)")
                    st.caption("Simulated by re-running the model with each "
                               "improvement applied to this student.")

    # ================= TAB 2: batch scoring =================
    with tab_batch:
        st.subheader("Score a whole cohort")
        st.write("Upload a CSV with one row per student. Required columns: "
                 f"`{'`, `'.join(RAW_INPUTS)}`. Extra columns (like "
                 "`Student_ID`) are kept in the output.")
        up = st.file_uploader("Upload student CSV", type=["csv"])
        if up is not None:
            try:
                batch = pd.read_csv(up)
            except Exception as e:
                st.error(f"Could not read that file as a CSV: {e}")
                batch = None
            if batch is not None:
                missing = [c for c in RAW_INPUTS if c not in batch.columns]
                if missing:
                    st.error("Missing required columns: "
                             f"`{'`, `'.join(missing)}`")
                else:
                    probas = predict_proba(pipeline, batch[RAW_INPUTS])
                    out = batch.copy()
                    out["Placement_Probability"] = (probas * 100).round(1)
                    out["Risk_Band"] = [band_of(p, threshold) for p in probas]
                    out = out.sort_values("Placement_Probability")

                    n = len(out)
                    n_risk = int((out["Risk_Band"] == "At risk").sum())
                    n_border = int((out["Risk_Band"] == "Borderline").sum())
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Students scored", n)
                    m2.metric("At risk", n_risk)
                    m3.metric("Borderline", n_border)
                    m4.metric("Placement-ready", n - n_risk - n_border)

                    fig, ax = plt.subplots(figsize=(7.5, 2.6))
                    fig.patch.set_facecolor(PAPER)
                    styled_ax(ax)
                    ax.hist(probas * 100, bins=30, color=MUTED,
                            edgecolor=PAPER)
                    ax.axvline(threshold * 100, color=INK, linestyle="--",
                               linewidth=1)
                    ax.set_xlabel("Placement probability (%)", color=MUTED,
                                  fontsize=9)
                    ax.set_yticks([])
                    fig.tight_layout()
                    st.pyplot(fig)

                    # ---------- search & table ----------
                    st.write("**Most at-risk students first:**")
                    query = st.text_input(
                        "🔎 Search students",
                        placeholder="Type a Student ID (e.g. SP100042) or any text to filter…")
                    view = out
                    if query:
                        hay = (out.fillna("").astype(str)
                               .agg(" ".join, axis=1).str.lower())
                        view = out[hay.str.contains(query.lower(), regex=False)]
                        st.caption(f"{len(view)} student(s) match '{query}'.")
                        if view.empty:
                            st.warning("No student matches that search. "
                                       "Check the ID spelling and try again.")
                    st.dataframe(view, use_container_width=True, height=380)
                    st.download_button(
                        "⬇️ Download scored CSV",
                        out.to_csv(index=False).encode(),
                        file_name="placement_scores.csv", mime="text/csv",
                        use_container_width=True)

                    # ---------- individual student report ----------
                    st.divider()
                    st.subheader("Individual student report")
                    id_col = "Student_ID" if "Student_ID" in view.columns else None
                    if id_col:
                        options = view[id_col].astype(str).tolist()
                    else:
                        options = [f"Row {i}" for i in view.index]
                    if len(options) > 300:
                        st.caption("Showing the first 300 of "
                                   f"{len(options)} students — use the search "
                                   "box above to narrow down.")
                        options = options[:300]
                    if options:
                        picked = st.selectbox("Open a full report for:", options)
                        if id_col:
                            srow = view[view[id_col].astype(str) == picked].iloc[0]
                        else:
                            srow = view.loc[int(picked.split(" ")[1])]
                        raw = {c: srow[c] for c in RAW_INPUTS}
                        raw["Internship"] = str(raw["Internship"])
                        p_i = float(predict_proba(
                            pipeline, pd.DataFrame([raw]))[0])
                        band_i = band_of(p_i, threshold)

                        b1, b2 = st.columns([2.6, 1])
                        with b1:
                            if band_i == "Placement-ready":
                                st.success(f"**{picked}** — likely to be placed 🎉")
                            elif band_i == "Borderline":
                                st.warning(f"**{picked}** — borderline; targeted "
                                           "practice will tip the balance")
                            else:
                                st.error(f"**{picked}** — at risk; recommend "
                                         "early intervention")
                            readiness_band(p_i, threshold)
                        with b2:
                            st.metric("Placement probability", f"{p_i*100:.1f}%")

                        contrib_i = shap_contributions(
                            pipeline, pd.DataFrame([raw]))
                        d1, d2 = st.columns([1.15, 1])
                        with d1:
                            st.markdown("**Why this prediction?**")
                            st.pyplot(shap_chart(contrib_i))
                        with d2:
                            st.markdown("**Action plan**")
                            plans_i = action_plan(pipeline, raw, contrib_i, p_i)
                            if not plans_i:
                                st.info("Already strong across the board — no "
                                        "single change moves the needle much.")
                            else:
                                for i, (desc, new_p, gain) in enumerate(plans_i, 1):
                                    st.markdown(
                                        f"**{i}. {desc}**  \n"
                                        f"→ probability rises to "
                                        f"**{new_p*100:.1f}%** (+{gain*100:.1f} pts)")

    # ================= TAB 3: model performance =================
    with tab_model:
        st.subheader("How good is this model?")
        st.write("All numbers below come from a **20% hold-out test set** "
                 "(2,000 students the model never saw during training) — "
                 "so they reflect real-world performance, not memorisation.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        m2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        m3.metric("Precision", f"{metrics['precision']*100:.1f}%")
        m4.metric("Recall", f"{metrics['recall']*100:.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confusion matrix** (threshold 0.50)")
            cm = metrics["confusion"]
            fig, ax = plt.subplots(figsize=(4.4, 3.8))
            fig.patch.set_facecolor(PAPER)
            ax.imshow(cm, cmap="Greens")
            labels = ["Not Placed", "Placed"]
            ax.set_xticks([0, 1], labels)
            ax.set_yticks([0, 1], labels)
            ax.set_xlabel("Predicted", color=MUTED)
            ax.set_ylabel("Actual", color=MUTED)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                            color=INK, fontsize=13, fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig)
        with c2:
            st.markdown("**What the model relies on most**")
            imp = metrics["importances"].head(10).iloc[::-1]
            fig, ax = plt.subplots(figsize=(5.2, 3.8))
            fig.patch.set_facecolor(PAPER)
            styled_ax(ax)
            ax.barh([n.replace("_", " ") for n in imp.index], imp.values,
                    color=GREEN, height=0.6)
            ax.set_xlabel("Importance", color=MUTED, fontsize=9)
            fig.tight_layout()
            st.pyplot(fig)

        st.caption("The deployed model is refit on all 10,000 rows after "
                   "evaluation, so predictions use every bit of available "
                   "data. Demographic fields (gender, state, college type) "
                   "are deliberately excluded to keep predictions fair.")

    # ---------------- footer ----------------
    st.markdown(
        f"""
        <hr class="hero-rule" style="margin-top:2.2rem">
        <div style="text-align:center; color:{MUTED}; font-size:0.85rem;
                    padding: 4px 0 14px 0;">
            © 2026 <b style="color:{INK}">Megha Mishra</b> — All rights reserved.
            <br><span style="font-size:0.78rem;">
            Conceived, developed &amp; deployed by Megha Mishra
            &nbsp;·&nbsp; Python · scikit-learn · SHAP · Streamlit &nbsp;·&nbsp;
            <a href="https://www.linkedin.com/in/megha-mishra-9014b336a"
               target="_blank" style="color:{GREEN}; text-decoration:none;
               font-weight:600;">LinkedIn</a> &nbsp;·&nbsp;
            <a href="https://github.com/meghamishra344-eng/Student-placement-prediction-APP"
               target="_blank" style="color:{GREEN}; text-decoration:none;
               font-weight:600;">View source on GitHub</a></span>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
