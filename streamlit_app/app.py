import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             precision_score, recall_score, roc_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

st.set_page_config(page_title="3CLpro bioactivity predictor", layout="wide")
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Fetching IC50 records from ChEMBL...")
def fetch_chembl(target_id):
    from chembl_webresource_client.new_client import new_client
    rows = new_client.activity.filter(
        target_chembl_id=target_id, standard_type="IC50"
    ).only(["canonical_smiles", "standard_value", "standard_units", "standard_relation"])
    return pd.DataFrame(list(rows))


def clean_chembl(df):
    df = df.dropna(subset=["canonical_smiles", "standard_value"]).copy()
    df = df[(df["standard_units"] == "nM") & (df["standard_relation"] == "=")]
    df["ic50_nM"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[df["ic50_nM"] > 0].rename(columns={"canonical_smiles": "smiles"})
    return df.groupby("smiles", as_index=False)["ic50_nM"].median()


def label(df, active_cut, inactive_cut):
    df = df[(df["ic50_nM"] <= active_cut) | (df["ic50_nM"] >= inactive_cut)].copy()
    df["active"] = (df["ic50_nM"] <= active_cut).astype(int)
    return df.reset_index(drop=True)


def featurize(smiles, radius, n_bits):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            X.append(gen.GetFingerprintAsNumPy(m))
            keep.append(i)
    return np.array(X, dtype=np.uint8), keep


def properties(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append({
            "Molecular weight": Descriptors.MolWt(m) if m else np.nan,
            "LogP": Descriptors.MolLogP(m) if m else np.nan,
            "H-bond donors": Descriptors.NumHDonors(m) if m else np.nan,
            "H-bond acceptors": Descriptors.NumHAcceptors(m) if m else np.nan,
        })
    return pd.DataFrame(out)

# ------------------------------------------------------------- sidebar
st.sidebar.header("Data")
source = st.sidebar.radio("Source", ["ChEMBL", "Upload CSV"])
target_id = st.sidebar.text_input(
    "ChEMBL target ID", "CHEMBL4523582",
    help="SARS-CoV-2 3C-like proteinase. Check the ID on chembl.org if you change it.")
upload = st.sidebar.file_uploader("CSV with columns smiles, ic50_nM", type="csv") \
    if source == "Upload CSV" else None

st.sidebar.header("Activity labels (IC50, nM)")
active_cut = st.sidebar.number_input("Active at or below", 1, 100000, 1000, step=100)
inactive_cut = st.sidebar.number_input("Inactive at or above", 1, 1000000, 10000, step=1000)

st.sidebar.header("Fingerprint")
radius = st.sidebar.slider("Morgan radius", 1, 4, 2)
n_bits = st.sidebar.select_slider("Bits", [512, 1024, 2048, 4096], value=2048)

st.sidebar.header("Training")
test_size = st.sidebar.slider("Test set share", 0.1, 0.4, 0.2, 0.05)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

# ---------------------------------------------------------------- load
st.title("Predicting SARS-CoV-2 3CLpro inhibitor bioactivity")
st.caption("ChEMBL IC50 data, RDKit Morgan fingerprints, and three classifiers compared side by side.")

try:
    if source == "ChEMBL":
        raw = clean_chembl(fetch_chembl(target_id))
    elif upload is not None:
        raw = pd.read_csv(upload)[["smiles", "ic50_nM"]].dropna()
    else:
        st.info("Upload a CSV with columns smiles and ic50_nM to continue.")
        st.stop()
except Exception as e:
    st.error(f"Could not load data: {e}. Try again, or switch to Upload CSV.")
    st.stop()

if active_cut >= inactive_cut:
    st.error("The active cutoff must be lower than the inactive cutoff.")
    st.stop()

df = label(raw, active_cut, inactive_cut)
if df["active"].nunique() < 2 or len(df) < 30:
    st.error("Too few labelled molecules. Widen the IC50 cutoffs or use a different dataset.")
    st.stop()

tab_data, tab_feat, tab_model, tab_pred = st.tabs(
    ["Data", "Molecular features", "Models", "Predict a molecule"])

# ---------------------------------------------------------------- data
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Labelled molecules", len(df))
    c2.metric("Active", int(df["active"].sum()))
    c3.metric("Inactive", int((1 - df["active"]).sum()))
    st.caption("Molecules with IC50 between the two cutoffs are left out as ambiguous.")
    st.dataframe(df.assign(label=df["active"].map({1: "active", 0: "inactive"}))
                 .drop(columns="active"), use_container_width=True, height=320)
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_df = pd.DataFrame({
        "logIC50": np.log10(df["ic50_nM"].to_numpy()),
        "label": df["active"].map({1: "active", 0: "inactive"}).to_numpy(),
    })
    sns.histplot(plot_df, x="logIC50", bins=30, hue="label", ax=ax)
    ax.set_xlabel("log10 IC50 (nM)")
    st.pyplot(fig)

# ------------------------------------------------------------ features
with tab_feat:
    props = properties(df["smiles"])
    props["Class"] = df["active"].map({1: "active", 0: "inactive"})
    choice = st.selectbox("Property", [c for c in props.columns if c != "Class"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.boxplot(data=props, x="Class", y=choice, ax=ax)
    st.pyplot(fig)

# -------------------------------------------------------------- models
with tab_model:
    chosen = st.multiselect("Classifiers", ["Random Forest", "XGBoost", "LightGBM"],
                            default=["Random Forest", "XGBoost", "LightGBM"])
    if st.button("Train and evaluate", type="primary") and chosen:
        X, keep = featurize(df["smiles"], radius, n_bits)
        y = df["active"].iloc[keep].to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=int(seed))
        makers = {
            "Random Forest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=int(seed)),
            "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                             eval_metric="logloss", random_state=int(seed)),
            "LightGBM": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.1,
                                               random_state=int(seed), verbose=-1),
        }
        models, rows, curves = {}, [], {}
        with st.spinner("Training..."):
            for name in chosen:
                m = makers[name]().fit(Xtr, ytr)
                p = m.predict_proba(Xte)[:, 1]
                pred = (p >= 0.5).astype(int)
                models[name] = m
                curves[name] = roc_curve(yte, p)
                rows.append({"Model": name, "AUC-ROC": roc_auc_score(yte, p),
                             "F1": f1_score(yte, pred), "Accuracy": accuracy_score(yte, pred),
                             "Precision": precision_score(yte, pred), "Recall": recall_score(yte, pred)})
        st.session_state["res"] = dict(models=models, table=pd.DataFrame(rows),
                                       curves=curves, radius=radius, n_bits=n_bits)

    res = st.session_state.get("res")
    if res:
        st.dataframe(res["table"].set_index("Model").style.format("{:.3f}").highlight_max(axis=0),
                     use_container_width=True)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for name, (fpr, tpr, _) in res["curves"].items():
            ax.plot(fpr, tpr, label=name)
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Pick classifiers and press Train and evaluate.")

# ------------------------------------------------------------- predict
with tab_pred:
    res = st.session_state.get("res")
    smi = st.text_input("SMILES", "CC(C)C[C@H](NC(=O)OCc1ccccc1)C(=O)N")
    if not res:
        st.info("Train the models first (Models tab).")
    elif smi:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            st.error("That SMILES string could not be parsed.")
        else:
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=res["radius"], fpSize=res["n_bits"])
            x = gen.GetFingerprintAsNumPy(mol).reshape(1, -1)
            left, right = st.columns([1, 2])
            left.image(Draw.MolToImage(mol, size=(300, 300)))
            probs = {n: float(m.predict_proba(x)[0, 1]) for n, m in res["models"].items()}
            for n, p in probs.items():
                right.metric(f"{n}: probability active", f"{p:.1%}")
            right.caption("Probabilities come from models trained on the cutoffs and fingerprint settings used at training time.")
