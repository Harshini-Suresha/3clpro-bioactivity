#!/usr/bin/env python3
"""Run the 3CLpro bioactivity pipeline on REAL data and write results.js / results.json
for the website. Every number and plot on the page comes from this file.

    pip install rdkit scikit-learn xgboost lightgbm shap pandas numpy chembl_webresource_client
    python export_results.py                      # pulls IC50 data from ChEMBL
    python export_results.py --csv my_data.csv    # or use your own CSV (columns: smiles, ic50_nM, optional id)
    python export_results.py --fast               # smaller grids, quicker run

Keep results.js next to index.html and deploy both.
"""
import argparse, datetime, json
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdDepictor, rdFingerprintGenerator, rdMolDescriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters import rdSimDivPickers
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RDLogger.DisableLog("rdApp.*")
ap = argparse.ArgumentParser()
ap.add_argument("--target", default="CHEMBL4523582", help="ChEMBL target ID (verify on chembl.org)")
ap.add_argument("--csv"); ap.add_argument("--out", default=".")
ap.add_argument("--active", type=float, default=1000); ap.add_argument("--inactive", type=float, default=10000)
ap.add_argument("--radius", type=int, default=2); ap.add_argument("--bits", type=int, default=2048)
ap.add_argument("--test-size", type=float, default=0.2); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fast", action="store_true")
ap.add_argument("--inline", action="store_true", help="also embed the results inside index.html so it works as a single file")
a = ap.parse_args()
rng = np.random.RandomState(a.seed)
r3 = lambda x, n=3: [round(float(v), n) for v in x]

# ---------------------------------------------------------------- data
if a.csv:
    df = pd.read_csv(a.csv)
    if "id" not in df: df["id"] = [f"mol{i}" for i in range(len(df))]
    df = df[["id", "smiles", "ic50_nM"]].dropna()
else:
    from chembl_webresource_client.new_client import new_client
    rows = new_client.activity.filter(target_chembl_id=a.target, standard_type="IC50").only(
        ["molecule_chembl_id", "canonical_smiles", "standard_value", "standard_units", "standard_relation"])
    df = pd.DataFrame(list(rows)).dropna(subset=["canonical_smiles", "standard_value"])
    df = df[(df.standard_units == "nM") & (df.standard_relation == "=")]
    df["ic50_nM"] = pd.to_numeric(df.standard_value, errors="coerce")
    df = df[df.ic50_nM > 0].rename(columns={"molecule_chembl_id": "id", "canonical_smiles": "smiles"})
    df = df.groupby("smiles", as_index=False).agg(id=("id", "first"), ic50_nM=("ic50_nM", "median"))
df["mol"] = df.smiles.apply(Chem.MolFromSmiles)
df = df[df.mol.notna()].reset_index(drop=True)
df["l"] = np.log10(df.ic50_nM)
print(f"{len(df)} curated molecules")

gen = rdFingerprintGenerator.GetMorganGenerator(radius=a.radius, fpSize=a.bits)
def fp_info(m):
    ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
    f = gen.GetFingerprint(m, additionalOutput=ao)
    return f, ao.GetBitInfoMap()
def env_smiles(m, at, r):
    if r == 0: return m.GetAtomWithIdx(at).GetSymbol()
    env = Chem.FindAtomEnvironmentOfRadiusN(m, r, at)
    return Chem.MolToSmiles(Chem.PathToSubmol(m, env)) if env else ""

lab = df[(df.ic50_nM <= a.active) | (df.ic50_nM >= a.inactive)].reset_index(drop=True)
lab["y"] = (lab.ic50_nM <= a.active).astype(int)
X = np.array([gen.GetFingerprintAsNumPy(m) for m in lab.mol], dtype=np.uint8); y = lab.y.values
print(f"{len(lab)} labelled: {y.sum()} active, {len(y)-y.sum()} inactive")
itr, ite = train_test_split(np.arange(len(y)), test_size=a.test_size, stratify=y, random_state=a.seed)
Xtr, Xte, ytr, yte = X[itr], X[ite], y[itr], y[ite]

def make(name, **k):
    if name == "Random Forest": return RandomForestClassifier(n_estimators=k.get("n", 300), max_depth=k.get("d"), n_jobs=-1, random_state=a.seed)
    if name == "XGBoost": return XGBClassifier(n_estimators=k.get("n", 300), max_depth=k.get("d", 6), learning_rate=0.1, eval_metric="logloss", random_state=a.seed, n_jobs=-1)
    return LGBMClassifier(n_estimators=k.get("n", 300), max_depth=k.get("d", -1), learning_rate=0.1, random_state=a.seed, verbose=-1)
NAMES = ["Random Forest", "XGBoost", "LightGBM"]

# ---------------------------------------------------------------- test-set predictions
models, fitted = [], {}
for n in NAMES:
    m = make(n).fit(Xtr, ytr); fitted[n] = m
    models.append({"name": n, "scores": r3(m.predict_proba(Xte)[:, 1], 4), "y": [int(v) for v in yte]})
    print(n, "test AUC", round(roc_auc_score(yte, models[-1]["scores"]), 3))

# ---------------------------------------------------------------- learning curve (5-fold CV on the training set)
fracs = [round(f, 2) for f in np.arange(0.1, 1.01, 0.1)]
skf = StratifiedKFold(5, shuffle=True, random_state=a.seed)
lc = {"sizes": fracs, "curves": {}, "folds": {}}
for n in NAMES:
    c = {k: [] for k in ["auc_val", "auc_val_sd", "auc_tr", "f1_val", "f1_val_sd", "f1_tr"]}
    for f in fracs:
        av, fv, at, ft = [], [], [], []
        for tr, va in skf.split(Xtr, ytr):
            sub = tr if f >= 1 else train_test_split(tr, train_size=f, stratify=ytr[tr], random_state=a.seed)[0]
            m = make(n).fit(Xtr[sub], ytr[sub]); pv, pt = m.predict_proba(Xtr[va])[:, 1], m.predict_proba(Xtr[sub])[:, 1]
            av.append(roc_auc_score(ytr[va], pv)); fv.append(f1_score(ytr[va], pv >= .5)); at.append(roc_auc_score(ytr[sub], pt)); ft.append(f1_score(ytr[sub], pt >= .5))
        for k, v in zip(["auc_val", "f1_val", "auc_tr", "f1_tr"], [av, fv, at, ft]): c[k].append(float(np.mean(v)))
        c["auc_val_sd"].append(float(np.std(av))); c["f1_val_sd"].append(float(np.std(fv)))
        if f >= 1: lc["folds"][n] = {"auc": r3(av), "f1": r3(fv)}
    lc["curves"][n] = {k: r3(v, 4) for k, v in c.items()}
    print("learning curve", n)

# ---------------------------------------------------------------- tuning grid (5-fold CV AUC)
NE = [50, 100, 200] if a.fast else [50, 100, 200, 300, 500]
DE = [2, 6, 10] if a.fast else [2, 4, 6, 8, 10]
tuning = {"n": NE, "depth": DE, "grid": {}}
for n in NAMES:
    mean = [[0] * len(NE) for _ in DE]; sd = [[0] * len(NE) for _ in DE]
    for i, d in enumerate(DE):
        for j, ne in enumerate(NE):
            s = [roc_auc_score(ytr[va], make(n, n=ne, d=d).fit(Xtr[tr], ytr[tr]).predict_proba(Xtr[va])[:, 1]) for tr, va in skf.split(Xtr, ytr)]
            mean[i][j], sd[i][j] = round(float(np.mean(s)), 4), round(float(np.std(s)), 4)
    tuning["grid"][n] = {"mean": mean, "sd": sd}
    print("tuning", n)

# ---------------------------------------------------------------- chemical space (PCA of fingerprints, k-means chemotypes, Tanimoto neighbours)
idx = np.sort(rng.choice(len(df), min(800, len(df)), replace=False))
sub = df.iloc[idx].reset_index(drop=True)
fps = [gen.GetFingerprint(m) for m in sub.mol]
Xs = np.array([gen.GetFingerprintAsNumPy(m) for m in sub.mol], dtype=np.float32)
pca = PCA(2, random_state=a.seed).fit(Xs); P = pca.transform(Xs)
km = KMeans(7, n_init=10, random_state=a.seed).fit(PCA(min(30, Xs.shape[1], len(sub)-1), random_state=a.seed).fit_transform(Xs))
nn = []
for i, f in enumerate(fps):
    s = np.array(DataStructs.BulkTanimotoSimilarity(f, fps)); s[i] = -1
    top = np.argsort(-s)[:5]; nn.append([[int(j), round(float(s[j]), 3)] for j in top])
space = {"id": list(sub.id), "x": r3(P[:, 0]), "y": r3(P[:, 1]), "l": r3(sub.l), "k": [int(v) for v in km.labels_], "nn": nn,
         "var": r3(pca.explained_variance_ratio_, 4)}

# ---------------------------------------------------------------- example molecules, real fingerprints and pairwise Tanimoto
top = df.sort_values("ic50_nM").head(80).reset_index(drop=True)
tf = [gen.GetFingerprint(m) for m in top.mol]
pick = list(rdSimDivPickers.MaxMinPicker().LazyBitVectorPick(tf, len(tf), min(8, len(tf)), seed=a.seed))
ex = []
for i in pick:
    m = top.mol[i]; f, info = fp_info(m)
    d = Chem.Mol(m); rdDepictor.Compute2DCoords(d); Chem.Kekulize(d, clearAromaticFlags=True); cf = d.GetConformer()
    envs = []
    for bit, lst in info.items():
        for at, r in lst:
            an = [at] if r == 0 else sorted({at} | {x for b in Chem.FindAtomEnvironmentOfRadiusN(m, r, at) for x in (m.GetBondWithIdx(b).GetBeginAtomIdx(), m.GetBondWithIdx(b).GetEndAtomIdx())})
            envs.append({"bit": int(bit), "atom": int(at), "r": int(r), "atoms": an})
    ex.append({"id": top.id[i], "l": round(float(top.l[i]), 3), "smiles": top.smiles[i],
               "atoms": [[round(cf.GetAtomPosition(k).x, 3), round(cf.GetAtomPosition(k).y, 3), d.GetAtomWithIdx(k).GetSymbol()] for k in range(d.GetNumAtoms())],
               "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(round(b.GetBondTypeAsDouble()))] for b in d.GetBonds()],
               "envs": envs, "on": sorted(int(b) for b in f.GetOnBits())})
fx = [tf[i] for i in pick]
sim = [[round(DataStructs.TanimotoSimilarity(p, q), 3) for q in fx] for p in fx]

# ---------------------------------------------------------------- molecular properties (all curated molecules, sampled)
pn = ["Molecular weight", "LogP", "H-bond donors", "H-bond acceptors", "TPSA", "Rotatable bonds", "Ring count", "Heavy atoms",
      "Fraction CSP3", "Aromatic rings", "Heterocycles", "QED"]
pf = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.NumHDonors, Descriptors.NumHAcceptors, rdMolDescriptors.CalcTPSA,
      Descriptors.NumRotatableBonds, rdMolDescriptors.CalcNumRings, lambda m: m.GetNumHeavyAtoms(),
      Descriptors.FractionCSP3, rdMolDescriptors.CalcNumAromaticRings, rdMolDescriptors.CalcNumHeterocycles, QED.qed]
pidx = np.sort(rng.choice(len(df), min(1500, len(df)), replace=False))
props = {"names": pn, "l": r3(df.l.iloc[pidx]), "vals": [[round(float(f(df.mol.iloc[i])), 2) for f in pf] for i in pidx]}

# ---------------------------------------------------------------- full browse table (every curated molecule, real data)
def _mw(m):
    try: return round(float(Descriptors.MolWt(m)), 2)
    except Exception: return None
def _logp(m):
    try: return round(float(Descriptors.MolLogP(m)), 2)
    except Exception: return None
full = {"id": [str(v) for v in df.id], "smiles": list(df.smiles), "l": r3(df.l),
        "mw": [_mw(m) for m in df.mol], "logp": [_logp(m) for m in df.mol]}

# ---------------------------------------------------------------- scaffolds and random vs scaffold split
def scaf(m):
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception: return ""
lab["scaf"] = lab.mol.apply(scaf)
g = lab.groupby("scaf").agg(n=("y", "size"), act=("y", "sum")).reset_index()
nz = g[g.scaf != ""].sort_values("n", ascending=False)
scaf_out = {"top": [{"smi": r.scaf, "n": int(r.n), "act": int(r.act)} for r in nz.head(12).itertuples()],
            "n_scaffolds": int(len(nz)), "n_singletons": int((nz.n == 1).sum()), "n_acyclic": int(g[g.scaf == ""].n.sum())}
groups = np.array([s_ if s_ else f"acyclic{i}" for i, s_ in enumerate(lab.scaf)])
split = {n: {k: {"auc": [], "f1": []} for k in ("random", "scaffold")} for n in NAMES}
for kind, spl in [("random", StratifiedShuffleSplit(5, test_size=a.test_size, random_state=a.seed).split(X, y)),
                  ("scaffold", GroupShuffleSplit(5, test_size=a.test_size, random_state=a.seed).split(X, y, groups))]:
    for tr, te in spl:
        if len(set(y[te])) < 2 or len(set(y[tr])) < 2: continue
        for n in NAMES:
            p = make(n).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            split[n][kind]["auc"].append(round(float(roc_auc_score(y[te], p)), 4)); split[n][kind]["f1"].append(round(float(f1_score(y[te], p >= .5)), 4))
    print("split comparison", kind)

# ---------------------------------------------------------------- SHAP (XGBoost)
shap_out = None
try:
    import shap
    ns = min(300, len(X)); si = rng.choice(len(X), ns, replace=False)
    sv = shap.TreeExplainer(fitted["XGBoost"]).shap_values(X[si].astype(np.float32))
    top_bits = np.argsort(-np.abs(sv).mean(0))[:20]
    feats = []
    for b in top_bits:
        frag = ""
        for k in np.where(X[:, b] == 1)[0][:50]:
            m = lab.mol[k]; _, info = fp_info(m)
            if int(b) in info:
                at, r = info[int(b)][0]; frag = env_smiles(m, at, r); break
        feats.append({"bit": int(b), "frag": frag, "mean_abs": round(float(np.abs(sv[:, b]).mean()), 4), "vals": r3(sv[:, b]), "present": [int(v) for v in X[si, b]]})
    shap_out = {"model": "XGBoost", "n": int(ns), "features": feats}
except Exception as e:
    print("SHAP skipped:", e)

res = {"meta": {"target": a.target if not a.csv else a.csv, "n_curated": int(len(df)), "n_labelled": int(len(lab)), "n_active": int(y.sum()), "n_inactive": int(len(y) - y.sum()),
                "n_test": int(len(yte)), "active_nM": a.active, "inactive_nM": a.inactive, "radius": a.radius, "bits": a.bits, "seed": a.seed,
                "source": "csv" if a.csv else "ChEMBL", "generated": datetime.date.today().isoformat()},
       "ic50": r3(df.l), "models": models, "lc": lc, "tuning": tuning, "space": space, "ex": ex, "sim": sim, "shap": shap_out,
       "props": props, "scaf": scaf_out, "split": split, "full": full,
       "test": {"id": [str(v) for v in lab.id.iloc[ite]], "smiles": [str(v) for v in lab.smiles.iloc[ite]], "l": r3(lab.l.iloc[ite])}}
js = json.dumps(res, separators=(",", ":"))
open(f"{a.out}/results.json", "w").write(js); open(f"{a.out}/results.js", "w").write("window.RESULTS=" + js + ";\n")
print(f"Wrote results.js and results.json ({len(js)/1024:.0f} KB)")
if a.inline:
    import re, os
    p = f"{a.out}/index.html"
    if os.path.exists(p):
        html = open(p).read(); inl = '<script id="inline-results">window.RESULTS=' + js + ';</script>'
        if 'id="inline-results"' in html: html = re.sub(r'<script id="inline-results">.*?</script>', lambda m: inl, html, flags=re.S)
        else: html = html.replace('<script src="results.js"></script>', inl, 1)
        open(p, "w").write(html); print("Embedded results into index.html")
