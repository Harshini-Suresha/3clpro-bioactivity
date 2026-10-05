#!/usr/bin/env python3
"""
Radius sweep experiment for 3CLpro bioactivity prediction.
Tests Morgan fingerprint radii 1-4 and compares model performance.
"""

import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from chembl_webresource_client.new_client import new_client

TARGET_ID = "CHEMBL4523582"
ACTIVE_NM, INACTIVE_NM = 1000, 10000
N_BITS = 2048
SEED = 42
RADII = [1, 2, 3, 4]

def fetch_and_clean():
    print(f"Fetching IC50 data for {TARGET_ID}...")
    records = new_client.activity.filter(
        target_chembl_id=TARGET_ID, standard_type="IC50"
    ).only(["canonical_smiles", "standard_value", "standard_units", "standard_relation"])
    raw = pd.DataFrame(list(records))
    print(f"Raw records: {len(raw)}")

    df = raw.dropna(subset=["canonical_smiles", "standard_value"]).copy()
    df = df[(df["standard_units"] == "nM") & (df["standard_relation"] == "=")]
    df["ic50_nM"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[df["ic50_nM"] > 0].rename(columns={"canonical_smiles": "smiles"})
    df = df.groupby("smiles", as_index=False)["ic50_nM"].median()
    df = df[(df["ic50_nM"] <= ACTIVE_NM) | (df["ic50_nM"] >= INACTIVE_NM)].copy()
    df["active"] = (df["ic50_nM"] <= ACTIVE_NM).astype(int)
    df = df.reset_index(drop=True)
    print(f"Labelled molecules: {len(df)} (active: {df['active'].sum()}, inactive: {(1-df['active']).sum()})")
    return df

def featurize(smiles, radius, n_bits):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            X.append(gen.GetFingerprintAsNumPy(m))
            keep.append(i)
    return np.array(X, dtype=np.uint8), keep

def evaluate_models(Xtr, Xte, ytr, yte, seed):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                  eval_metric="logloss", random_state=seed),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.1,
                                    random_state=seed, verbose=-1),
    }
    results = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        results[name] = {
            "AUC-ROC": float(roc_auc_score(yte, p)),
            "F1": float(f1_score(yte, pred)),
            "Accuracy": float(accuracy_score(yte, pred)),
            "Precision": float(precision_score(yte, pred)),
            "Recall": float(recall_score(yte, pred)),
        }
    return results

def main():
    df = fetch_and_clean()
    smiles = df["smiles"].tolist()
    y_all = df["active"].to_numpy()

    all_results = {"meta": {
        "target": TARGET_ID,
        "n_labelled": int(len(df)),
        "n_active": int(df["active"].sum()),
        "n_inactive": int((1-df["active"]).sum()),
        "active_nM": ACTIVE_NM,
        "inactive_nM": INACTIVE_NM,
        "bits": N_BITS,
        "seed": SEED,
        "source": "ChEMBL"
    }, "radii": {}}

    for radius in RADII:
        print(f"\n=== Radius {radius} ===")
        X, keep = featurize(smiles, radius, N_BITS)
        y = y_all[keep]
        print(f"Valid molecules: {len(X)}")

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED
        )
        print(f"Train: {len(Xtr)}, Test: {len(Xte)}")

        results = evaluate_models(Xtr, Xte, ytr, yte, SEED)
        all_results["radii"][str(radius)] = {
            "n_valid": int(len(X)),
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
            "models": results
        }

        for model_name, metrics in results.items():
            print(f"  {model_name}: AUC-ROC={metrics['AUC-ROC']:.3f}, F1={metrics['F1']:.3f}")

    output_path = "radius_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n=== Summary ===")
    for radius in RADII:
        r = all_results["radii"][str(radius)]
        best_auc = max(r["models"].values(), key=lambda x: x["AUC-ROC"])
        best_model = [k for k, v in r["models"].items() if v["AUC-ROC"] == best_auc["AUC-ROC"]][0]
        print(f"Radius {radius}: Best={best_model} AUC-ROC={best_auc['AUC-ROC']:.3f}")

if __name__ == "__main__":
    main()