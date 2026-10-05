# 3CLpro bioactivity project page

Every plot and number on this page is drawn from `results.js`, which is produced by running the real pipeline.
Until you generate it, the page shows a notice and no plots (nothing is made up).

## 1. Generate the real results
```bash
pip install rdkit scikit-learn xgboost lightgbm shap pandas numpy chembl_webresource_client
python export_results.py              # pulls IC50 data for the ChEMBL target and runs everything
python export_results.py --fast       # smaller tuning grid, quicker
python export_results.py --inline     # also embeds the results inside index.html (works as one file, e.g. in a file viewer)
python export_results.py --csv my.csv # or use your own CSV with columns: smiles, ic50_nM (optional: id)
```
- Default target is `CHEMBL4523582` (SARS-CoV-2 3C-like proteinase). Confirm it on chembl.org, or pass `--target`.
- Cleaning matches the notebook and Streamlit app: exact nM values, duplicates averaged, active <= 1000 nM, inactive >= 10000 nM, ambiguous molecules dropped.
- It writes `results.js` (used by the page) and `results.json`. Expect several minutes; the learning curves and tuning grid run many cross-validated fits.

## 2. Check it
Open `index.html` in a browser. The red "No results loaded" bar should be gone and all plots filled.

## 3. Deploy
Set `CONFIG` near the bottom of `index.html` (Streamlit app URL, GitHub repo URL), then deploy this whole folder to Vercel.
`index.html`, `results.js` and `3clpro_bioactivity.ipynb` must stay together.

## What comes from where
| Section | Source |
|---|---|
| Cutoff swarm, density, table | all curated IC50 values |
| Property distributions | RDKit descriptors of the curated molecules, split by the live cutoffs |
| Fingerprints, similarity | RDKit Morgan fingerprints of real compounds (8 diverse picks) |
| Chemical space | PCA and k-means on fingerprints, Tanimoto neighbours |
| Classifier, calibration | held-out test-set predictions of RF, XGBoost, LightGBM |
| Error analysis | held-out test compounds the models misjudge most confidently |
| Random vs scaffold split, scaffolds | repeated splits plus Murcko scaffolds of the labelled molecules |
| Learning curves, tuning | 5-fold cross-validation on the training set |
| Attribution | SHAP TreeExplainer on XGBoost |
| IC50 curve, accuracy calculator | explanatory maths, labelled as such |
