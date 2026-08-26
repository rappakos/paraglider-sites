---
name: model-inventory
description: Report which sites have which trained model artifacts, and which are missing
context: fork
allowed-tools: Bash(ls:*), Bash(python:*), Read, Glob
argument-hint: [site-name (optional, default: all sites)]
---

Inventory the trained models in `glider_sites_app/analysis/models/`.

1. Glob the `.joblib` files and parse each name as `<Site>_<kind>.joblib`
   (kinds: bayesian_model, classifier_rf_model, ensemble_model, svm_model,
   prior_counts). Site names may contain non-ASCII characters (Börry, Königszinne).
2. Query the DB for the full site list:
   `python -c "..."` using the read pattern from the repo, or simply glob and
   compare against the sites the models imply.
3. Produce a compact matrix: rows = sites, columns = model kinds, ✓/– cells,
   plus each file's modification date.
4. End with: which site/kind combinations are missing, and the exact
   `retrain_model.sh <Site>` commands to fill the gaps.

If $ARGUMENTS names a site, restrict the report to that site.