---
name: model-inventory
description: Report which sites have which trained model artifacts, and which are missing
context: fork
allowed-tools: Bash(ls:*), Bash(python:*), Read, Glob
argument-hint: [site-name (optional, default: all sites)]
---

Inventory the trained models in `glider_sites_app/analysis/models/`.

1. Glob `glider_sites_app/analysis/models/*.joblib`. Parse each filename by stripping
   one of the known kind suffixes off the END of the stem — kinds: bayesian_model,
   classifier_rf_model, ensemble_model, svm_model, prior_counts. Whatever remains is
   the site name; it may itself contain underscores (Rammelsberg_NW) and non-ASCII
   characters (Börry, Königszinne).
2. Get the authoritative site list from the DB (mandatory — a site with zero
   artifacts leaves no filename to glob):
   `python -c "import asyncio; from glider_sites_app.repositories.sites_repository import get_stats; print(asyncio.run(get_stats())[['site_name']])"`
   Run from the repo root with venv active. No inline SQL.
3. Produce a compact matrix: rows = sites, columns = model kinds, ✓/– cells,
   plus each file's modification date.
4. End with: which site/kind combinations are missing, and the exact
   `retrain_model.sh "<Site>"` commands to fill the gaps.

If $ARGUMENTS names a site, restrict the report to that site.