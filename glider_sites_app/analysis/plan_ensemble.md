# Plan: Ensemble (RF Classifier + SVR)

## Overview

Combine the RF classifier (`rf_prob` from `predict_proba`) and the SVR regressor
(log-count converted to probability via a sigmoid) using a weighted blend:

```
ensemble_prob = alpha * rf_prob + (1 - alpha) * svm_prob
```

The optimal `alpha` is found **per site** by sweeping over out-of-fold predictions
and maximising CV F1.  The fitted alpha + both trained models are saved together
as a single ensemble artefact.  Service integration is deferred.

---

## Phase A — New `glider_sites_app/analysis/ensemble.py`

- [x] **`svm_to_prob(log_count, threshold, scale=0.5)`** helper
  - `sigmoid((log_count - threshold) / scale)`
  - `scale` controls steepness: at `scale=0.5`, one log-unit spans roughly 0.12 → 0.88
  - `threshold` defaults to `FLYABLE_THRESHOLD` imported from `svm.py` (single source of truth)

- [x] **`_collect_oof_predictions(site_name)`** private async function
  - `prepare_training_data(site_name)` → same 7 features
  - Ground truth: `y = flight_count > 0` (binary, matches RF training target)
  - `KFold(k_folds, shuffle=True, random_state=SEED)` — same folds as RF/SVM
  - Per fold: fit RF classifier + SVM pipeline on train split, predict on val split
  - Returns `(y_true, rf_proba_oof, svm_log_oof)` — three aligned arrays

- [x] **`scan_ensemble_alpha(site_name, alphas=np.arange(0, 1.05, 0.05))`** async function
  - Calls `_collect_oof_predictions` once; reuses OOF arrays for all alpha values
  - For each alpha: `ensemble_prob = alpha * rf_prob + (1-alpha) * svm_prob(svm_log)`
    - Decision boundary at 0.5; compute F1
  - Returns DataFrame `[alpha, mean_f1, std_f1]` sorted by `mean_f1` desc; logs full table
  - Also logs standalone RF F1 (alpha=1.0) and SVM F1 (alpha=0.0) for reference

- [x] **`train_ensemble(site_name, save=False)`** async function
  - Calls `scan_ensemble_alpha` → picks best `alpha`
  - Logs: best alpha + ensemble F1 vs standalone RF F1 vs standalone SVM F1
  - Fits final RF classifier + SVM pipeline each on **full** dataset
  - Returns dict:
    ```python
    {
        'site_name': site_name,
        'rf_model': rf_pipeline,
        'svm_model': svm_pipeline,
        'alpha': best_alpha,
        'sigmoid_scale': SIGMOID_SCALE,
        'flyable_threshold': FLYABLE_THRESHOLD,
        'feature_importance_rf': importance_df,
        'feature_importance_svm': importance_df,
    }
    ```
  - If `save=True`: calls `save_ensemble_results(site_name, result)`

- [x] **`predict_ensemble(ensemble_data, X)`** pure function (sync, for use in service)
  - `rf_prob = ensemble_data['rf_model'].predict_proba(X)[:, 1]`
  - `svm_log = ensemble_data['svm_model'].predict(X)`
  - `svm_prob = svm_to_prob(svm_log, ...)`
  - `ensemble_prob = alpha * rf_prob + (1 - alpha) * svm_prob`
  - Returns DataFrame with columns: `ensemble_prob`, `is_flyable`, `rf_prob`, `svm_prob`

- [x] **`__main__` block**
  - `train_ensemble('Rammelsberg NW', save=True)`
  - `scan_ensemble_alpha('Rammelsberg NW')` (prints alpha sweep table)

---

## Phase B — Update `glider_sites_app/analysis/model_loader.py`

- [x] `get_ensemble_model_path(site_name)` → `models/{site_name_clean}_ensemble_model.joblib`
- [x] `save_ensemble_results(site_name, results)` → `joblib.dump(results, path)`
- [x] `load_ensemble_model(site_name)` → returns full dict or `None` if file missing
- Note: no changes to existing RF or SVM loader functions

---

## Phase C — Verification

- [ ] `python -m glider_sites_app.analysis.ensemble` runs for Rammelsberg NW
  - Prints alpha sweep table
  - Logs best alpha + ensemble F1 vs standalone RF and SVM
- [ ] Ensemble F1 ≥ max(RF F1, SVM F1) — sanity check; if alpha=1.0 wins, RF dominates and ensemble adds no value
- [ ] `models/Rammelsberg_NW_ensemble_model.joblib` created
- [ ] `load_ensemble_model('Rammelsberg NW')` returns dict with keys `rf_model`, `svm_model`, `alpha`
- [ ] `predict_ensemble(data, X)` returns DataFrame with correct columns and shapes

---

## Phase D — Service integration (deferred)

> Wire ensemble into `site_service.py` once Phase A–C are stable.

- [ ] Update `get_forecast_data()` to load ensemble model via `load_ensemble_model()`
- [ ] Replace standalone RF predict/predict_proba calls with `predict_ensemble()`
- [ ] Pass `ensemble_prob` (instead of `rf_confidence`) into the Bayesian Network's
      `RF_Flyability_Confidence` discretisation
- [ ] Update `get_site_data()` to expose ensemble feature importances
- [ ] Invalidate and update forecast cache after rollout

---

## Key decisions

| Decision | Choice |
|---|---|
| SVM → prob conversion | Sigmoid centred at `FLYABLE_THRESHOLD`, `scale=0.5` |
| Weight optimisation | Per-site alpha sweep over OOF, maximise F1 at decision boundary 0.5 |
| `FLYABLE_THRESHOLD` | Imported from `svm.py` — shared constant between SVM and ensemble |
| Service integration | Deferred — saved `.joblib` is the handoff artefact |

---

## Further considerations

1. **Sigmoid scale**: `scale=0.5` is a reasonable default but could be swept alongside
   `alpha` in a 2D grid.  Start with fixed scale; revisit if the ensemble gain is small.

2. **Global alpha**: if per-site alpha values cluster around the same value, a single
   global alpha may generalise better to new sites with little data.

3. **Bayes integration**: once wired in, the `RF_Flyability_Confidence` node in the
   Bayesian Network should use `ensemble_prob` instead of raw `rf_confidence`.  The
   discretisation boundaries (Low/Medium/High) may need recalibration.
